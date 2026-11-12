from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .comfy_client import (
    ComfyApiError,
    ComfyUIClient,
    history_entry_error,
    history_entry_is_complete,
)
from .config import Settings
from .params import (
    InvalidParameterError,
    apply_img2img_resize_params,
    normalize_standard_params,
    resolve_standard_overrides,
)
from .registry import WorkflowEntry, WorkflowRegistry
from .workflow import (
    KIND_IMG2IMG,
    iter_file_outputs,
    normalize_prompt_enum_inputs,
    prepare_prompt,
    prune_invalid_orphan_output_nodes,
)

logger = logging.getLogger(__name__)

_PREFERRED_OUTPUT_EXTS = (
    ".mp4",
    ".webm",
    ".mov",
    ".gif",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_filename_part(value: str, *, max_len: int = 120) -> str:
    invalid = '<>:"/\\|?*'
    for char in invalid:
        value = value.replace(char, "_")
    value = value.replace("\0", "_").strip().strip(".")
    if not value:
        value = "_"
    if len(value) > max_len:
        value = value[:max_len].rstrip()
    return value


def guess_media_type(filename: str) -> str:
    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def guess_image_extension(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def guess_audio_extension(data: bytes) -> str:
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE":
        return ".wav"
    if data.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return ".mp3"
    if data.startswith(b"fLaC"):
        return ".flac"
    if data.startswith(b"OggS"):
        return ".ogg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


@dataclass
class InputFile:
    data: bytes
    filename: str
    kind: str = "image"


@dataclass
class JobOutput:
    filename: str
    media_type: str
    node_id: str
    output_key: str
    size: int = 0

    def public(self, job_id: str) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "media_type": self.media_type,
            "node_id": self.node_id,
            "output_key": self.output_key,
            "size": self.size,
            "url": f"/files/{job_id}/{self.filename}",
        }


@dataclass
class Job:
    job_id: str
    kind: str
    workflow: str
    model: str
    prompt: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    overrides: list[tuple[str, str, Any]] = field(default_factory=list)

    status: str = "pending"
    progress: int = 0
    current_node: Optional[str] = None
    message: str = ""
    error: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    prompt_id: Optional[str] = None

    outputs: list[JobOutput] = field(default_factory=list)
    input_summary: dict[str, Any] = field(default_factory=dict)

    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)

    @property
    def finished(self) -> bool:
        return self.status in {"completed", "failed"}

    @property
    def duration_s(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return max(0.0, round(self.finished_at - self.created_at, 2))

    def primary_output(self) -> Optional[JobOutput]:
        if not self.outputs:
            return None
        for ext in _PREFERRED_OUTPUT_EXTS:
            for output in self.outputs:
                if output.filename.lower().endswith(ext):
                    return output
        return self.outputs[0]

    def public(self) -> dict[str, Any]:
        primary = self.primary_output()
        return {
            "job_id": self.job_id,
            "id": self.job_id,
            "kind": self.kind,
            "workflow": self.workflow,
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "current_node": self.current_node,
            "message": self.message,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "params": dict(self.params),
            "error": self.error,
            "prompt_id": self.prompt_id,
            "created_at": int(self.created_at),
            "created_at_iso": datetime.fromtimestamp(
                self.created_at, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "inputs": dict(self.input_summary),
            "outputs": [output.public(self.job_id) for output in self.outputs],
            "url": primary.public(self.job_id)["url"] if primary else None,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.public()
        payload["created_at"] = self.created_at
        payload["overrides"] = [[node, key, value] for node, key, value in self.overrides]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, job_id: str, run_dir: Path) -> "Job":
        outputs: list[JobOutput] = []
        for item in payload.get("outputs") or []:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename") or "")
            if not filename or not (run_dir / filename).is_file():
                continue
            outputs.append(
                JobOutput(
                    filename=filename,
                    media_type=str(item.get("media_type") or guess_media_type(filename)),
                    node_id=str(item.get("node_id") or ""),
                    output_key=str(item.get("output_key") or ""),
                    size=int(item.get("size") or 0),
                )
            )
        job = cls(
            job_id=job_id,
            kind=str(payload.get("kind") or "unknown"),
            workflow=str(payload.get("workflow") or ""),
            model=str(payload.get("model") or ""),
            prompt=str(payload.get("prompt") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            params=dict(payload.get("params") or {}),
            status=str(payload.get("status") or "failed"),
            progress=int(payload.get("progress") or 0),
            current_node=payload.get("current_node"),
            message=str(payload.get("message") or ""),
            error=payload.get("error"),
            created_at=float(payload.get("created_at") or time.time()),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            prompt_id=payload.get("prompt_id"),
            outputs=outputs,
            input_summary=dict(payload.get("inputs") or {}),
        )
        job.done.set()
        return job


class JobManager:
    """Runs jobs against the single configured ComfyUI instance.

    ComfyUI owns the GPU queue, so jobs are submitted immediately and tracked
    locally. Completion is decided by polling ``/history/{prompt_id}``; the
    WebSocket only supplies live progress telemetry. This keeps a dropped push
    from parking a job (and its caller) until the timeout.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        registry: WorkflowRegistry,
        client: ComfyUIClient | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.client = client or ComfyUIClient(
            settings.comfy_base_url,
            http_timeout_s=settings.http_timeout_s,
            auth_token=settings.comfy_auth_token or None,
        )
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    # ---------------------------------------------------------------- lifecycle

    def _load_existing(self) -> None:
        for path in sorted(
            self.settings.outputs_dir.glob("*/job.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:100]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                job = Job.from_dict(payload, job_id=path.parent.name, run_dir=path.parent)
                if job.status not in {"completed", "failed"}:
                    job.status = "failed"
                    job.error = "服务重启，任务已中断。"
                    job.finished_at = job.finished_at or time.time()
                    job.progress = 0
                    self._persist(job)
                self._jobs[job.job_id] = job
            except Exception as exc:  # noqa: BLE001 - one bad file must not block startup
                logger.warning("读取历史任务失败: %s: %s", path, exc)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.client.aclose()

    # ------------------------------------------------------------------ queries

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(str(job_id or "").strip())

    def list_recent(self, limit: int = 20) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return jobs[: max(1, int(limit))]

    # ------------------------------------------------------------------- submit

    async def submit(
        self,
        *,
        entry: WorkflowEntry,
        prompt: str,
        negative_prompt: str = "",
        params: dict[str, Any] | None = None,
        images: list[InputFile] | None = None,
        audio: InputFile | None = None,
        overrides: dict[str, Any] | None = None,
        prompt_node: str = "",
        negative_prompt_node: str = "",
        image_node: str = "",
        audio_node: str = "",
        kind: str | None = None,
    ) -> Job:
        spec = entry.parameter_spec
        job_kind = kind or entry.kind
        raw_params = dict(params or {})
        if job_kind == KIND_IMG2IMG:
            raw_params = apply_img2img_resize_params(raw_params)
        if spec is not None and "seed" in spec.parameters:
            seed_value = raw_params.get("seed")
            if seed_value in (None, "", "-1", -1):
                raw_params["seed"] = random.randint(0, 2**32 - 1)

        normalized = normalize_standard_params(spec=spec, request_params=raw_params)

        parsed_overrides: list[tuple[str, str, Any]] = []
        for node_ref, value in (overrides or {}).items():
            node_id, input_key = str(node_ref).split(".", 1) if "." in str(node_ref) else (
                str(node_ref),
                "value",
            )
            parsed_overrides.append((node_id, input_key, value))

        images = list(images or [])
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            kind=job_kind,
            workflow=entry.name,
            model=entry.model_id,
            prompt=str(prompt or ""),
            negative_prompt=str(negative_prompt or ""),
            params=normalized,
            overrides=parsed_overrides,
            input_summary={
                "images": [item.filename for item in images],
                "audio": audio.filename if audio else None,
            },
        )
        async with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        task = asyncio.create_task(
            self._run(
                job=job,
                entry=entry,
                images=images,
                audio=audio,
                prompt_node=prompt_node,
                negative_prompt_node=negative_prompt_node,
                image_node=image_node,
                audio_node=audio_node,
            ),
            name=f"cf-gradio-job-{job_id[:8]}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def wait(self, job: Job) -> Job:
        await job.done.wait()
        return job

    def _update(self, job: Job, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(job, key, value)
        self._persist(job)

    def _persist(self, job: Job) -> None:
        run_dir = self.settings.outputs_dir / job.job_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            import json

            (run_dir / "job.json").write_text(
                json.dumps(job.to_dict(), ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - persistence is best effort
            logger.debug("任务状态写入失败: %s: %s", job.job_id, exc)

    # ---------------------------------------------------------------- execution

    async def _run(
        self,
        *,
        job: Job,
        entry: WorkflowEntry,
        images: list[InputFile],
        audio: InputFile | None,
        prompt_node: str,
        negative_prompt_node: str,
        image_node: str,
        audio_node: str,
    ) -> None:
        ws_task: asyncio.Task[Any] | None = None
        try:
            self._update(job, status="queued", message="正在准备任务…")
            spec = entry.parameter_spec
            run_dir = self.settings.outputs_dir / job.job_id
            input_dir = run_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)

            image_rels: list[str] = []
            for index, image in enumerate(images, start=1):
                self._check_upload_size(image)
                filename = self._input_filename(job, image, index=index)
                (input_dir / filename).write_bytes(image.data)
                rel = await self.client.upload_file(
                    data=image.data,
                    filename=filename,
                    subfolder=self.settings.input_subdir,
                    content_type=guess_media_type(filename),
                )
                image_rels.append(rel)

            audio_rel = ""
            if audio is not None:
                self._check_upload_size(audio)
                filename = self._input_filename(job, audio, index=1)
                (input_dir / filename).write_bytes(audio.data)
                audio_rel = await self.client.upload_file(
                    data=audio.data,
                    filename=filename,
                    subfolder=self.settings.input_subdir,
                    content_type=guess_media_type(filename),
                )

            execution_params = dict(job.params)
            if image_rels:
                execution_params.setdefault("inputImgCount", len(image_rels))
                for index, rel in enumerate(image_rels[1:], start=2):
                    execution_params.setdefault(f"image{index}", rel)
                    if spec is not None and f"image{index}" in spec.parameters:
                        execution_params[f"image{index}"] = rel
            if spec is not None and "inputImgCount" in spec.parameters:
                execution_params["inputImgCount"] = len(image_rels)
            if execution_params != job.params:
                self._update(job, params=execution_params)

            workflow_obj = entry.workflow_obj
            image_count = len(image_rels)
            positive_target = prompt_node or (spec.prompt_node if spec else "")
            negative_target = negative_prompt_node or (
                spec.negative_prompt_node_for(image_count=image_count) if spec else ""
            )
            image_target = image_node or (spec.image_node if spec else "")
            audio_target = audio_node or (spec.audio_node if spec else "")

            standard_overrides = resolve_standard_overrides(
                workflow_obj=workflow_obj,
                spec=spec,
                request_params=execution_params,
            )
            prompt_graph, extra_data, _applied, trace = prepare_prompt(
                workflow_obj=workflow_obj,
                positive_prompt=job.prompt or None,
                negative_prompt=job.negative_prompt or None,
                positive_prompt_node=positive_target or None,
                negative_prompt_node=negative_target or None,
                image=image_rels[0] if image_rels else None,
                image_node=image_target or None,
                audio=audio_rel or None,
                audio_node=audio_target or None,
                overrides=standard_overrides + list(job.overrides),
            )

            object_info = await self.client.object_info()
            removed = prune_invalid_orphan_output_nodes(prompt_graph, object_info=object_info)
            normalized_inputs = normalize_prompt_enum_inputs(
                prompt_graph, object_info=object_info
            )
            if removed or normalized_inputs:
                logger.info(
                    "任务 %s 预处理: removed=%s normalized=%s",
                    job.job_id,
                    removed,
                    normalized_inputs,
                )

            (run_dir / "prompt.json").write_text(
                _json_dumps(
                    {
                        "prompt": prompt_graph,
                        "extra_data": extra_data,
                        "trace": trace,
                        "params": execution_params,
                    }
                ),
                encoding="utf-8",
            )

            client_id = uuid.uuid4().hex
            queued = await self.client.queue_prompt(
                prompt=prompt_graph, client_id=client_id, extra_data=extra_data
            )
            self._update(
                job,
                status="queued",
                prompt_id=queued.prompt_id,
                message=f"已提交到 ComfyUI 队列 (#{queued.number})"
                if queued.number is not None
                else "已提交到 ComfyUI 队列",
            )

            ws_task = asyncio.create_task(
                self._monitor_ws(job=job, client_id=client_id, prompt_id=queued.prompt_id),
                name=f"cf-gradio-ws-{job.job_id[:8]}",
            )

            history = await self._await_completion(job=job, prompt_id=queued.prompt_id)
            (run_dir / "history.json").write_text(_json_dumps(history), encoding="utf-8")

            outputs = await self._collect_outputs(job=job, history=history, run_dir=run_dir)
            if not outputs:
                raise ComfyApiError("ComfyUI 执行完成但没有产生可下载的输出文件。")

            self._update(
                job,
                status="completed",
                progress=100,
                current_node=None,
                finished_at=time.time(),
                outputs=outputs,
                message=f"完成，共 {len(outputs)} 个输出",
            )
        except asyncio.CancelledError:
            self._update(
                job,
                status="failed",
                error="任务已取消。",
                finished_at=time.time(),
            )
            raise
        except (ComfyApiError, TimeoutError, InvalidParameterError, KeyError, ValueError) as exc:
            logger.warning("任务 %s 失败: %s", job.job_id, exc)
            self._update(
                job,
                status="failed",
                error=str(exc),
                finished_at=time.time(),
                message="任务失败",
            )
        except Exception as exc:  # noqa: BLE001 - never leak a task without a terminal state
            logger.exception("任务 %s 异常: %s", job.job_id, exc)
            self._update(
                job,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time(),
                message="任务失败",
            )
        finally:
            if ws_task is not None:
                ws_task.cancel()
                await asyncio.gather(ws_task, return_exceptions=True)
            job.done.set()

    def _check_upload_size(self, item: InputFile) -> None:
        limit = (
            self.settings.max_image_bytes
            if item.kind == "image"
            else self.settings.max_upload_bytes
        )
        if len(item.data) > limit:
            raise ValueError(
                f"输入文件 {item.filename!r} 超过大小限制 ({len(item.data)} > {limit} bytes)"
            )

    def _input_filename(self, job: Job, item: InputFile, *, index: int) -> str:
        raw = Path(str(item.filename or "input")).name
        stem = safe_filename_part(Path(raw).stem, max_len=50) or "input"
        suffix = Path(raw).suffix.lower()
        allowed = _IMAGE_EXTS if item.kind == "image" else _AUDIO_EXTS
        if suffix not in allowed:
            suffix = (
                guess_image_extension(item.data)
                if item.kind == "image"
                else guess_audio_extension(item.data)
            )
        return f"{job.job_id[:12]}-{index}-{stem}{suffix}"

    async def _monitor_ws(self, *, job: Job, client_id: str, prompt_id: str) -> None:
        try:
            async for message in self.client.ws_events(client_id=client_id):
                payload = message.get("data")
                data = payload if isinstance(payload, dict) else {}
                event_prompt = data.get("prompt_id")
                if event_prompt not in (None, prompt_id):
                    continue
                event_type = message.get("type")

                if event_type == "execution_start":
                    if job.status == "queued":
                        self._update(
                            job, status="running", started_at=job.started_at or time.time()
                        )
                elif event_type == "executing":
                    node = data.get("node")
                    if node is None:
                        return
                    fields: dict[str, Any] = {"current_node": str(node)}
                    if job.status == "queued":
                        fields["status"] = "running"
                        fields["started_at"] = job.started_at or time.time()
                    self._update(job, **fields)
                elif event_type == "progress":
                    try:
                        value = float(data.get("value") or 0)
                        maximum = float(data.get("max") or 0)
                    except (TypeError, ValueError):
                        continue
                    if maximum > 0:
                        percent = max(0, min(99, int(value / maximum * 100)))
                        if percent >= job.progress:
                            self._update(job, progress=percent)
                elif event_type == "execution_error":
                    error = data.get("exception_message") or data.get("exception_type")
                    if error:
                        job.error = str(error)
                elif event_type == "execution_interrupted":
                    job.error = "ComfyUI 执行被中断。"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - WS is telemetry only
            logger.debug("任务 %s WebSocket 结束: %s", job.job_id, exc)

    async def _await_completion(self, *, job: Job, prompt_id: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(1, self.settings.timeout_s)
        poll_interval = max(0.1, float(self.settings.poll_interval_s))
        consecutive_errors = 0
        missing_queue_checks = 0
        checks = 0

        while True:
            try:
                entry = await self.client.get_history_entry(prompt_id)
                consecutive_errors = 0
            except Exception as exc:  # noqa: BLE001 - transient ComfyUI hiccup
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    raise ComfyApiError(
                        f"连续 {consecutive_errors} 次读取 ComfyUI /history 失败: {exc}"
                    ) from exc
                entry = None

            if entry is not None:
                error = history_entry_error(entry)
                if error:
                    raise ComfyApiError(error)
                if history_entry_is_complete(entry):
                    return entry

            checks += 1
            if checks % 10 == 0:
                missing_queue_checks = await self._check_queue_state(
                    job=job,
                    prompt_id=prompt_id,
                    missing=missing_queue_checks,
                )

            if loop.time() > deadline:
                raise TimeoutError(
                    f"任务等待超时（{self.settings.timeout_s}s, prompt_id={prompt_id}）。"
                )
            await asyncio.sleep(poll_interval)

    async def _check_queue_state(
        self, *, job: Job, prompt_id: str, missing: int
    ) -> int:
        try:
            queue = await self.client.get_queue()
        except Exception:  # noqa: BLE001 - /history remains the source of truth
            return missing
        running_ids, pending_ids = _queue_prompt_ids(queue)
        if prompt_id in running_ids:
            if job.status == "queued":
                self._update(
                    job,
                    status="running",
                    started_at=job.started_at or time.time(),
                    message="ComfyUI 正在执行",
                )
            return 0
        if prompt_id in pending_ids:
            return 0
        missing += 1
        if missing >= 3:
            raise ComfyApiError(
                f"ComfyUI 队列中已找不到该任务（prompt_id={prompt_id}），可能后端已重启或队列被清空。"
            )
        return missing

    async def _collect_outputs(
        self, *, job: Job, history: dict[str, Any], run_dir: Path
    ) -> list[JobOutput]:
        outputs: list[JobOutput] = []
        for node_id, output_key, fileinfo in iter_file_outputs(history):
            filename = fileinfo["filename"]
            blob = await self.client.view_bytes(
                filename=filename,
                subfolder=fileinfo["subfolder"],
                folder_type=fileinfo["type"],
            )
            safe_node = safe_filename_part(node_id, max_len=60)
            safe_name = safe_filename_part(Path(filename).name, max_len=120)
            out_name = f"{safe_node}__{safe_name}"
            (run_dir / out_name).write_bytes(blob)
            outputs.append(
                JobOutput(
                    filename=out_name,
                    media_type=guess_media_type(out_name),
                    node_id=node_id,
                    output_key=output_key,
                    size=len(blob),
                )
            )
        return outputs


def _queue_prompt_ids(queue: Any) -> tuple[set[str], set[str]]:
    if not isinstance(queue, dict):
        return set(), set()
    running: set[str] = set()
    pending: set[str] = set()
    for key, target in (("queue_running", running), ("queue_pending", pending)):
        items = queue.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], str):
                target.add(item[1])
    return running, pending


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
