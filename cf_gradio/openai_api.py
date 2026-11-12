from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import re
import secrets
import socket
import time
from binascii import Error as BinasciiError
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .jobs import InputFile, Job, JobManager, guess_media_type
from .params import STANDARD_PARAMETER_ORDER, public_parameter_spec
from .registry import WorkflowEntry, WorkflowRegistry
from .workflow import (
    KIND_DIGITALHUMAN,
    KIND_IMG2IMG,
    KIND_IMG2VIDEO,
    KIND_TXT2AUDIO,
    KIND_TXT2IMG,
    KIND_TXT2VIDEO,
)


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code


async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    error: dict[str, Any] = {"message": exc.message, "type": exc.error_type}
    if exc.code:
        error["code"] = exc.code
    return JSONResponse(status_code=exc.status_code, content={"error": error})


def _check_api_token(settings: Settings, authorization: str | None) -> None:
    expected = (settings.api_token or "").strip()
    if not expected:
        return
    raw = (authorization or "").strip()
    if not raw.lower().startswith("bearer "):
        raise ApiError(
            "缺少 API Key。请在 Authorization 头中使用 'Bearer <API_TOKEN>'。",
            401,
            code="invalid_api_key",
        )
    token = raw.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, expected):
        raise ApiError("API Key 无效。", 401, code="invalid_api_key")


def _origin(request: Request, settings: Settings) -> str:
    if settings.public_base_url:
        return settings.public_base_url
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    return f"{proto}://{host}".rstrip("/")


def _absolute(request: Request, settings: Settings, path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith(("http://", "https://")):
        return path
    return _origin(request, settings) + "/" + path.lstrip("/")


def _openai_created() -> int:
    return int(time.time())


def _is_async_header(value: str | None) -> bool:
    return str(value or "").strip().lower() not in {"", "0", "false", "no", "off"}


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _get(values: Any, name: str) -> Any:
    if values is None:
        return None
    try:
        return values.get(name)
    except Exception:  # noqa: BLE001 - FormData / dict both support get
        return None


def _collect_params(values: Any, entry: WorkflowEntry) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name in STANDARD_PARAMETER_ORDER:
        value = _clean(_get(values, name))
        if value is not None:
            params[name] = value
    if "cfg" not in params:
        cfg_scale = _clean(_get(values, "cfg_scale"))
        if cfg_scale is not None:
            params["cfg"] = cfg_scale
    spec = entry.parameter_spec
    if spec is not None:
        for name, definition in spec.parameters.items():
            if definition.type == "image":
                # Reference-image slots are filled from the request's image
                # list, not from a path string supplied by the caller.
                continue
            value = _clean(_get(values, name))
            if value is not None:
                params[name] = value
    return params


def _parse_overrides(raw: Any) -> dict[str, Any]:
    if raw in (None, "", {}):
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(f"overrides 不是合法 JSON: {exc}", 400) from exc
    if not isinstance(raw, dict):
        raise ApiError("overrides 必须是对象，例如 {\"12.seed\": 42}", 400)
    return {str(key): value for key, value in raw.items()}


def _node_ref(values: Any, name: str) -> str:
    value = _clean(_get(values, name))
    return str(value) if value is not None else ""


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"请求体不是合法 JSON: {exc}", 400) from exc
    if not isinstance(body, dict):
        raise ApiError("JSON 请求体必须是对象。", 400)
    return body


def _is_upload(value: Any) -> bool:
    return hasattr(value, "read") and hasattr(value, "filename")


def _collect_image_items(values: Any) -> list[Any]:
    """Ordered image references from multipart or JSON (canonical fields first)."""

    pairs: list[tuple[str, Any]] = []
    multi_items = getattr(values, "multi_items", None)
    if callable(multi_items):
        pairs = [(str(key), value) for key, value in multi_items()]
    elif isinstance(values, dict):
        for key, value in values.items():
            items = value if isinstance(value, list) else [value]
            pairs.extend((str(key), item) for item in items)
    if not pairs:
        return []

    canonical_fields = ("image", "input_reference", "images", "input_references")
    canonical: list[tuple[int, int, Any]] = []
    indexed: list[tuple[int, Any]] = []
    numbered: list[tuple[int, Any]] = []
    for order, (name, value) in enumerate(pairs):
        if name in canonical_fields:
            canonical.append((canonical_fields.index(name), order, value))
            continue
        match = re.match(r"^(?:image|input_reference)\[(\d+)\]$", name)
        if match:
            indexed.append((int(match.group(1)), value))
            continue
        match = re.match(r"^(?:image|input_reference)(\d+)$", name)
        if match and int(match.group(1)) >= 1:
            numbered.append((int(match.group(1)), value))

    canonical.sort(key=lambda item: (item[0], item[1]))
    indexed.sort(key=lambda item: item[0])
    numbered.sort(key=lambda item: item[0])

    items = [value for _rank, _order, value in canonical]
    items.extend(value for _index, value in indexed)
    items.extend(value for _index, value in numbered)

    seen: set[Any] = set()
    unique: list[Any] = []
    for item in items:
        if isinstance(item, (str, bytes, int, float, bool)) or item is None:
            marker: Any = ("v", item)
        elif isinstance(item, dict):
            marker = ("v", json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
        else:
            marker = ("id", id(item))
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(item, str) and not item.strip():
            continue
        if item is None:
            continue
        unique.append(item)
    return unique


def _reference_value(item: Any) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value or None
    if isinstance(item, dict):
        for key in ("url", "image_url", "audio_url", "data"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


async def _download_url(
    url: str, *, settings: Settings, kind: str
) -> tuple[bytes, str]:
    parsed = httpx.URL(url)
    if parsed.scheme not in {"http", "https"}:
        raise ApiError("只支持 http/https 的引用 URL。", 400)

    if not settings.allow_private_image_urls:
        allowed = await asyncio.to_thread(_host_is_public, parsed.host or "")
        if not allowed:
            raise ApiError("该 URL 指向内网/回环地址，已拒绝下载。", 400)

    max_bytes = (
        settings.max_image_bytes if kind == "image" else settings.max_upload_bytes
    )
    accept = "image/*,*/*" if kind == "image" else "audio/*,*/*"
    timeout = httpx.Timeout(timeout=settings.http_timeout_s)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        current = parsed
        for _redirect in range(3):
            async with client.stream("GET", current, headers={"Accept": accept}) as response:
                if 300 <= response.status_code < 400 and response.headers.get("location"):
                    nxt = httpx.URL(response.headers["location"])
                    current = nxt if nxt.scheme else current.join(nxt)
                    if current.scheme not in {"http", "https"} or not current.host:
                        raise ApiError("重定向地址不合法。", 400)
                    if not settings.allow_private_image_urls and not await asyncio.to_thread(
                        _host_is_public, current.host
                    ):
                        raise ApiError("重定向指向内网地址，已拒绝。", 400)
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise ApiError(
                        f"下载参考文件失败: HTTP {exc.response.status_code}", 400
                    ) from exc

                content_type = (response.headers.get("content-type") or "").lower()
                prefix = "image/" if kind == "image" else "audio/"
                if content_type and not (
                    content_type.startswith(prefix) or "octet-stream" in content_type
                ):
                    raise ApiError("URL 返回的类型与预期不符。", 400)

                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > max_bytes:
                        raise ApiError("引用文件超过大小限制。", 413)
                filename = Path(urlparse(str(current)).path).name or f"{kind}.bin"
                return bytes(buffer), filename

        raise ApiError("引用 URL 重定向次数过多。", 400)


def _host_is_public(host: str) -> bool:
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except Exception:  # noqa: BLE001
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def _decode_base64(value: str) -> bytes:
    text = value.strip()
    if text.startswith("data:"):
        if "," not in text:
            raise ApiError("data URL 格式不合法。", 400)
        _header, text = text.split(",", 1)
    text = "".join(text.split())
    try:
        return base64.b64decode(text, validate=False)
    except BinasciiError as exc:
        raise ApiError(f"base64 解码失败: {exc}", 400) from exc


async def _stage_reference(
    item: Any, *, settings: Settings, kind: str, index: int
) -> InputFile:
    if _is_upload(item):
        data = await item.read()
        if not data:
            raise ApiError("上传的参考文件为空。", 400)
        filename = getattr(item, "filename", None) or f"{kind}{index}.bin"
        return InputFile(data=data, filename=str(filename), kind=kind)

    value = _reference_value(item)
    if value is None:
        raise ApiError(
            "参考文件必须是上传文件、URL，或 base64/data URL 字符串。", 400
        )

    if value.startswith(("http://", "https://")):
        data, filename = await _download_url(value, settings=settings, kind=kind)
        return InputFile(data=data, filename=filename, kind=kind)

    data = _decode_base64(value)
    # No extension hint: JobManager sniffs the magic bytes and picks .png/.jpg/.wav/...
    return InputFile(data=data, filename=f"{kind}{index}.bin", kind=kind)


async def _collect_inputs(
    items: Iterable[Any], *, settings: Settings, kind: str, limit: int = 8
) -> list[InputFile]:
    staged: list[InputFile] = []
    for index, item in enumerate(items, start=1):
        if index > limit:
            raise ApiError(f"参考文件数量超过上限 ({limit})。", 400)
        staged.append(
            await _stage_reference(item, settings=settings, kind=kind, index=index)
        )
    return staged


def _resolve_entry(
    registry: WorkflowRegistry,
    *,
    model: str,
    kinds: Iterable[str],
    endpoint: str,
) -> WorkflowEntry:
    registry.refresh()
    allowed = set(kinds)
    if model:
        entry = registry.get(model)
        if entry is None or not entry.ready:
            available = ", ".join(
                f"{item.model_id} ({item.kind})" for item in registry.ready_entries()
            ) or "(无)"
            raise ApiError(
                f"模型 '{model}' 不存在。可用模型: {available}",
                404,
                code="model_not_found",
            )
        if not any(entry.supports(kind) for kind in allowed):
            compatible = ", ".join(
                item.model_id for item in registry.by_kind(allowed)
            ) or "(无)"
            raise ApiError(
                f"模型 '{model}' 支持 {', '.join(entry.capabilities.kinds) or '(未知)'}，"
                f"{endpoint} 需要 {', '.join(sorted(allowed))}。兼容模型: {compatible}",
                400,
                code="model_kind_mismatch",
            )
        return entry

    candidates = registry.by_kind(allowed)
    if not candidates:
        raise ApiError(
            f"工作流目录中没有可用于 {endpoint} 的工作流（kind={', '.join(sorted(allowed))}）。",
            400,
            code="no_available_model",
        )
    return candidates[0]


def _video_status(job: Job) -> str:
    if job.status == "completed":
        return "completed"
    if job.status == "failed":
        return "failed"
    if job.status == "running":
        return "in_progress"
    return "queued"


def _video_object(job: Job, *, request: Request, settings: Settings) -> dict[str, Any]:
    status = _video_status(job)
    primary = job.primary_output()
    url = (
        _absolute(request, settings, primary.public(job.job_id)["url"])
        if status == "completed" and primary
        else None
    )
    error = None
    if status == "failed":
        error = {"message": job.error or "视频生成失败", "type": "server_error"}
    return {
        "id": f"video_{job.job_id}",
        "object": "video",
        "model": job.model,
        "status": status,
        "progress": 100 if status == "completed" else int(job.progress),
        "created_at": int(job.created_at),
        "seconds": str(job.params.get("duration") or job.params.get("seconds") or ""),
        "size": str(job.params.get("size") or ""),
        "quality": "standard",
        "url": url,
        "error": error,
    }


def _video_id_to_job_id(video_id: str) -> str:
    raw = str(video_id or "").strip()
    return raw[len("video_") :] if raw.startswith("video_") else raw


def _pick_audio_output(job: Job):
    for output in job.outputs:
        if output.media_type.startswith("audio/") or output.filename.lower().endswith(
            (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
        ):
            return output
    return job.outputs[0] if job.outputs else None


async def _wait_job(job: Job) -> Job:
    await job.done.wait()
    return job


def _job_outputs_absolute(
    job: Job, *, request: Request, settings: Settings
) -> list[dict[str, Any]]:
    outputs = []
    for item in job.outputs:
        payload = item.public(job.job_id)
        payload["url"] = _absolute(request, settings, payload["url"])
        outputs.append(payload)
    return outputs


def _public_job_payload(
    job: Job, *, request: Request, settings: Settings
) -> dict[str, Any]:
    payload = job.public()
    payload["outputs"] = _job_outputs_absolute(job, request=request, settings=settings)
    primary = job.primary_output()
    payload["url"] = (
        _absolute(request, settings, primary.public(job.job_id)["url"]) if primary else None
    )
    return payload


def create_api_router(
    *, settings: Settings, registry: WorkflowRegistry, jobs: JobManager
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def auth(authorization: str | None) -> None:
        _check_api_token(settings, authorization)

    # ------------------------------------------------------------------ models

    @router.get("/models")
    async def list_models(authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        registry.refresh()
        return {"object": "list", "data": registry.model_objects()}

    @router.get("/models/{model_id}")
    async def get_model(model_id: str, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        registry.refresh()
        entry = registry.get(model_id)
        if entry is None or not entry.ready:
            raise ApiError(f"模型 '{model_id}' 不存在。", 404, code="model_not_found")
        return {
            "id": entry.model_id,
            "object": "model",
            "created": int(entry.mtime_ns / 1_000_000_000),
            "owned_by": "cf-gradio",
            "kind": entry.kind,
            "kinds": list(entry.capabilities.kinds),
            "workflow": entry.name,
            "display_name": entry.display_name,
            "parameter_spec": public_parameter_spec(entry.parameter_spec),
            "parameter_error": entry.parameter_error,
        }

    # ------------------------------------------------------------------ images

    @router.post("/images/generations")
    async def images_generations(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        body = await _json_body(request)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise ApiError("缺少 'prompt'。")

        entry = _resolve_entry(
            registry,
            model=str(body.get("model") or "").strip(),
            kinds={KIND_TXT2IMG},
            endpoint="images/generations",
        )
        params = _collect_params(body, entry)
        job = await jobs.submit(
            entry=entry,
            prompt=prompt,
            negative_prompt=str(body.get("negative_prompt") or "").strip(),
            params=params,
            overrides=_parse_overrides(body.get("overrides")),
            prompt_node=_node_ref(body, "prompt_node"),
            negative_prompt_node=_node_ref(body, "negative_prompt_node"),
            kind=KIND_TXT2IMG,
        )
        if _is_async_header(x_comfyui_async):
            return {"job_id": job.job_id, "status": "pending"}

        await _wait_job(job)
        if job.status != "completed":
            raise ApiError(job.error or "图片生成失败", 500, error_type="server_error")

        response_format = str(body.get("response_format") or "url").strip().lower()
        created = _openai_created()
        if response_format in {"b64_json", "b64", "base64", "base64_json"}:
            data = []
            for output in job.outputs:
                path = settings.outputs_dir / job.job_id / output.filename
                if path.is_file():
                    data.append(
                        {"b64_json": base64.b64encode(path.read_bytes()).decode("ascii")}
                    )
            if not data:
                raise ApiError("生成完成但没有可读取的图片。", 500, error_type="server_error")
            return {"created": created, "data": data}

        data = [
            {"url": _absolute(request, settings, item.public(job.job_id)["url"])}
            for item in job.outputs
        ]
        return {"created": created, "data": data}

    @router.post("/images/edits")
    async def images_edits(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].lower()
        if content_type == "application/json":
            values: Any = await _json_body(request)
        else:
            values = await request.form()

        prompt = str(_get(values, "prompt") or "").strip()
        if not prompt:
            raise ApiError("缺少 'prompt'。")

        image_items = _collect_image_items(values)
        if not image_items:
            raise ApiError("缺少 'image'。")
        images = await _collect_inputs(
            image_items, settings=settings, kind="image", limit=8
        )

        entry = _resolve_entry(
            registry,
            model=str(_get(values, "model") or _get(values, "workflow") or "").strip(),
            kinds={KIND_IMG2IMG},
            endpoint="images/edits",
        )
        params = _collect_params(values, entry)
        job = await jobs.submit(
            entry=entry,
            prompt=prompt,
            negative_prompt=str(_get(values, "negative_prompt") or "").strip(),
            params=params,
            images=images,
            overrides=_parse_overrides(_get(values, "overrides")),
            prompt_node=_node_ref(values, "prompt_node"),
            negative_prompt_node=_node_ref(values, "negative_prompt_node"),
            image_node=_node_ref(values, "image_node"),
            kind=KIND_IMG2IMG,
        )
        if _is_async_header(x_comfyui_async):
            return {"job_id": job.job_id, "status": "pending"}

        await _wait_job(job)
        if job.status != "completed":
            raise ApiError(job.error or "图片编辑失败", 500, error_type="server_error")

        response_format = str(_get(values, "response_format") or "url").strip().lower()
        created = _openai_created()
        if response_format in {"b64_json", "b64", "base64", "base64_json"}:
            data = []
            for output in job.outputs:
                path = settings.outputs_dir / job.job_id / output.filename
                if path.is_file():
                    data.append(
                        {"b64_json": base64.b64encode(path.read_bytes()).decode("ascii")}
                    )
            return {"created": created, "data": data}
        return {
            "created": created,
            "data": [
                {"url": _absolute(request, settings, item.public(job.job_id)["url"])}
                for item in job.outputs
            ],
        }

    # ------------------------------------------------------------------ videos

    @router.post("/videos")
    async def create_video(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].lower()
        if content_type == "application/json":
            values: Any = await _json_body(request)
        else:
            values = await request.form()

        prompt = str(_get(values, "prompt") or "").strip()
        if not prompt:
            raise ApiError("缺少 'prompt'。")

        image_items = _collect_image_items(values)
        images = await _collect_inputs(
            image_items, settings=settings, kind="image", limit=8
        )

        audio_value = _get(values, "audio")
        audio_files: list[InputFile] = []
        if audio_value not in (None, ""):
            if _is_upload(audio_value) or _reference_value(audio_value):
                audio_files = await _collect_inputs(
                    [audio_value], settings=settings, kind="audio", limit=1
                )

        if audio_files:
            kinds = {KIND_DIGITALHUMAN}
            endpoint = "videos (数字人)"
        elif images:
            kinds = {KIND_IMG2VIDEO}
            endpoint = "videos (图生视频)"
        else:
            kinds = {KIND_TXT2VIDEO}
            endpoint = "videos (文生视频)"

        entry = _resolve_entry(
            registry,
            model=str(_get(values, "model") or _get(values, "workflow") or "").strip(),
            kinds=kinds,
            endpoint=endpoint,
        )
        params = _collect_params(values, entry)
        seconds = _clean(_get(values, "seconds"))
        if seconds is not None and "duration" not in params:
            params["duration"] = seconds

        job = await jobs.submit(
            entry=entry,
            prompt=prompt,
            negative_prompt=str(_get(values, "negative_prompt") or "").strip(),
            params=params,
            images=images,
            audio=audio_files[0] if audio_files else None,
            overrides=_parse_overrides(_get(values, "overrides")),
            prompt_node=_node_ref(values, "prompt_node"),
            negative_prompt_node=_node_ref(values, "negative_prompt_node"),
            image_node=_node_ref(values, "image_node"),
            audio_node=_node_ref(values, "audio_node"),
            kind=next(iter(kinds)),
        )
        return _video_object(job, request=request, settings=settings)

    @router.get("/videos/{video_id}")
    async def get_video(
        video_id: str,
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        job = jobs.get(_video_id_to_job_id(video_id))
        if job is None:
            raise ApiError("视频任务不存在。", 404, code="video_not_found")
        return _video_object(job, request=request, settings=settings)

    @router.get("/videos/{video_id}/content")
    async def get_video_content(
        video_id: str,
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        job = jobs.get(_video_id_to_job_id(video_id))
        if job is None:
            raise ApiError("视频任务不存在。", 404, code="video_not_found")
        if job.status != "completed":
            raise ApiError("视频尚未生成完成。", 409, code="video_not_ready")
        primary = job.primary_output()
        if primary is None:
            raise ApiError("任务没有输出文件。", 404)
        path = settings.outputs_dir / job.job_id / primary.filename
        if not path.is_file():
            raise ApiError("输出文件缺失。", 404)
        return FileResponse(
            path=str(path),
            media_type=primary.media_type or guess_media_type(primary.filename),
            filename=primary.filename,
            content_disposition_type="inline",
        )

    # ------------------------------------------------------------------- audio

    @router.post("/audio/speech")
    async def audio_speech(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        body = await _json_body(request)
        text = str(body.get("input") or "").strip()
        if not text:
            raise ApiError("缺少 'input'。")

        entry = _resolve_entry(
            registry,
            model=str(body.get("model") or "").strip(),
            kinds={KIND_TXT2AUDIO},
            endpoint="audio/speech",
        )
        params = _collect_params(body, entry)
        job = await jobs.submit(
            entry=entry,
            prompt=text,
            params=params,
            overrides=_parse_overrides(body.get("overrides")),
            prompt_node=_node_ref(body, "prompt_node"),
            kind=KIND_TXT2AUDIO,
        )
        if _is_async_header(x_comfyui_async):
            return {"job_id": job.job_id, "status": "pending"}

        await _wait_job(job)
        if job.status != "completed":
            raise ApiError(job.error or "语音合成失败", 500, error_type="server_error")
        output = _pick_audio_output(job)
        if output is None:
            raise ApiError("语音合成没有产生音频输出。", 500, error_type="server_error")
        path = settings.outputs_dir / job.job_id / output.filename
        if not path.is_file():
            raise ApiError("音频输出文件缺失。", 500, error_type="server_error")
        return FileResponse(
            path=str(path),
            media_type=output.media_type or guess_media_type(output.filename),
            filename=output.filename,
            content_disposition_type="inline",
        )

    # ------------------------------------------------------------ digital human

    @router.post("/digital-human")
    async def digital_human(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].lower()
        if content_type == "application/json":
            values: Any = await _json_body(request)
        else:
            values = await request.form()

        image_items = _collect_image_items(values)
        if not image_items:
            raise ApiError("缺少 'image'（数字人参考图片）。")
        images = await _collect_inputs(
            image_items[:1], settings=settings, kind="image", limit=1
        )

        audio_value = _get(values, "audio")
        if audio_value in (None, "") or not (
            _is_upload(audio_value) or _reference_value(audio_value)
        ):
            raise ApiError("缺少 'audio'（数字人驱动音频）。")
        audio = (
            await _collect_inputs([audio_value], settings=settings, kind="audio", limit=1)
        )[0]

        entry = _resolve_entry(
            registry,
            model=str(_get(values, "model") or _get(values, "workflow") or "").strip(),
            kinds={KIND_DIGITALHUMAN},
            endpoint="digital-human",
        )
        params = _collect_params(values, entry)
        seconds = _clean(_get(values, "seconds"))
        if seconds is not None and "duration" not in params:
            params["duration"] = seconds

        job = await jobs.submit(
            entry=entry,
            prompt=str(_get(values, "prompt") or "").strip(),
            negative_prompt=str(_get(values, "negative_prompt") or "").strip(),
            params=params,
            images=images,
            audio=audio,
            overrides=_parse_overrides(_get(values, "overrides")),
            prompt_node=_node_ref(values, "prompt_node"),
            image_node=_node_ref(values, "image_node"),
            audio_node=_node_ref(values, "audio_node"),
            kind=KIND_DIGITALHUMAN,
        )
        return _video_object(job, request=request, settings=settings)

    # -------------------------------------------------------------------- jobs

    @router.get("/jobs/{job_id}")
    async def get_job(
        job_id: str,
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth(authorization)
        job = jobs.get(job_id)
        if job is None:
            raise ApiError("任务不存在。", 404, code="job_not_found")
        return _public_job_payload(job, request=request, settings=settings)

    return router


def create_root_router(*, settings: Settings, registry: WorkflowRegistry, jobs: JobManager) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        registry.refresh()
        comfy_ok = False
        try:
            await asyncio.wait_for(jobs.client.system_stats(), timeout=3.0)
            comfy_ok = True
        except Exception:  # noqa: BLE001 - health is informational
            pass
        return {
            "status": "ok",
            "comfyui": comfy_ok,
            "comfyui_base_url": settings.comfy_base_url,
            "models": len(registry.ready_entries()),
            "workflows_dir": str(settings.workflows_dir),
            "load_errors": [
                {"name": item.name, "error": item.error} for item in registry.load_errors()
            ],
        }

    @router.get("/files/{job_id}/{filename}")
    async def get_file(job_id: str, filename: str):
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ApiError("文件名不合法。", 404)
        job = jobs.get(job_id)
        if job is None:
            raise ApiError("任务不存在。", 404)
        # /files is intentionally unauthenticated for easy media embedding, but
        # it only exposes deliverables registered on the job -- never job.json,
        # prompt.json, history.json or staged inputs.
        if filename not in {output.filename for output in job.outputs}:
            raise ApiError("文件不存在。", 404)
        job_dir = (settings.outputs_dir / job_id).resolve()
        path = (job_dir / filename).resolve()
        if path.parent != job_dir or not path.is_file():
            raise ApiError("文件不存在。", 404)
        return FileResponse(
            path=str(path),
            media_type=guess_media_type(filename),
            filename=filename,
            content_disposition_type="inline",
        )

    return router


def install_routes(
    app: Any,
    *,
    settings: Settings,
    registry: WorkflowRegistry,
    jobs: JobManager,
) -> None:
    """Attach the OpenAI + file routes to a FastAPI app exactly once."""

    if getattr(app.state, "cf_gradio_installed", False):
        return
    app.include_router(create_api_router(settings=settings, registry=registry, jobs=jobs))
    app.include_router(
        create_root_router(settings=settings, registry=registry, jobs=jobs)
    )
    app.add_exception_handler(ApiError, _api_error_handler)
    app.state.cf_gradio_installed = True
