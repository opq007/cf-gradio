from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from .config import Settings
from .jobs import InputFile, Job, JobManager
from .params import PARAMETER_DESCRIPTIONS, PARAMETER_LABELS
from .registry import WorkflowEntry, WorkflowRegistry
from .workflow import (
    KIND_DIGITALHUMAN,
    KIND_IMG2IMG,
    KIND_IMG2VIDEO,
    KIND_LABELS,
    KIND_TXT2AUDIO,
    KIND_TXT2IMG,
    KIND_TXT2VIDEO,
)

STATUS_LABELS = {
    "pending": "排队中",
    "queued": "排队中",
    "running": "生成中",
    "completed": "已完成",
    "failed": "失败",
}

_IMAGE_KINDS = {KIND_IMG2IMG, KIND_IMG2VIDEO, KIND_DIGITALHUMAN}


@dataclass(frozen=True)
class TabSpec:
    title: str
    modes: tuple[tuple[str, str], ...]
    prompt_label: str
    prompt_placeholder: str
    output_kind: str  # image | video | audio
    show_negative: bool = True


TABS = (
    TabSpec(
        title="生图",
        modes=(("文生图", KIND_TXT2IMG), ("图生图", KIND_IMG2IMG)),
        prompt_label="提示词",
        prompt_placeholder="描述你想要的画面…",
        output_kind="image",
    ),
    TabSpec(
        title="生视频",
        modes=(("文生视频", KIND_TXT2VIDEO), ("图生视频", KIND_IMG2VIDEO)),
        prompt_label="提示词",
        prompt_placeholder="描述镜头、动作与氛围…",
        output_kind="video",
    ),
    TabSpec(
        title="TTS 语音",
        modes=(("语音合成", KIND_TXT2AUDIO),),
        prompt_label="要合成的文本",
        prompt_placeholder="输入要朗读的文字…",
        output_kind="audio",
        show_negative=False,
    ),
    TabSpec(
        title="数字人",
        modes=(("图片 + 音频 → 视频", KIND_DIGITALHUMAN),),
        prompt_label="动作/风格描述（可选）",
        prompt_placeholder="例如：自然说话，轻微点头…",
        output_kind="video",
    ),
)


CSS = """
.cf-status {line-height: 1.7}
"""


def _kind_choices(registry: WorkflowRegistry, kind: str) -> list[tuple[str, str]]:
    return [
        (f"{entry.display_name}  ·  {entry.model_id}", entry.model_id)
        for entry in registry.by_kind({kind})
    ]


def _first_choice(registry: WorkflowRegistry, kind: str) -> Optional[str]:
    choices = _kind_choices(registry, kind)
    return choices[0][1] if choices else None


def _build_param_control(name: str, definition: Any) -> Any:
    label = PARAMETER_LABELS.get(name) or name
    info = definition.description or PARAMETER_DESCRIPTIONS.get(name, "")
    default = definition.default

    if definition.type == "size":
        return gr.Textbox(
            value=str(default) if default not in (None, "") else "",
            label=label,
            info=info,
            placeholder="例如 1024x1024",
        )
    if definition.type == "bool":
        return gr.Checkbox(
            value=bool(default) if default is not None else False,
            label=label,
            info=info,
        )
    if definition.type == "float":
        return gr.Textbox(
            value=str(default) if default is not None else "",
            label=label,
            info=info,
            placeholder="留空使用工作流默认值",
        )
    if definition.type == "int":
        return gr.Textbox(
            value=str(default) if default is not None else "",
            label=label,
            info=info,
            placeholder="留空使用工作流默认值",
        )
    return gr.Textbox(
        value=str(default) if default not in (None, "") else "",
        label=label,
        info=info,
    )


def _job_status_markdown(job: Job) -> str:
    lines = [
        f"**任务** `{job.job_id}`",
        f"**状态** {STATUS_LABELS.get(job.status, job.status)} · {job.progress}%",
    ]
    if job.current_node:
        lines.append(f"**当前节点** `{job.current_node}`")
    if job.message:
        lines.append(f"**信息** {job.message}")
    if job.error:
        lines.append(f"**错误** {job.error}")
    if job.duration_s is not None:
        lines.append(f"**耗时** {job.duration_s}s")
    return "\n\n".join(lines)


def _output_paths(job: Job, settings: Settings) -> list[str]:
    paths = []
    for output in job.outputs:
        path = settings.outputs_dir / job.job_id / output.filename
        if path.is_file():
            paths.append(str(path))
    return paths


def _blank_outputs(output_kind: str) -> tuple[Any, ...]:
    if output_kind == "image":
        return (None, [], None)
    return (None, None, None)


def _result_outputs(
    job: Job, settings: Settings, output_kind: str
) -> tuple[Any, Any, Any]:
    paths = _output_paths(job, settings)
    if output_kind == "image":
        return (None, paths, paths or None)
    primary = job.primary_output()
    primary_path = (
        str(settings.outputs_dir / job.job_id / primary.filename) if primary else None
    )
    return (None, primary_path, paths or None)


def _render_output_panel(
    spec: TabSpec, *, key_suffix: str = ""
) -> tuple[Any, Any, Any, Any]:
    gallery = gr.Gallery(
        label="生成结果",
        columns=2,
        height=460,
        visible=spec.output_kind == "image",
        interactive=False,
        key=f"gallery-{key_suffix}" if key_suffix else None,
    )
    video = gr.Video(
        label="生成结果",
        visible=spec.output_kind == "video",
        interactive=False,
        key=f"video-{key_suffix}" if key_suffix else None,
    )
    audio = gr.Audio(
        label="生成结果",
        type="filepath",
        visible=spec.output_kind == "audio",
        interactive=False,
        key=f"audio-{key_suffix}" if key_suffix else None,
    )
    files = gr.File(
        label="下载产物",
        file_count="multiple",
        interactive=False,
        key=f"files-{key_suffix}" if key_suffix else None,
    )
    if spec.output_kind == "image":
        return gallery, video, audio, files
    return gallery, video, audio, files


def _health_markdown(settings: Settings, registry: WorkflowRegistry) -> str:
    counts = registry.kind_counts()
    kind_text = (
        " · ".join(
            f"{KIND_LABELS.get(kind, kind)} {count}"
            for kind, count in sorted(counts.items())
        )
        or "无可用工作流"
    )
    lines = [
        "### ComfyUI Gradio 创作台",
        f"- ComfyUI: `{settings.comfy_base_url}`",
        f"- 工作流: {kind_text}",
        "- 公开接口: `/v1`（OpenAI 兼容） · `/files`（产物下载） · `/health`",
    ]
    errors = registry.load_errors()
    if errors:
        lines.append(
            "- ⚠️ 加载失败: "
            + ", ".join(f"{item.name} ({item.error})" for item in errors)
        )
    return "\n".join(lines)


async def _health_check(
    settings: Settings, registry: WorkflowRegistry, jobs: JobManager
) -> str:
    registry.refresh()
    try:
        await asyncio.wait_for(jobs.client.system_stats(), timeout=3.0)
        comfy = "🟢 已连接"
    except Exception as exc:  # noqa: BLE001
        comfy = f"🔴 无法连接（{type(exc).__name__}）"
    counts = registry.kind_counts()
    kind_text = (
        " · ".join(
            f"{KIND_LABELS.get(kind, kind)} {count}"
            for kind, count in sorted(counts.items())
        )
        or "无可用工作流"
    )
    lines = [
        "### ComfyUI Gradio 创作台",
        f"- ComfyUI: {comfy} · `{settings.comfy_base_url}`",
        f"- 工作流: {kind_text}",
        "- 公开接口: `/v1`（OpenAI 兼容） · `/files`（产物下载） · `/health`",
    ]
    errors = registry.load_errors()
    if errors:
        lines.append(
            "- ⚠️ 加载失败: "
            + ", ".join(f"{item.name} ({item.error})" for item in errors)
        )
    return "\n".join(lines)


def _recent_rows(jobs: JobManager) -> list[list[str]]:
    rows: list[list[str]] = []
    for job in jobs.list_recent(20):
        rows.append(
            [
                job.job_id,
                KIND_LABELS.get(job.kind, job.kind),
                STATUS_LABELS.get(job.status, job.status),
                f"{job.progress}%",
                f"{job.duration_s}s" if job.duration_s is not None else "-",
                job.workflow,
                ", ".join(output.filename for output in job.outputs[:2])
                or job.error
                or "-",
            ]
        )
    return rows


def build_ui(
    *,
    settings: Settings,
    registry: WorkflowRegistry,
    jobs: JobManager,
) -> gr.Blocks:
    registry.refresh()

    with gr.Blocks(title="ComfyUI Gradio 创作台") as demo:
        status = gr.Markdown(
            _health_markdown(settings, registry), elem_classes=["cf-status"]
        )

        with gr.Row():
            refresh = gr.Button("刷新工作流", size="sm")
            api_box = gr.Textbox(
                label="OpenAI Base URL（复制给远程 Agent）",
                value="/v1",
                interactive=False,
                scale=3,
            )

        dropdowns: list[Any] = []
        kind_states: list[Any] = []

        def _create_tab(spec: TabSpec) -> None:
            initial_kind = spec.modes[0][1]

            with gr.Tab(spec.title):
                with gr.Row():
                    mode = gr.Radio(
                        choices=[label for label, _kind in spec.modes],
                        value=spec.modes[0][0],
                        label="模式",
                        visible=len(spec.modes) > 1,
                        scale=1,
                    )
                    kind_state = gr.State(initial_kind)
                    workflow = gr.Dropdown(
                        choices=_kind_choices(registry, initial_kind),
                        value=_first_choice(registry, initial_kind),
                        label="工作流",
                        scale=3,
                    )

                def _on_mode(mode_label: str):
                    kind = dict(spec.modes).get(mode_label, initial_kind)
                    choices = _kind_choices(registry, kind)
                    return (
                        gr.update(
                            choices=choices,
                            value=choices[0][1] if choices else None,
                        ),
                        kind,
                    )

                mode.change(_on_mode, inputs=[mode], outputs=[workflow, kind_state])

                @gr.render(inputs=[workflow, kind_state])
                def _render_form(workflow_id: Optional[str], kind: str):
                    entry: WorkflowEntry | None = (
                        registry.get(workflow_id) if workflow_id else None
                    )
                    if entry is None:
                        gr.Markdown(
                            "当前分类下没有可用工作流。请把 ComfyUI **Export (API)** 格式的 JSON "
                            "放入 `workflows/`，可选地在 `.comfyui2api/` 下添加参数 sidecar，"
                            "然后点击上方「刷新工作流」。"
                        )
                        return

                    if entry.parameter_error:
                        gr.Markdown(
                            "⚠️ sidecar 参数配置无效，将只使用提示词与工作流默认值："
                            f"`{entry.parameter_error}`"
                        )

                    needs_image = kind in _IMAGE_KINDS
                    needs_audio = kind == KIND_DIGITALHUMAN

                    with gr.Row():
                        with gr.Column(scale=3):
                            prompt = gr.Textbox(
                                label=spec.prompt_label,
                                placeholder=spec.prompt_placeholder,
                                lines=4,
                            )
                            negative = gr.Textbox(
                                label="负向提示词（可选）",
                                placeholder="不希望出现的内容…",
                                lines=2,
                                visible=spec.show_negative,
                            )
                            image = gr.File(
                                label="参考图片（可多张，顺序即 主图 / image2 / image3…）",
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                                visible=needs_image,
                                allow_reordering=True,
                            )
                            audio = gr.Audio(
                                label="驱动音频",
                                type="filepath",
                                visible=needs_audio,
                            )

                            side_params: list[tuple[str, Any]] = []
                            if entry.parameter_spec is not None:
                                for name, definition in entry.parameter_spec.parameters.items():
                                    if (
                                        definition.type == "image"
                                        or name in {"inputImgCount", "resizeFlag"}
                                    ):
                                        continue
                                    side_params.append((name, definition))

                            param_components: dict[str, Any] = {}
                            if side_params:
                                with gr.Accordion("工作流参数", open=True):
                                    with gr.Row():
                                        for name, definition in side_params:
                                            with gr.Column(min_width=180):
                                                param_components[name] = _build_param_control(
                                                    name, definition
                                                )

                            submit = gr.Button("开始生成", variant="primary")
                            job_status = gr.Markdown("等待提交…")

                        with gr.Column(scale=4):
                            gallery, video, audio_out, files = _render_output_panel(
                                spec, key_suffix=f"{entry.model_id}-{kind}"
                            )

                    param_names = list(param_components.keys())
                    inputs = [prompt, negative, image, audio, *param_components.values()]

                    async def _run(
                        prompt_text: str,
                        negative_text: str,
                        image_paths: Any,
                        audio_path: Optional[str],
                        *param_values: Any,
                        progress: gr.Progress = gr.Progress(),
                    ):
                        current = registry.get(entry.model_id)
                        if current is None:
                            raise gr.Error("工作流已被删除，请刷新后重试。")
                        if not current.supports(kind):
                            raise gr.Error(
                                f"该工作流不支持 {KIND_LABELS.get(kind, kind)} 模式，请刷新后重试。"
                            )

                        if isinstance(image_paths, str):
                            image_paths = [image_paths]
                        image_paths = [path for path in (image_paths or []) if path]
                        if len(image_paths) > 8:
                            raise gr.Error("参考图片最多 8 张。")

                        if kind in _IMAGE_KINDS and not image_paths:
                            raise gr.Error("这个工作流需要参考图片，请先上传图片。")
                        if kind == KIND_DIGITALHUMAN and not audio_path:
                            raise gr.Error("数字人工作流需要驱动音频，请先上传音频。")

                        params: dict[str, Any] = {}
                        for name, value in zip(param_names, param_values):
                            if value is None:
                                continue
                            if isinstance(value, str) and not value.strip():
                                continue
                            params[name] = value

                        images: list[InputFile] = [
                            InputFile(
                                data=Path(path).read_bytes(),
                                filename=Path(path).name,
                                kind="image",
                            )
                            for path in image_paths
                        ]
                        audio_file = None
                        if audio_path:
                            audio_file = InputFile(
                                data=Path(audio_path).read_bytes(),
                                filename=Path(audio_path).name,
                                kind="audio",
                            )

                        try:
                            job = await jobs.submit(
                                entry=current,
                                prompt=str(prompt_text or "").strip(),
                                negative_prompt=str(negative_text or "").strip(),
                                params=params,
                                images=images,
                                audio=audio_file,
                                kind=kind,
                            )
                        except ValueError as exc:
                            raise gr.Error(str(exc)) from exc

                        empty = _blank_outputs(spec.output_kind)
                        while not job.finished:
                            await asyncio.sleep(0.5)
                            progress(
                                min(max(job.progress, 0) / 100.0, 0.99),
                                desc=(
                                    f"{STATUS_LABELS.get(job.status, job.status)}"
                                    f" · {job.progress}%"
                                ),
                            )
                            yield (_job_status_markdown(job), *empty)

                        progress(1.0, desc="已完成")
                        _status, media, files_value = _result_outputs(
                            job, settings, spec.output_kind
                        )
                        yield (_job_status_markdown(job), media, files_value)

                    if spec.output_kind == "image":
                        outputs = [job_status, gallery, files]
                    elif spec.output_kind == "video":
                        outputs = [job_status, video, files]
                    else:
                        outputs = [job_status, audio_out, files]
                    submit.click(_run, inputs=inputs, outputs=outputs)

                dropdowns.append(workflow)
                kind_states.append(kind_state)

        for tab_spec in TABS:
            _create_tab(tab_spec)

        def _on_refresh(*values: Any):
            registry.refresh()
            current_kinds = list(values[: len(TABS)])
            current_values = list(values[len(TABS) :])
            updates = []
            for kind, selected in zip(current_kinds, current_values):
                choices = _kind_choices(registry, kind)
                ids = {value for _label, value in choices}
                value = (
                    selected
                    if selected in ids
                    else (choices[0][1] if choices else None)
                )
                updates.append(gr.update(choices=choices, value=value))
            return [_health_markdown(settings, registry), *updates]

        refresh.click(
            _on_refresh,
            inputs=[*kind_states, *dropdowns],
            outputs=[status, *dropdowns],
        )

        with gr.Accordion("最近任务", open=False):
            recent = gr.Dataframe(
                headers=["任务 ID", "类型", "状态", "进度", "耗时", "工作流", "输出"],
                datatype=["str"] * 7,
                value=_recent_rows(jobs),
                interactive=False,
                wrap=True,
            )
            refresh_jobs = gr.Button("刷新任务列表", size="sm")
            refresh_jobs.click(lambda: _recent_rows(jobs), outputs=[recent])

        async def _load_health() -> str:
            return await _health_check(settings, registry, jobs)

        demo.load(fn=None, js="() => window.location.origin + '/v1'", outputs=[api_box])
        demo.load(_load_health, outputs=[status])

    return demo
