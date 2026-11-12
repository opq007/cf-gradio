from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .config import Settings, load_settings
from .jobs import JobManager
from .openai_api import install_routes
from .registry import WorkflowRegistry
from .ui import CSS, build_ui

logger = logging.getLogger(__name__)

_GRADIO_PATCHED = False
_ACTIVE_INSTALLER: Callable[[Any], None] | None = None


@dataclass
class Runtime:
    settings: Settings
    registry: WorkflowRegistry
    jobs: JobManager
    demo: Any

    def close(self) -> None:
        try:
            self.demo.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass


def _patch_gradio_app_factory(installer: Callable[[Any], None]) -> None:
    """Inject our routes into every FastAPI app Gradio creates for ``launch()``.

    ``Blocks.launch()`` rebuilds ``blocks.app`` via ``App.create_app(...)`` even
    though a placeholder app already exists, so mounting routes beforehand would
    be discarded. Wrapping the factory keeps the UI and the OpenAI API on one
    server -- and therefore behind the same gradio.live share tunnel.
    """

    global _GRADIO_PATCHED, _ACTIVE_INSTALLER
    _ACTIVE_INSTALLER = installer
    if _GRADIO_PATCHED:
        return

    from gradio.routes import App as GradioApp

    original = GradioApp.create_app
    func = getattr(original, "__func__", original)

    def _create_app(*args: Any, **kwargs: Any):
        app = func(*args, **kwargs)
        installer = _ACTIVE_INSTALLER
        if installer is not None:
            installer(app)
        return app

    GradioApp.create_app = staticmethod(_create_app)
    _GRADIO_PATCHED = True


def create_runtime(settings: Settings) -> Runtime:
    settings.workflows_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    registry = WorkflowRegistry(settings.workflows_dir)
    registry.refresh()
    jobs = JobManager(settings=settings, registry=registry)
    demo = build_ui(settings=settings, registry=registry, jobs=jobs)
    return Runtime(settings=settings, registry=registry, jobs=jobs, demo=demo)


def _probe_api(local_url: str) -> bool:
    url = local_url.rstrip("/") + "/v1/models"
    for _attempt in range(10):
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code in {200, 401}:
                return True
        except Exception:  # noqa: BLE001 - server may still be starting
            pass
        time.sleep(0.2)
    return False


def launch(
    settings: Settings | None = None,
    *,
    share: bool | None = None,
    server_name: str | None = None,
    server_port: int | None = None,
    auth: tuple[str, str] | None = None,
    inbrowser: bool = False,
    block: bool = True,
    quiet: bool = False,
) -> Runtime:
    """Start the Gradio UI + OpenAI API and keep the process alive.

    In a notebook pass ``block=False`` and keep the cell (or a background
    thread) alive; Gradio's share tunnel lives as long as the process does.
    """

    settings = settings or load_settings()
    if share is None:
        share = settings.gradio_share
    if server_name is not None:
        settings = settings.with_overrides(gradio_server_name=server_name)
    if server_port is not None:
        settings = settings.with_overrides(gradio_server_port=server_port)
    if auth is not None:
        settings = settings.with_overrides(gradio_auth=auth)

    if share and settings.gradio_auth is None:
        logger.warning(
            "公开 share 地址未设置页面登录（GRADIO_AUTH）。任何拿到链接的人都能打开界面。"
        )
    if share and not settings.api_token:
        logger.warning(
            "公开 share 地址未设置 API_TOKEN。/v1 接口将对拿到链接的人开放，"
            "建议设置 API_TOKEN 后由远程 Agent 使用 Bearer 调用。"
        )

    runtime = create_runtime(settings)
    registry = runtime.registry
    jobs = runtime.jobs
    demo = runtime.demo

    def installer(app: Any) -> None:
        install_routes(app, settings=settings, registry=registry, jobs=jobs)

    _patch_gradio_app_factory(installer)

    demo.queue(default_concurrency_limit=4)
    demo.launch(
        share=share,
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        auth=settings.gradio_auth,
        auth_message=settings.gradio_auth_message,
        share_server_address=settings.gradio_share_server_address,
        share_server_protocol=settings.gradio_share_server_protocol,
        inbrowser=inbrowser,
        prevent_thread_lock=True,
        quiet=quiet,
        show_error=True,
        footer_links=[],
        theme="soft",
        css=CSS,
    )

    # Safety net for Gradio versions that recreate the app after launch().
    server_app = getattr(demo, "server_app", None) or getattr(demo, "app", None)
    if server_app is not None:
        installer(server_app)

    local_url = str(getattr(demo, "local_url", "") or "").rstrip("/")
    share_url = str(getattr(demo, "share_url", "") or "").rstrip("/")
    if local_url and not _probe_api(local_url):
        logger.warning("本地 /v1 探测失败：OpenAI 路由可能未注入，请查看启动日志。")

    print("")
    print("=" * 68)
    print("  ComfyUI Gradio 创作台已启动")
    print(f"  本地页面 : {local_url or '(未知)'}")
    if share_url:
        print(f"  公网页面 : {share_url}")
        print(f"  OpenAI  : {share_url}/v1")
        print(f"  健康检查 : {share_url}/health")
    else:
        print("  公网页面 : 未开启（--share 可生成 gradio.live 链接）")
        print(f"  OpenAI  : {local_url}/v1")
    print(f"  ComfyUI  : {settings.comfy_base_url}")
    print(f"  工作流   : {settings.workflows_dir}")
    counts = registry.kind_counts()
    print(
        "  已加载   : "
        + (
            " · ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
            or "无"
        )
    )
    print("=" * 68)
    print("")

    if not block:
        return runtime

    stop_event = threading.Event()
    try:
        while not stop_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
    return runtime
