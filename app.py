from __future__ import annotations

import argparse
import sys

from cf_gradio.config import load_settings, parse_gradio_auth
from cf_gradio.registry import WorkflowRegistry
from cf_gradio.server import launch


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cf-gradio",
        description="把本机 ComfyUI 工作流转成 Gradio 页面 + OpenAI 兼容接口（支持公网 share）。",
    )
    share = parser.add_mutually_exclusive_group()
    share.add_argument("--share", action="store_true", help="生成 gradio.live 公网地址")
    share.add_argument("--no-share", action="store_true", help="不生成公网地址（默认）")
    parser.add_argument("--host", default=None, help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 7860")
    parser.add_argument("--auth", default=None, help="页面登录，格式 user:pass")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--comfy-url", default=None, help="ComfyUI 地址，默认 127.0.0.1:8188")
    parser.add_argument("--env-file", default=None, help="指定 .env 文件路径")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只扫描工作流并打印模型列表，不启动服务",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(env_file=args.env_file)

    overrides = {}
    if args.host:
        overrides["gradio_server_name"] = args.host
    if args.port:
        overrides["gradio_server_port"] = args.port
    if args.auth:
        overrides["gradio_auth"] = parse_gradio_auth(args.auth)
    if args.comfy_url:
        overrides["comfy_base_url"] = args.comfy_url.rstrip("/")
    if args.share:
        overrides["gradio_share"] = True
    if args.no_share:
        overrides["gradio_share"] = False
    if overrides:
        settings = settings.with_overrides(**overrides)

    if args.check:
        registry = WorkflowRegistry(settings.workflows_dir)
        registry.refresh()
        print(f"工作流目录: {settings.workflows_dir}")
        entries = registry.ready_entries()
        if not entries:
            print("没有可用工作流。")
        for entry in entries:
            sidecar = "sidecar ✓" if entry.parameter_spec else "无 sidecar"
            print(
                f"- {entry.model_id:<36} kind={entry.kind:<12} "
                f"file={entry.name} ({sidecar})"
            )
        errors = registry.load_errors()
        if errors:
            print("\n加载失败:")
            for item in errors:
                print(f"- {item.name}: {item.error}")
        return 0 if not errors else 1

    launch(
        settings,
        share=settings.gradio_share,
        inbrowser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
