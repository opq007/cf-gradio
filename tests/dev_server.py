"""Local dev server: cf-gradio against a fake ComfyUI (no GPU needed).

Usage:
    python tests/dev_server.py [--port 7864] [--share]

Useful for clicking through the Gradio UI. Not imported by the app.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_gradio.config import load_settings  # noqa: E402
from cf_gradio.server import launch  # noqa: E402
from tests.fake_comfy import FakeComfy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    fake = FakeComfy().start()
    outputs = Path(tempfile.mkdtemp(prefix="cf-gradio-dev-")) / "outputs"
    settings = load_settings(use_env_file=False).with_overrides(
        comfy_base_url=fake.base_url,
        outputs_dir=outputs,
        gradio_server_name="127.0.0.1",
        gradio_server_port=args.port,
        gradio_share=args.share,
        gradio_auth=None,
    )
    try:
        launch(settings, share=args.share, block=True, inbrowser=False)
    finally:
        fake.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
