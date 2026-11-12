from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else str(value).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name, "")
    if not raw:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = _env_str(name, "")
    return Path(raw).expanduser() if raw else default


def parse_gradio_auth(value: str) -> Optional[tuple[str, str]]:
    raw = (value or "").strip()
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError("GRADIO_AUTH must look like 'username:password'")
    username, password = raw.split(":", 1)
    username = username.strip()
    password = password.strip()
    if not username or not password:
        raise ValueError("GRADIO_AUTH must look like 'username:password'")
    return username, password


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT

    comfy_base_url: str = "http://127.0.0.1:8188"
    comfy_auth_token: str = ""

    workflows_dir: Path = PROJECT_ROOT / "workflows"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    api_token: str = ""
    public_base_url: str = ""

    timeout_s: int = 3600
    http_timeout_s: int = 600
    poll_interval_s: float = 0.5
    max_upload_mb: int = 30
    max_image_bytes: int = 20_000_000
    allow_private_image_urls: bool = False
    input_subdir: str = "cfgradio"

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    gradio_share: bool = False
    gradio_auth: Optional[tuple[str, str]] = None
    gradio_auth_message: str = "请输入访问账号"
    gradio_share_server_address: Optional[str] = None
    gradio_share_server_protocol: Optional[str] = None

    @property
    def max_upload_bytes(self) -> int:
        return max(1, int(self.max_upload_mb)) * 1024 * 1024

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)


def load_settings(
    *,
    env_file: str | Path | None = None,
    use_env_file: bool = True,
) -> Settings:
    """Read configuration from environment variables (optionally a .env file).

    Explicit environment variables always win: the .env file is loaded with
    ``override=False`` so Kaggle/notebook cells can export values before import.
    """

    if use_env_file and load_dotenv is not None:
        candidate = Path(env_file).expanduser() if env_file else PROJECT_ROOT / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)

    auth_raw = _env_str("GRADIO_AUTH", "")

    settings = Settings(
        project_root=PROJECT_ROOT,
        comfy_base_url=_env_str("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/"),
        comfy_auth_token=_env_str("COMFYUI_AUTH_TOKEN", ""),
        workflows_dir=_env_path("WORKFLOWS_DIR", PROJECT_ROOT / "workflows").resolve(),
        outputs_dir=_env_path("OUTPUTS_DIR", PROJECT_ROOT / "outputs").resolve(),
        api_token=_env_str("API_TOKEN", ""),
        public_base_url=_env_str("PUBLIC_BASE_URL", "").rstrip("/"),
        timeout_s=max(1, _env_int("TIMEOUT_S", 3600)),
        http_timeout_s=max(1, _env_int("HTTP_TIMEOUT_S", 600)),
        poll_interval_s=max(0.1, _env_float("POLL_INTERVAL_S", 0.5)),
        max_upload_mb=max(1, _env_int("MAX_UPLOAD_MB", 30)),
        max_image_bytes=max(1, _env_int("MAX_IMAGE_BYTES", 20_000_000)),
        allow_private_image_urls=_env_bool("ALLOW_PRIVATE_IMAGE_URLS", False),
        input_subdir=_env_str("INPUT_SUBDIR", "cfgradio").strip("/\\") or "cfgradio",
        gradio_server_name=_env_str("GRADIO_SERVER_NAME", "0.0.0.0"),
        gradio_server_port=max(1, _env_int("GRADIO_SERVER_PORT", 7860)),
        gradio_share=_env_bool("GRADIO_SHARE", False),
        gradio_auth=parse_gradio_auth(auth_raw),
        gradio_auth_message=_env_str("GRADIO_AUTH_MESSAGE", "请输入访问账号"),
        gradio_share_server_address=_env_str("GRADIO_SHARE_SERVER_ADDRESS", "") or None,
        gradio_share_server_protocol=_env_str("GRADIO_SHARE_SERVER_PROTOCOL", "") or None,
    )
    return settings
