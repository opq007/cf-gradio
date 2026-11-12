from __future__ import annotations

import asyncio
import ipaddress
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Mapping, Optional
from urllib.parse import urlencode, urljoin, urlparse

import httpx
import websockets


class ComfyApiError(RuntimeError):
    pass


def _truncate_text(value: str, *, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _format_http_error(endpoint: str, exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    body = _truncate_text(response.text)
    return (
        f"ComfyUI {endpoint} failed: status={response.status_code}, "
        f"url={exc.request.url}, body={body!r}"
    )


def _join(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _base_host(base_url: str) -> str:
    raw = base_url if "://" in base_url else f"http://{base_url}"
    return (urlparse(raw).hostname or "").strip().lower()


def _should_trust_env(base_url: str) -> bool:
    """Keep loopback traffic off of HTTP(S)_PROXY (common on Windows/Colab)."""

    host = _base_host(base_url)
    if not host:
        return True
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def _ws_url(base_url: str, *, client_id: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    query = urlencode({"clientId": client_id})
    return f"{scheme}://{netloc}{path.rstrip('/')}/ws?{query}"


def _auth_headers(auth_token: str | None, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = dict(extra or {})
    token = (auth_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _connect_ws(url: str, headers: Dict[str, str]) -> Any:
    """Open a websocket on both the new (>=13) and legacy asyncio clients."""

    try:
        from websockets.asyncio.client import connect as ws_connect

        return ws_connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            additional_headers=headers or None,
        )
    except ImportError:  # pragma: no cover - only for websockets < 13
        return websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            extra_headers=headers or None,
        )


@dataclass(frozen=True)
class QueuedPrompt:
    prompt_id: str
    client_id: str
    number: Optional[int] = None


class ComfyUIClient:
    """Minimal ComfyUI HTTP + WebSocket client for a single local instance."""

    def __init__(
        self,
        base_url: str,
        *,
        http_timeout_s: int = 600,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = (auth_token or "").strip() or None
        self._timeout = httpx.Timeout(timeout=http_timeout_s)
        self._client = httpx.AsyncClient(timeout=self._timeout, trust_env=_should_trust_env(self.base_url))
        self._object_info_lock = asyncio.Lock()
        self._object_info_cache: Optional[Dict[str, Any]] = None

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        return _auth_headers(self.auth_token, extra)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def system_stats(self) -> Any:
        url = _join(self.base_url, "/system_stats")
        response = await self._client.get(url, headers=self._headers({"Accept": "application/json"}))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ComfyApiError(_format_http_error("/system_stats", exc)) from exc
        return response.json()

    async def object_info(self, *, force: bool = False) -> Dict[str, Any]:
        async with self._object_info_lock:
            if self._object_info_cache is not None and not force:
                return self._object_info_cache
            url = _join(self.base_url, "/object_info")
            response = await self._client.get(url, headers=self._headers({"Accept": "application/json"}))
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ComfyApiError(_format_http_error("/object_info", exc)) from exc
            payload = response.json()
            if not isinstance(payload, dict):
                raise ComfyApiError(
                    f"Unexpected /object_info response type: {type(payload).__name__}"
                )
            self._object_info_cache = payload
            return payload

    async def queue_prompt(
        self,
        *,
        prompt: Dict[str, Any],
        client_id: str,
        extra_data: Optional[Dict[str, Any]] = None,
        prompt_id: Optional[str] = None,
    ) -> QueuedPrompt:
        payload: Dict[str, Any] = {"prompt": prompt, "client_id": client_id}
        if prompt_id:
            payload["prompt_id"] = prompt_id
        if isinstance(extra_data, dict) and extra_data:
            payload["extra_data"] = extra_data

        url = _join(self.base_url, "/prompt")
        response = await self._client.post(
            url, json=payload, headers=self._headers({"Accept": "application/json"})
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ComfyApiError(_format_http_error("/prompt", exc)) from exc

        data = response.json()
        if not isinstance(data, dict):
            raise ComfyApiError(f"Unexpected /prompt response type: {type(data).__name__}")
        node_errors = data.get("node_errors")
        if isinstance(node_errors, dict) and node_errors:
            raise ComfyApiError(
                "ComfyUI rejected the prompt (node_errors present): "
                + json.dumps(node_errors, ensure_ascii=False, indent=2)
            )
        pid = data.get("prompt_id")
        if not isinstance(pid, str) or not pid.strip():
            raise ComfyApiError(f"Missing prompt_id in /prompt response: {data}")
        number = data.get("number")
        return QueuedPrompt(
            prompt_id=pid.strip(),
            client_id=client_id,
            number=number if isinstance(number, int) else None,
        )

    async def get_history_entry(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        url = _join(self.base_url, f"/history/{prompt_id}")
        response = await self._client.get(
            url, headers=self._headers({"Accept": "application/json"})
        )
        response.raise_for_status()
        return extract_history_entry(response.json(), prompt_id=prompt_id)

    async def get_queue(self) -> Any:
        url = _join(self.base_url, "/queue")
        response = await self._client.get(url, headers=self._headers({"Accept": "application/json"}))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ComfyApiError(_format_http_error("/queue", exc)) from exc
        return response.json()

    async def view_bytes(
        self,
        *,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        url = _join(self.base_url, "/view?" + query)
        response = await self._client.get(url, headers=self._headers({"Accept": "*/*"}))
        response.raise_for_status()
        return response.content

    async def upload_file(
        self,
        *,
        data: bytes,
        filename: str,
        subfolder: str = "",
        folder_type: str = "input",
        overwrite: bool = True,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file into ComfyUI's input dir and return its relative name.

        ComfyUI exposes one generic upload route; audio/video inputs use it too,
        with the original extension preserved.
        """

        url = _join(self.base_url, "/upload/image")
        safe_subfolder = (subfolder or "").strip().replace("\\", "/").strip("/")
        form = {
            "subfolder": safe_subfolder,
            "type": (folder_type or "input").strip(),
            "overwrite": "true" if overwrite else "false",
        }
        files = {"image": (filename or "input.bin", data, content_type)}
        response = await self._client.post(
            url, data=form, files=files, headers=self._headers({"Accept": "application/json"})
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ComfyApiError(_format_http_error("/upload/image", exc)) from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise ComfyApiError(
                f"Unexpected /upload/image response type: {type(payload).__name__}"
            )
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ComfyApiError(f"Missing 'name' in /upload/image response: {payload}")
        resp_sub = payload.get("subfolder") or ""
        if not isinstance(resp_sub, str):
            resp_sub = ""
        resp_sub = resp_sub.strip().replace("\\", "/").strip("/")
        rel = f"{resp_sub}/{name}" if resp_sub else name
        return rel.replace("\\", "/")

    async def ws_events(self, *, client_id: str) -> AsyncIterator[Dict[str, Any]]:
        url = _ws_url(self.base_url, client_id=client_id)
        headers = _auth_headers(self.auth_token)
        async with _connect_ws(url, headers) as ws:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    yield message


def extract_history_entry(data: Any, *, prompt_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict) or not data:
        return None
    nested = data.get(prompt_id)
    if isinstance(nested, dict):
        return nested
    if "outputs" in data or "status" in data:
        return data
    if len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, dict) and ("outputs" in only or "status" in only):
            return only
    return None


def history_entry_error(entry: Mapping[str, Any] | None) -> str | None:
    if not isinstance(entry, dict):
        return None
    status = entry.get("status")
    if not isinstance(status, dict):
        return None
    status_str = str(status.get("status_str") or status.get("status") or "").strip().lower()
    if status_str in {"error", "failed", "interrupted"}:
        messages = status.get("messages")
        if isinstance(messages, list) and messages:
            return f"ComfyUI 执行失败: {messages[-1]}"
        return f"ComfyUI 执行失败 ({status_str})"
    return None


def history_entry_is_complete(entry: Mapping[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    if history_entry_error(entry):
        return False
    status = entry.get("status")
    if isinstance(status, dict):
        if status.get("completed") is True:
            return True
        status_str = str(status.get("status_str") or status.get("status") or "").strip().lower()
        if status_str in {"success", "completed", "complete"}:
            return True
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return False
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for items in node_out.values():
            if isinstance(items, list) and any(
                isinstance(item, dict) and item.get("filename") for item in items
            ):
                return True
    return False
