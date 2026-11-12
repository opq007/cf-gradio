from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
)
MP4_BYTES = (
    b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    b"\x00\x00\x00\x08free"
)
WAV_BYTES = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


class FakeComfy:
    """Tiny in-process ComfyUI stand-in for API/end-to-end tests."""

    def __init__(self) -> None:
        self.prompts: dict[str, dict[str, Any]] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> "FakeComfy":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # silence
                return

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload: Any, status: int = 200) -> None:
                self._send(
                    status,
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/system_stats":
                    self._json({"system": {"comfyui_version": "fake"}})
                    return
                if parsed.path == "/object_info":
                    self._json({})
                    return
                if parsed.path == "/queue":
                    self._json({"queue_running": [], "queue_pending": []})
                    return
                if parsed.path.startswith("/history/"):
                    prompt_id = parsed.path.rsplit("/", 1)[-1]
                    with fake._lock:
                        prompt = fake.prompts.get(prompt_id)
                    if prompt is None:
                        self._json({})
                        return
                    filename, content_type = fake.output_for(prompt)
                    self._json(
                        {
                            prompt_id: {
                                "status": {"status_str": "success", "completed": True},
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {
                                                "filename": filename,
                                                "subfolder": "",
                                                "type": "output",
                                            }
                                        ]
                                    }
                                },
                            }
                        }
                    )
                    return
                if parsed.path == "/view":
                    query = parse_qs(parsed.query)
                    filename = (query.get("filename") or ["out.png"])[0]
                    if filename.endswith(".mp4"):
                        self._send(200, MP4_BYTES, "video/mp4")
                    elif filename.endswith(".wav"):
                        self._send(200, WAV_BYTES, "audio/wav")
                    else:
                        self._send(200, PNG_BYTES, "image/png")
                    return
                self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                parsed = urlparse(self.path)
                if parsed.path == "/prompt":
                    payload = json.loads(body or b"{}")
                    prompt = payload.get("prompt") or {}
                    with fake._lock:
                        fake._counter += 1
                        prompt_id = f"prompt-{fake._counter}"
                        fake.prompts[prompt_id] = prompt
                        number = fake._counter
                    self._json({"prompt_id": prompt_id, "number": number})
                    return
                if parsed.path == "/upload/image":
                    match = re.search(rb'filename="([^"]+)"', body)
                    name = match.group(1).decode("utf-8", "replace") if match else "upload.bin"
                    subfolder_match = re.search(
                        rb'name="subfolder"\r\n\r\n([^\r\n]+)', body
                    )
                    subfolder = (
                        subfolder_match.group(1).decode("utf-8", "replace")
                        if subfolder_match
                        else ""
                    )
                    self._json({"name": name, "subfolder": subfolder, "type": "input"})
                    return
                self._json({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @staticmethod
    def output_for(prompt: dict[str, Any]) -> tuple[str, str]:
        classes = {
            str(node.get("class_type") or "").lower()
            for node in prompt.values()
            if isinstance(node, dict)
        }
        if any("saveaudio" in cls for cls in classes):
            return "output.wav", "audio/wav"
        if any(
            "savevideo" in cls or "createvideo" in cls or "videocombine" in cls
            for cls in classes
        ):
            return "output.mp4", "video/mp4"
        return "output.png", "image/png"

    @property
    def base_url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_port}"

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
