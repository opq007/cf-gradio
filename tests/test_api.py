from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from cf_gradio.config import load_settings  # noqa: E402
from cf_gradio.jobs import JobManager  # noqa: E402
from cf_gradio.openai_api import install_routes  # noqa: E402
from cf_gradio.registry import WorkflowRegistry  # noqa: E402
from tests.fake_comfy import FakeComfy, MP4_BYTES, PNG_BYTES, WAV_BYTES  # noqa: E402


def _build_client(
    *,
    base_url: str,
    workflows_dir: Path,
    outputs_dir: Path,
    api_token: str = "",
) -> TestClient:
    settings = load_settings(use_env_file=False).with_overrides(
        comfy_base_url=base_url,
        workflows_dir=workflows_dir,
        outputs_dir=outputs_dir,
        api_token=api_token,
        poll_interval_s=0.05,
        timeout_s=30,
        gradio_share=False,
    )
    registry = WorkflowRegistry(settings.workflows_dir)
    registry.refresh()
    jobs = JobManager(settings=settings, registry=registry)
    app = FastAPI()
    install_routes(app, settings=settings, registry=registry, jobs=jobs)
    return TestClient(app)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake = FakeComfy().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake.stop()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.outputs = Path(self.tmp.name) / "outputs"
        self.client = _build_client(
            base_url=self.fake.base_url,
            workflows_dir=ROOT / "workflows",
            outputs_dir=self.outputs,
        )
        # Keep one portal/event loop alive across requests so background jobs
        # created by async endpoints are not cancelled between requests.
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_models(self) -> None:
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        ids = {item["id"] for item in payload["data"]}
        self.assertIn("moss-image-aio", ids)
        self.assertIn("voxcpm2-base", ids)

    def test_images_generations_url_and_file(self) -> None:
        # moss-image-aio is an AIO graph: text-only generation must work without
        # an input image (inputImgCount auto-set to 0).
        response = self.client.post(
            "/v1/images/generations",
            json={
                "model": "moss-image-aio",
                "prompt": "a cat",
                "size": "512x512",
                "steps": 4,
                "seed": 7,
                "n": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertGreaterEqual(len(data), 1)
        url = data[0]["url"]
        self.assertTrue(url.startswith("http://testserver/files/"))

        file_response = self.client.get(url.split("testserver", 1)[1])
        self.assertEqual(file_response.status_code, 200)
        self.assertEqual(file_response.content, PNG_BYTES)

    def test_images_generations_b64_json(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            json={
                "model": "moss-image-aio",
                "prompt": "a dog",
                "response_format": "b64_json",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertTrue(data and data[0]["b64_json"])

    def test_images_generations_async_job(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            headers={"x-comfyui-async": "1"},
            json={"model": "moss-image-aio", "prompt": "a bird"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["job_id"]

        job = self._poll_job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertTrue(job["url"].startswith("http://testserver/files/"))

    def _poll_job(self, job_id: str) -> dict:
        deadline = time.time() + 10
        while time.time() < deadline:
            response = self.client.get(f"/v1/jobs/{job_id}")
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail("job did not finish in time")

    def test_files_only_serves_registered_outputs(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            headers={"x-comfyui-async": "1"},
            json={"model": "moss-image-aio", "prompt": "file scope"},
        )
        job = self._poll_job(response.json()["job_id"])
        filename = job["outputs"][0]["filename"]

        served = self.client.get(f"/files/{job['job_id']}/{filename}")
        self.assertEqual(served.status_code, 200)
        for internal in ("job.json", "prompt.json", "history.json"):
            blocked = self.client.get(f"/files/{job['job_id']}/{internal}")
            self.assertEqual(blocked.status_code, 404, internal)

    def test_images_edits_multipart(self) -> None:
        response = self.client.post(
            "/v1/images/edits",
            data={"model": "moss-image-aio", "prompt": "make it cinematic"},
            files={"image": ("input.png", PNG_BYTES, "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["data"][0]["url"])

    def test_images_edits_multiple_images_fill_count_and_slots(self) -> None:
        response = self.client.post(
            "/v1/images/edits",
            headers={"x-comfyui-async": "1"},
            data={"model": "qwen-image21-turbo-aio", "prompt": "combine them"},
            files=[
                ("image", ("first.png", PNG_BYTES, "image/png")),
                ("image", ("second.png", PNG_BYTES, "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        job = self._poll_job(response.json()["job_id"])
        self.assertEqual(job["status"], "completed")
        prompt = self.fake.prompts[job["prompt_id"]]

        first = prompt["470"]["inputs"]["image"].rsplit("/", 1)[-1]
        second = prompt["544"]["inputs"]["image"].rsplit("/", 1)[-1]
        self.assertTrue(first.endswith("-1-first.png"), first)
        self.assertTrue(second.endswith("-2-second.png"), second)
        self.assertEqual(prompt["537"]["inputs"]["value"], 2)

    def test_aio_text_to_image_count_is_zero(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            headers={"x-comfyui-async": "1"},
            json={"model": "qwen-image21-turbo-aio", "prompt": "text only"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        job = self._poll_job(response.json()["job_id"])
        self.assertEqual(job["kind"], "txt2img")
        prompt = self.fake.prompts[job["prompt_id"]]
        self.assertEqual(prompt["537"]["inputs"]["value"], 0)
        self.assertNotIn("cfgradio/", prompt["470"]["inputs"]["image"])

    def test_audio_speech(self) -> None:
        response = self.client.post(
            "/v1/audio/speech",
            json={
                "model": "voxcpm2-base",
                "input": "你好，世界",
                "voice": "default",
                "speed": 1.1,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, WAV_BYTES)
        self.assertIn("audio/", response.headers.get("content-type", ""))

    def test_video_lifecycle(self) -> None:
        response = self.client.post(
            "/v1/videos",
            data={
                "model": "kaggle-ltx2-5-flf2v",
                "prompt": "camera pans right",
                "seconds": "2",
                "fps": "8",
            },
            files={"input_reference": ("first.png", PNG_BYTES, "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        video = response.json()
        self.assertEqual(video["object"], "video")
        video_id = video["id"]

        deadline = time.time() + 10
        while time.time() < deadline:
            status = self.client.get(f"/v1/videos/{video_id}")
            self.assertEqual(status.status_code, 200, status.text)
            payload = status.json()
            if payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        else:
            self.fail("video did not finish in time")

        self.assertEqual(payload["status"], "completed", payload)
        content = self.client.get(f"/v1/videos/{video_id}/content")
        self.assertEqual(content.status_code, 200, content.text)
        self.assertEqual(content.content, MP4_BYTES)

    def test_video_multiple_images_fill_count_and_last_frame(self) -> None:
        response = self.client.post(
            "/v1/videos",
            data={"model": "kaggle-ltx2-5-flf2v", "prompt": "interpolate"},
            files=[
                ("input_reference", ("first.png", PNG_BYTES, "image/png")),
                ("input_reference", ("last.png", PNG_BYTES, "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        video_id = response.json()["id"]
        job = self._poll_job(video_id[len("video_") :])
        self.assertEqual(job["status"], "completed", job)
        prompt = self.fake.prompts[job["prompt_id"]]

        first = prompt["31"]["inputs"]["image"].rsplit("/", 1)[-1]
        last = prompt["39"]["inputs"]["image"].rsplit("/", 1)[-1]
        self.assertTrue(first.endswith("-1-first.png"), first)
        self.assertTrue(last.endswith("-2-last.png"), last)
        self.assertEqual(prompt["299"]["inputs"]["value"], 2)

    def test_kind_mismatch_and_unknown_model(self) -> None:
        mismatch = self.client.post(
            "/v1/images/generations",
            json={"model": "voxcpm2-base", "prompt": "x"},
        )
        self.assertEqual(mismatch.status_code, 400)
        self.assertEqual(
            mismatch.json()["error"]["code"], "model_kind_mismatch"
        )

        unknown = self.client.post(
            "/v1/images/generations",
            json={"model": "nope", "prompt": "x"},
        )
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["code"], "model_not_found")

    def test_missing_prompt(self) -> None:
        response = self.client.post("/v1/images/generations", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("prompt", response.json()["error"]["message"])


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake = FakeComfy().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake.stop()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.client = _build_client(
            base_url=self.fake.base_url,
            workflows_dir=ROOT / "workflows",
            outputs_dir=Path(self.tmp.name) / "outputs",
            api_token="secret-token",
        )
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_requires_bearer_token(self) -> None:
        unauthorized = self.client.get("/v1/models")
        self.assertEqual(unauthorized.status_code, 401)

        wrong = self.client.get(
            "/v1/models", headers={"Authorization": "Bearer nope"}
        )
        self.assertEqual(wrong.status_code, 401)

        ok = self.client.get(
            "/v1/models", headers={"Authorization": "Bearer secret-token"}
        )
        self.assertEqual(ok.status_code, 200)


DIGITAL_HUMAN_WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "default.png"}},
    "2": {"class_type": "LoadAudio", "inputs": {"audio": "default.wav"}},
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "default"},
        "_meta": {"title": "Positive Prompt"},
    },
    "4": {"class_type": "SaveVideo", "inputs": {"images": ["1", 0]}},
}


class DigitalHumanApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake = FakeComfy().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake.stop()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        workflows = root / "workflows"
        sidecars = workflows / ".comfyui2api"
        sidecars.mkdir(parents=True)
        (workflows / "dh.json").write_text(
            json.dumps(DIGITAL_HUMAN_WORKFLOW), encoding="utf-8"
        )
        (sidecars / "dh.params.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "digitalhuman",
                    "prompt_node": "3.text",
                    "image_node": "1.image",
                    "audio_node": "2.audio",
                    "parameters": {},
                }
            ),
            encoding="utf-8",
        )
        self.client = _build_client(
            base_url=self.fake.base_url,
            workflows_dir=workflows,
            outputs_dir=root / "outputs",
        )
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_digital_human_endpoint(self) -> None:
        response = self.client.post(
            "/v1/digital-human",
            data={"model": "dh", "prompt": "natural talking"},
            files={
                "image": ("face.png", PNG_BYTES, "image/png"),
                "audio": ("speech.wav", WAV_BYTES, "audio/wav"),
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        video_id = response.json()["id"]
        self.assertTrue(video_id.startswith("video_"))

        deadline = time.time() + 10
        while time.time() < deadline:
            status = self.client.get(f"/v1/videos/{video_id}")
            payload = status.json()
            if payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(payload["status"], "completed", payload)

        content = self.client.get(f"/v1/videos/{video_id}/content")
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, MP4_BYTES)


if __name__ == "__main__":
    unittest.main()
