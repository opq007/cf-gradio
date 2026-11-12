from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_gradio.params import (  # noqa: E402
    InvalidParameterError,
    load_workflow_parameter_spec,
    normalize_standard_params,
    public_parameter_spec,
    resolve_standard_overrides,
)
from cf_gradio.workflow import read_json  # noqa: E402


class SidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflows_dir = ROOT / "workflows"
        self.workflow = self.workflows_dir / "moss_image_aio_api.json"
        self.spec = load_workflow_parameter_spec(
            workflows_dir=self.workflows_dir,
            workflow_path=self.workflow,
            expected_kind="img2img",
        )

    def test_resolves_size_steps_seed_and_batch(self) -> None:
        overrides = resolve_standard_overrides(
            workflow_obj=read_json(self.workflow),
            spec=self.spec,
            request_params={"size": "512x768", "steps": 4, "seed": 42, "n": 2},
        )
        mapping = {(node, key): value for node, key, value in overrides}
        self.assertEqual(mapping[("56", "value")], 512)
        self.assertEqual(mapping[("57", "value")], 768)
        self.assertEqual(mapping[("37", "steps")], 4)
        self.assertEqual(mapping[("46", "steps")], 4)
        self.assertEqual(mapping[("72", "seed")], 42)
        self.assertEqual(mapping[("28", "batch_size")], 2)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(InvalidParameterError):
            normalize_standard_params(
                spec=self.spec, request_params={"size": "0x512"}
            )
        with self.assertRaises(InvalidParameterError):
            normalize_standard_params(spec=self.spec, request_params={"steps": 0})

    def test_unknown_params_pass_through(self) -> None:
        params = normalize_standard_params(
            spec=self.spec, request_params={"custom_flag": "x"}
        )
        self.assertEqual(params["custom_flag"], "x")

    def test_public_spec_has_ui_metadata(self) -> None:
        public = public_parameter_spec(self.spec)
        self.assertEqual(public["kind"], "img2img")
        names = {item["name"] for item in public["parameters"]}
        self.assertTrue({"size", "steps", "seed"} <= names)
        size = next(item for item in public["parameters"] if item["name"] == "size")
        self.assertEqual(size["type"], "size")
        self.assertTrue(size["label"])


class AudioSidecarTests(unittest.TestCase):
    def test_voice_and_speed_are_mapped(self) -> None:
        workflows_dir = ROOT / "workflows"
        workflow = workflows_dir / "voxcpm2_base_api.json"
        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir, workflow_path=workflow
        )
        overrides = resolve_standard_overrides(
            workflow_obj=read_json(workflow),
            spec=spec,
            request_params={"voice": "female", "speed": 1.25},
        )
        mapping = {(node, key): value for node, key, value in overrides}
        self.assertEqual(mapping[("7", "string")], "female")
        self.assertEqual(mapping[("19", "speed")], 1.25)


class SecondsToFramesTests(unittest.TestCase):
    def test_transform_uses_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow_path = root / "video.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "x"},
                            "_meta": {"title": "Positive Prompt"},
                        },
                        "2": {"class_type": "SaveVideo", "inputs": {"images": ["1", 0]}},
                        "3": {"class_type": "PrimitiveFloat", "inputs": {"value": 4.0}},
                        "4": {"class_type": "PrimitiveInt", "inputs": {"value": 24}},
                    }
                ),
                encoding="utf-8",
            )
            sidecar_dir = root / ".comfyui2api"
            sidecar_dir.mkdir()
            (sidecar_dir / "video.params.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "kind": "txt2video",
                        "prompt_node": "1.text",
                        "parameters": {
                            "fps": {"type": "int", "maps": [{"target": "4.value"}]},
                            "duration": {
                                "type": "float",
                                "maps": [
                                    {
                                        "target": "3.value",
                                        "transform": "seconds_to_frames",
                                        "fps_param": "fps",
                                        "round": "ceil",
                                    }
                                ],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec = load_workflow_parameter_spec(
                workflows_dir=root, workflow_path=workflow_path
            )
            overrides = resolve_standard_overrides(
                workflow_obj=read_json(workflow_path),
                spec=spec,
                request_params={"fps": 24, "duration": 2.5},
            )
            values = {key: value for _node, key, value in overrides}
            self.assertEqual(values["value"], 60)


if __name__ == "__main__":
    unittest.main()
