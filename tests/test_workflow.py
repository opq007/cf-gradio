from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_gradio.registry import WorkflowRegistry  # noqa: E402
from cf_gradio.workflow import (  # noqa: E402
    KIND_DIGITALHUMAN,
    KIND_IMG2IMG,
    KIND_IMG2VIDEO,
    KIND_TXT2AUDIO,
    KIND_TXT2IMG,
    detect_capabilities,
    extract_prompt_and_extra,
    find_load_audio_targets,
    find_load_image_targets,
    find_text_prompt_targets,
    prepare_prompt,
)


DIGITAL_HUMAN_WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "default.png"}},
    "2": {"class_type": "LoadAudio", "inputs": {"audio": "default.wav"}},
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "default prompt"},
        "_meta": {"title": "Positive Prompt"},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad"},
        "_meta": {"title": "Negative Prompt"},
    },
    "5": {"class_type": "SaveVideo", "inputs": {"images": ["1", 0]}},
}


class WorkflowRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = WorkflowRegistry(ROOT / "workflows")
        self.registry.refresh()

    def test_all_bundled_workflows_load(self) -> None:
        entries = self.registry.ready_entries()
        self.assertEqual(len(entries), 5)
        self.assertEqual(self.registry.load_errors(), [])
        kinds = {entry.kind for entry in entries}
        self.assertEqual(
            kinds,
            {KIND_IMG2IMG, KIND_IMG2VIDEO, KIND_TXT2AUDIO},
        )

    def test_model_ids_are_unique_and_readable(self) -> None:
        ids = [entry.model_id for entry in self.registry.ready_entries()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("moss-image-aio", ids)
        self.assertIn("voxcpm2-base", ids)

    def test_lookup_accepts_filename_and_model_id(self) -> None:
        entry = self.registry.get("moss-image-aio")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "moss_image_aio_api.json")
        self.assertIs(self.registry.get("moss_image_aio_api"), entry)

    def test_aio_image_workflow_supports_both_image_kinds(self) -> None:
        entry = self.registry.get("moss-image-aio")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.supports(KIND_TXT2IMG))
        self.assertTrue(entry.supports(KIND_IMG2IMG))
        self.assertIn(entry, self.registry.by_kind({KIND_TXT2IMG}))

    def test_every_bundled_sidecar_parses(self) -> None:
        for entry in self.registry.ready_entries():
            self.assertIsNotNone(entry.parameter_spec, entry.name)
            self.assertIsNone(entry.parameter_error, entry.name)


class CapabilityTests(unittest.TestCase):
    def test_digital_human_needs_video_and_audio(self) -> None:
        caps = detect_capabilities(DIGITAL_HUMAN_WORKFLOW)
        self.assertTrue(caps.has_load_audio)
        self.assertTrue(caps.has_save_video)
        self.assertEqual(caps.kind, KIND_DIGITALHUMAN)
        self.assertEqual(caps.kinds, (KIND_DIGITALHUMAN,))

    def test_txt2img_and_img2img(self) -> None:
        txt = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        }
        self.assertEqual(detect_capabilities(txt).kind, KIND_TXT2IMG)
        img = dict(txt)
        img["3"] = {"class_type": "LoadImage", "inputs": {"image": "x.png"}}
        self.assertEqual(detect_capabilities(img).kind, KIND_IMG2IMG)

    def test_txt2audio(self) -> None:
        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a"}},
            "2": {"class_type": "SaveAudio", "inputs": {"audio": ["1", 0]}},
        }
        self.assertEqual(detect_capabilities(workflow).kind, KIND_TXT2AUDIO)


class PreparePromptTests(unittest.TestCase):
    def test_sets_prompt_image_and_audio_targets(self) -> None:
        prompt, _extra, applied, trace = prepare_prompt(
            workflow_obj=DIGITAL_HUMAN_WORKFLOW,
            positive_prompt="hello",
            negative_prompt="ugly",
            positive_prompt_node=None,
            negative_prompt_node=None,
            image="cfgradio/in.png",
            image_node=None,
            audio="cfgradio/in.wav",
            audio_node=None,
        )
        self.assertEqual(prompt["3"]["inputs"]["text"], "hello")
        self.assertEqual(prompt["4"]["inputs"]["text"], "ugly")
        self.assertEqual(prompt["1"]["inputs"]["image"], "cfgradio/in.png")
        self.assertEqual(prompt["2"]["inputs"]["audio"], "cfgradio/in.wav")
        self.assertIn("positive", trace)
        self.assertEqual(len(applied), 4)

    def test_keep_workflow_prompt_sentinel(self) -> None:
        prompt, _extra, applied, trace = prepare_prompt(
            workflow_obj=DIGITAL_HUMAN_WORKFLOW,
            positive_prompt="-",
            negative_prompt=None,
            positive_prompt_node=None,
            negative_prompt_node=None,
            image=None,
            image_node=None,
            audio=None,
            audio_node=None,
        )
        self.assertEqual(prompt["3"]["inputs"]["text"], "default prompt")
        self.assertEqual(applied, [])
        self.assertEqual(trace, {})

    def test_finds_loader_targets(self) -> None:
        prompt, _extra = extract_prompt_and_extra(DIGITAL_HUMAN_WORKFLOW)
        positives, negatives = find_text_prompt_targets(prompt)
        self.assertEqual(positives[0][:2], ("3", "text"))
        self.assertEqual(negatives[0][:2], ("4", "text"))
        self.assertEqual(find_load_image_targets(prompt)[0][:2], ("1", "image"))
        self.assertEqual(find_load_audio_targets(prompt)[0][:2], ("2", "audio"))


if __name__ == "__main__":
    unittest.main()
