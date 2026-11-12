from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class WorkflowFormatError(ValueError):
    pass


KEEP_WORKFLOW_PROMPT_SENTINEL = "-"

KIND_TXT2IMG = "txt2img"
KIND_IMG2IMG = "img2img"
KIND_TXT2VIDEO = "txt2video"
KIND_IMG2VIDEO = "img2video"
KIND_TXT2AUDIO = "txt2audio"
KIND_DIGITALHUMAN = "digitalhuman"
KIND_UNKNOWN = "unknown"

ALL_KINDS = (
    KIND_TXT2IMG,
    KIND_IMG2IMG,
    KIND_TXT2VIDEO,
    KIND_IMG2VIDEO,
    KIND_TXT2AUDIO,
    KIND_DIGITALHUMAN,
)

KIND_LABELS = {
    KIND_TXT2IMG: "文生图",
    KIND_IMG2IMG: "图生图",
    KIND_TXT2VIDEO: "文生视频",
    KIND_IMG2VIDEO: "图生视频",
    KIND_TXT2AUDIO: "语音合成 (TTS)",
    KIND_DIGITALHUMAN: "数字人",
    KIND_UNKNOWN: "未知",
}


def is_keep_workflow_prompt(value: Any) -> bool:
    return str(value or "").strip() == KEEP_WORKFLOW_PROMPT_SENTINEL


def read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def looks_like_prompt_graph(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    hits = 0
    for key, value in list(obj.items())[:10]:
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        if "class_type" in value:
            hits += 1
    return hits > 0


def extract_prompt_and_extra(obj: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if isinstance(obj, dict) and isinstance(obj.get("prompt"), dict):
        prompt = obj["prompt"]
        extra_data: Optional[Dict[str, Any]] = None
        if isinstance(obj.get("extra_data"), dict):
            extra_data = obj["extra_data"]
        elif isinstance(obj.get("extra_pnginfo"), dict):
            extra_data = {"extra_pnginfo": obj["extra_pnginfo"]}
        elif isinstance(obj.get("workflow"), dict):
            extra_data = {"extra_pnginfo": {"workflow": obj["workflow"]}}
        return prompt, extra_data

    if looks_like_prompt_graph(obj):
        return obj, None

    if isinstance(obj, dict) and ("nodes" in obj or "links" in obj):
        raise WorkflowFormatError(
            "检测到 UI 格式工作流（包含 nodes/links）。请在 ComfyUI 使用 File -> Export (API) 导出后重试。"
        )
    raise WorkflowFormatError("无法识别工作流 JSON，需要 ComfyUI API prompt 格式。")


def parse_node_input_ref(raw: str, *, default_input: str = "text") -> Tuple[str, str]:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("节点引用不能为空")
    if "." in value:
        node_id, input_name = value.split(".", 1)
        node_id = node_id.strip()
        input_name = input_name.strip()
        if not node_id or not input_name:
            raise ValueError(f"无效的节点引用: {raw!r}")
        return node_id, input_name
    return value, default_input


def as_str(value: Any) -> str:
    return value if isinstance(value, str) else str(value or "")


def get_node_title(node: Dict[str, Any]) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict):
        title = meta.get("title")
        if isinstance(title, str):
            return title
    return ""


# Text encoder nodes expose the positive/negative prompt under different keys;
# look for the most specific one first so a shared node is not misread.
_PROMPT_KEY_ORDER: Dict[str, Tuple[str, ...]] = {
    "positive": ("positive_prompt", "positive", "text", "value", "prompt", "string"),
    "negative": ("negative_prompt", "negative", "neg", "text", "value", "prompt", "string"),
}

_TARGET = Tuple[str, str, str, str]


def find_text_prompt_targets(
    prompt: Dict[str, Any],
) -> Tuple[List[_TARGET], List[_TARGET]]:
    targets: Dict[str, List[_TARGET]] = {"positive": [], "negative": []}
    generic: List[_TARGET] = []

    def add(kind: str, node_id: str, input_key: str, cls: str, title: str) -> None:
        item = (node_id, input_key, cls, title)
        if item not in targets[kind]:
            targets[kind].append(item)

    def add_generic(node_id: str, input_key: str, cls: str, title: str) -> None:
        item = (node_id, input_key, cls, title)
        if item not in generic:
            generic.append(item)

    def resolve_text_target(start_node_id: str, *, kind: str) -> Optional[_TARGET]:
        current_id = start_node_id
        seen: set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            node = prompt.get(current_id)
            if not isinstance(node, dict):
                return None
            cls = as_str(node.get("class_type"))
            title = get_node_title(node)
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                return None

            next_node_id: Optional[str] = None
            for key in [key for key in _PROMPT_KEY_ORDER[kind] if key in inputs]:
                value = inputs.get(key)
                if isinstance(value, str):
                    return current_id, key, cls, title
                if isinstance(value, list) and value and isinstance(value[0], str):
                    next_node_id = value[0]
                    break
            if not next_node_id:
                return None
            current_id = next_node_id
        return None

    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            continue
        cls = as_str(node.get("class_type"))
        title = get_node_title(node)
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        string_inputs = [
            key for key, value in inputs.items() if isinstance(key, str) and isinstance(value, str)
        ]
        if not string_inputs:
            continue

        title_l = title.lower()
        cls_l = cls.lower()
        is_encode = "textencode" in cls_l
        preferred_key = (
            "text" if isinstance(inputs.get("text"), str) else string_inputs[0]
        )

        if "negative" in title_l or "neg" in title_l:
            add("negative", node_id, preferred_key, cls, title)
            continue
        if "positive" in title_l or "pos" in title_l:
            add("positive", node_id, preferred_key, cls, title)
            continue
        if is_encode:
            add_generic(node_id, preferred_key, cls, title)

    for _node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key in ("positive", "negative"):
            value = inputs.get(key)
            if not (isinstance(value, list) and value and isinstance(value[0], str)):
                continue
            resolved = resolve_text_target(value[0], kind=key)
            if not resolved:
                continue
            node_id, input_key, cls, title = resolved
            add(key, node_id, input_key, cls, title)

    if not targets["positive"]:
        targets["positive"].extend(generic)
    if not targets["negative"]:
        targets["negative"].extend(generic)
    return targets["positive"], targets["negative"]


def find_load_image_targets(prompt: Dict[str, Any]) -> List[_TARGET]:
    return _find_loader_targets(prompt, needle="loadimage", default_input="image")


def find_load_audio_targets(prompt: Dict[str, Any]) -> List[_TARGET]:
    return _find_loader_targets(prompt, needle="loadaudio", default_input="audio")


def _find_loader_targets(
    prompt: Dict[str, Any], *, needle: str, default_input: str
) -> List[_TARGET]:
    candidates: List[_TARGET] = []
    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            continue
        cls = as_str(node.get("class_type"))
        if needle not in cls.lower():
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if isinstance(inputs.get(default_input), str):
            candidates.append((node_id, default_input, cls, get_node_title(node)))
            continue
        string_inputs = [
            key for key, value in inputs.items() if isinstance(key, str) and isinstance(value, str)
        ]
        if not string_inputs:
            continue
        preferred = default_input if default_input in string_inputs else string_inputs[0]
        candidates.append((node_id, preferred, cls, get_node_title(node)))
    return candidates


def pick_unique_target(*, kind: str, candidates: List[_TARGET]) -> Tuple[str, str]:
    if not candidates:
        raise KeyError(f"工作流中找不到{kind}提示词节点。")

    scored: List[Tuple[int, _TARGET]] = []
    for candidate in candidates:
        node_id, input_key, cls, title = candidate
        score = 0
        title_l = title.lower()
        cls_l = cls.lower()
        if input_key == "text":
            score += 10
        if "textencode" in cls_l:
            score += 5
        if kind == "positive":
            if "positive" in title_l or "pos" in title_l:
                score += 100
        elif "negative" in title_l or "neg" in title_l:
            score += 100
        scored.append((score, candidate))

    scored.sort(key=lambda item: (item[0], item[1][0]), reverse=True)
    best_score = scored[0][0]
    best = [candidate for score, candidate in scored if score == best_score]
    if len(best) == 1:
        node_id, input_key, _cls, _title = best[0]
        return node_id, input_key

    lines = [f"无法确定唯一的{kind}提示词节点（候选同分 {best_score}）："]
    for node_id, input_key, cls, title in best[:12]:
        lines.append(f"  - {node_id}.{input_key}  class={cls}  title={title or '(无标题)'}")
    raise KeyError("\n".join(lines))


def pick_unique_load_target(
    *, kind: str, candidates: List[_TARGET], expected_input: str
) -> Tuple[str, str]:
    label = "图片" if kind == "image" else "音频"
    if not candidates:
        raise KeyError(f"工作流中找不到 Load{label} 节点。")
    if len(candidates) == 1:
        node_id, input_key, _cls, _title = candidates[0]
        return node_id, input_key

    scored: List[Tuple[int, _TARGET]] = []
    for candidate in candidates:
        node_id, input_key, _cls, title = candidate
        score = 0
        if "load" in title.lower():
            score += 10
        if input_key == expected_input:
            score += 5
        scored.append((score, candidate))

    scored.sort(key=lambda item: (item[0], item[1][0]), reverse=True)
    best_score = scored[0][0]
    best = [candidate for score, candidate in scored if score == best_score]
    if len(best) == 1:
        node_id, input_key, _cls, _title = best[0]
        return node_id, input_key

    lines = [f"无法确定唯一的{label}输入节点（候选同分 {best_score}）："]
    for node_id, input_key, cls, title in best[:12]:
        lines.append(f"  - {node_id}.{input_key}  class={cls}  title={title or '(无标题)'}")
    raise KeyError("\n".join(lines))


def pick_unique_load_image_target(candidates: List[_TARGET]) -> Tuple[str, str]:
    return pick_unique_load_target(kind="image", candidates=candidates, expected_input="image")


def pick_unique_load_audio_target(candidates: List[_TARGET]) -> Tuple[str, str]:
    return pick_unique_load_target(kind="audio", candidates=candidates, expected_input="audio")


def apply_overrides(prompt: Dict[str, Any], overrides: List[Tuple[str, str, Any]]) -> None:
    for node_id, input_key, value in overrides:
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            raise KeyError(f"工作流中不存在节点 {node_id!r}。")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            node["inputs"] = {}
            inputs = node["inputs"]
        inputs[str(input_key)] = value


def _input_schema_map(schema: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    inputs = schema.get("input")
    if not isinstance(inputs, dict):
        return out
    for section in ("required", "optional"):
        block = inputs.get(section)
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(key, str):
                out[key] = value
    return out


def _allowed_enum_values(spec: Any) -> Optional[set[str]]:
    if not isinstance(spec, list) or not spec:
        return None
    first = spec[0]
    if not isinstance(first, list):
        return None
    values = {item for item in first if isinstance(item, str)}
    return values or None


def normalize_prompt_enum_inputs(
    prompt: Dict[str, Any],
    *,
    object_info: Dict[str, Any],
) -> List[Tuple[str, str, str, str]]:
    """Rewrite enum inputs that carry a path to their basename (model loaders)."""

    changes: List[Tuple[str, str, str, str]] = []
    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            continue
        cls = as_str(node.get("class_type"))
        schema = object_info.get(cls)
        if not cls or not isinstance(schema, dict):
            continue
        input_map = _input_schema_map(schema)
        inputs = node.get("inputs")
        if not input_map or not isinstance(inputs, dict):
            continue
        for input_key, current_value in list(inputs.items()):
            if not isinstance(input_key, str) or not isinstance(current_value, str):
                continue
            allowed = _allowed_enum_values(input_map.get(input_key))
            if not allowed or current_value in allowed:
                continue
            normalized = current_value.replace("\\", "/").rsplit("/", 1)[-1]
            if normalized in allowed and normalized != current_value:
                inputs[input_key] = normalized
                changes.append((node_id, input_key, current_value, normalized))
    return changes


def prune_invalid_orphan_output_nodes(
    prompt: Dict[str, Any],
    *,
    object_info: Dict[str, Any],
) -> List[str]:
    """Drop output nodes that nothing consumes and whose required inputs are missing."""

    consumers: Dict[str, List[Tuple[str, str]]] = {}
    for source_id, node in prompt.items():
        if not isinstance(source_id, str) or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_key, value in inputs.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                consumers.setdefault(value[0], []).append((source_id, str(input_key)))

    removed: List[str] = []
    for node_id, node in list(prompt.items()):
        if consumers.get(node_id) or not isinstance(node, dict):
            continue
        cls = as_str(node.get("class_type"))
        schema = object_info.get(cls)
        if not isinstance(schema, dict) or not schema.get("output_node"):
            continue
        input_block = schema.get("input") if isinstance(schema.get("input"), dict) else {}
        required_inputs = input_block.get("required") if isinstance(input_block, dict) else {}
        if not isinstance(required_inputs, dict):
            continue
        node_inputs = node.get("inputs")
        if not isinstance(node_inputs, dict):
            node_inputs = {}
        if any(key not in node_inputs for key in required_inputs):
            prompt.pop(node_id, None)
            removed.append(node_id)
    return removed


@dataclass(frozen=True)
class WorkflowCapabilities:
    kind: str
    kinds: tuple[str, ...]
    has_load_image: bool
    has_save_image: bool
    has_save_video: bool
    has_save_audio: bool
    has_load_audio: bool


def detect_capabilities(prompt: Dict[str, Any]) -> WorkflowCapabilities:
    has_load_image = False
    has_save_image = False
    has_save_video = False
    has_save_audio = False
    has_load_audio = False

    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        cls = as_str(node.get("class_type")).lower()
        if "loadimage" in cls:
            has_load_image = True
        if "loadaudio" in cls:
            has_load_audio = True
        if "saveimage" in cls:
            has_save_image = True
        if "savevideo" in cls or "createvideo" in cls or "videocombine" in cls:
            has_save_video = True
        if "saveaudio" in cls:
            has_save_audio = True

    kinds: list[str] = []
    if has_save_image:
        # AIO graphs route text-to-image and image-to-image through different
        # branches; both are valid entry points for the same workflow.
        kinds.append(KIND_TXT2IMG)
        if has_load_image:
            kinds.append(KIND_IMG2IMG)
    if has_save_video:
        if has_load_audio:
            # Talking-head / lip-sync pipelines: image + driving audio -> video.
            kinds = [KIND_DIGITALHUMAN]
        else:
            kinds.append(KIND_TXT2VIDEO)
            if has_load_image:
                kinds.append(KIND_IMG2VIDEO)
    elif has_save_audio:
        kinds.append(KIND_TXT2AUDIO)

    if has_save_video and has_load_audio:
        kind = KIND_DIGITALHUMAN
    elif has_save_video and has_load_image:
        kind = KIND_IMG2VIDEO
    elif has_save_video:
        kind = KIND_TXT2VIDEO
    elif has_save_audio:
        kind = KIND_TXT2AUDIO
    elif has_save_image and has_load_image:
        kind = KIND_IMG2IMG
    elif has_save_image:
        kind = KIND_TXT2IMG
    else:
        kind = KIND_UNKNOWN

    if kind != KIND_UNKNOWN and kind not in kinds:
        kinds.insert(0, kind)

    return WorkflowCapabilities(
        kind=kind,
        kinds=tuple(kinds),
        has_load_image=has_load_image,
        has_save_image=has_save_image,
        has_save_video=has_save_video,
        has_save_audio=has_save_audio,
        has_load_audio=has_load_audio,
    )


def prepare_prompt(
    *,
    workflow_obj: Any,
    positive_prompt: Optional[str],
    negative_prompt: Optional[str],
    positive_prompt_node: Optional[str],
    negative_prompt_node: Optional[str],
    image: Optional[str],
    image_node: Optional[str],
    audio: Optional[str] = None,
    audio_node: Optional[str] = None,
    overrides: Optional[List[Tuple[str, str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Tuple[str, str, Any]], Dict[str, List[Dict[str, Any]]]]:
    prompt, extra_data = extract_prompt_and_extra(workflow_obj)

    prompt_overrides: List[Tuple[str, str, Any]] = []
    selected_targets: Dict[str, List[Tuple[str, str]]] = {"positive": [], "negative": []}
    apply_positive = bool(positive_prompt) and not is_keep_workflow_prompt(positive_prompt)
    apply_negative = bool(negative_prompt) and not is_keep_workflow_prompt(negative_prompt)
    need_positive_auto = apply_positive and not (positive_prompt_node or "").strip()
    need_negative_auto = apply_negative and not (negative_prompt_node or "").strip()
    need_image_auto = bool(image) and not (image_node or "").strip()
    need_audio_auto = bool(audio) and not (audio_node or "").strip()

    pos_candidates: List[_TARGET] = []
    neg_candidates: List[_TARGET] = []
    image_candidates: List[_TARGET] = []
    audio_candidates: List[_TARGET] = []
    if need_positive_auto or need_negative_auto:
        pos_candidates, neg_candidates = find_text_prompt_targets(prompt)
    if need_image_auto:
        image_candidates = find_load_image_targets(prompt)
    if need_audio_auto:
        audio_candidates = find_load_audio_targets(prompt)

    if apply_positive:
        node_ref = (positive_prompt_node or "").strip()
        if node_ref:
            node_id, input_key = parse_node_input_ref(node_ref, default_input="text")
        else:
            node_id, input_key = pick_unique_target(kind="positive", candidates=pos_candidates)
        prompt_overrides.append((node_id, input_key, str(positive_prompt)))
        selected_targets["positive"].append((node_id, input_key))

    if apply_negative:
        node_ref = (negative_prompt_node or "").strip()
        if node_ref:
            node_id, input_key = parse_node_input_ref(node_ref, default_input="text")
        else:
            node_id, input_key = pick_unique_target(kind="negative", candidates=neg_candidates)
        prompt_overrides.append((node_id, input_key, str(negative_prompt)))
        selected_targets["negative"].append((node_id, input_key))

    if image:
        node_ref = (image_node or "").strip()
        if node_ref:
            node_id, input_key = parse_node_input_ref(node_ref, default_input="image")
        else:
            node_id, input_key = pick_unique_load_image_target(image_candidates)
        prompt_overrides.append((node_id, input_key, str(image)))

    if audio:
        node_ref = (audio_node or "").strip()
        if node_ref:
            node_id, input_key = parse_node_input_ref(node_ref, default_input="audio")
        else:
            node_id, input_key = pick_unique_load_audio_target(audio_candidates)
        prompt_overrides.append((node_id, input_key, str(audio)))

    combined_overrides = prompt_overrides + list(overrides or [])
    if combined_overrides:
        apply_overrides(prompt, combined_overrides)

    prompt_trace: Dict[str, List[Dict[str, Any]]] = {}
    for kind, refs in selected_targets.items():
        if not refs:
            continue
        items: List[Dict[str, Any]] = []
        for node_id, input_key in refs:
            node = prompt.get(node_id)
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                continue
            items.append(
                {
                    "node_id": node_id,
                    "input_key": input_key,
                    "class_type": as_str(node.get("class_type")) if isinstance(node, dict) else None,
                    "title": get_node_title(node) if isinstance(node, dict) else None,
                    "value": inputs.get(input_key),
                }
            )
        if items:
            prompt_trace[kind] = items
    return prompt, extra_data, combined_overrides, prompt_trace


OUTPUT_FOLDER_TYPE = "output"


def iter_file_outputs(
    history_entry: Dict[str, Any],
) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """Yield files a save node wrote to ComfyUI's output folder.

    Preview nodes write to temp and node packs may republish their own copies
    under other folder types; those are not deliverables and are skipped.
    """

    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return
    for node_id, out in outputs.items():
        if not isinstance(node_id, str) or not isinstance(out, dict):
            continue
        for output_key, items in out.items():
            if not isinstance(items, list) or not items:
                continue
            for fileinfo in items:
                if not isinstance(fileinfo, dict):
                    continue
                filename = fileinfo.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                folder_type = fileinfo.get("type", OUTPUT_FOLDER_TYPE)
                if folder_type != OUTPUT_FOLDER_TYPE:
                    continue
                subfolder = fileinfo.get("subfolder")
                yield node_id, str(output_key), {
                    **fileinfo,
                    "filename": filename,
                    "subfolder": subfolder if isinstance(subfolder, str) else "",
                    "type": folder_type,
                }
