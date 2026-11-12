from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .workflow import (
    as_str,
    extract_prompt_and_extra,
    get_node_title,
    parse_node_input_ref,
    read_json,
)


class InvalidParameterError(ValueError):
    pass


AUDIO_PARAMETER_ORDER = ("voice", "speed", "instructions")

STANDARD_PARAMETER_ORDER = (
    "size",
    "width",
    "height",
    "n",
    "steps",
    "cfg",
    "seed",
    "fps",
    "duration",
    "frames",
    *AUDIO_PARAMETER_ORDER,
)

PARAMETER_LABELS = {
    "size": "尺寸 (宽x高)",
    "width": "宽度",
    "height": "高度",
    "n": "数量",
    "steps": "步数",
    "cfg": "CFG",
    "seed": "随机种子 (-1 随机)",
    "fps": "帧率",
    "duration": "时长 (秒)",
    "frames": "帧数",
    "voice": "音色",
    "speed": "语速",
    "instructions": "风格指令",
    "prompt_text": "参考音频文本",
    "inputImgCount": "输入图片数",
}

PARAMETER_DESCRIPTIONS = {
    "size": "图片/视频尺寸，格式 宽x高。",
    "width": "输出宽度。",
    "height": "输出高度。",
    "n": "生成数量 / latent batch size。",
    "steps": "采样步数。",
    "cfg": "CFG / guidance 强度。",
    "seed": "随机种子；留空或 -1 时每次随机。",
    "fps": "视频帧率。",
    "duration": "视频时长（秒）。",
    "frames": "视频总帧数。",
    "voice": "音色 id（映射到 sidecar 指定的参考音频或节点参数）。",
    "speed": "语速倍率（OpenAI 范围 0.25-4）。",
    "instructions": "音色/风格自然语言描述。",
    "prompt_text": "参考音频对应的文本，切换 voice 时同步修改。",
    "inputImgCount": "输入图片数量，由请求自动推导。",
}


@dataclass(frozen=True)
class WorkflowParamSelector:
    class_type: str = ""
    title: str = ""
    input_key: str = ""


@dataclass(frozen=True)
class WorkflowParamTarget:
    ref: str = ""
    selector: WorkflowParamSelector | None = None
    part: str = ""
    transform: str = "direct"
    fps_param: str = "fps"
    round_mode: str = "round"


@dataclass(frozen=True)
class WorkflowParameterDefinition:
    name: str
    type: str
    default: Any = None
    description: str = ""
    required: bool = False
    minimum: Any = None
    maximum: Any = None
    maps: tuple[WorkflowParamTarget, ...] = ()


@dataclass(frozen=True)
class WorkflowParameterSpec:
    version: int
    kind: str
    parameters: dict[str, WorkflowParameterDefinition]
    path: Path
    prompt_node: str = ""
    negative_prompt_node: str = ""
    negative_prompt_node_with_image: str = ""
    image_node: str = ""
    audio_node: str = ""
    model: str = ""
    display_name: str = ""

    def negative_prompt_node_for(self, *, image_count: int = 0) -> str:
        if image_count > 0 and self.negative_prompt_node_with_image:
            return self.negative_prompt_node_with_image
        return self.negative_prompt_node


def parameter_sidecar_dir(workflows_dir: Path) -> Path:
    return workflows_dir / ".comfyui2api"


def parameter_sidecar_path(workflows_dir: Path, workflow_path: Path) -> Path:
    return parameter_sidecar_dir(workflows_dir) / f"{workflow_path.stem}.params.json"


def _as_mapping(obj: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{context} 必须是对象")
    return obj


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _parse_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width = int(value[0])
        height = int(value[1])
    else:
        text = _normalize_string(value).lower().replace("*", "x")
        if "x" not in text:
            raise ValueError(f"无效的尺寸值: {value!r}")
        left, right = text.split("x", 1)
        width = int(left.strip())
        height = int(right.strip())
    if width <= 0 or height <= 0:
        raise ValueError(f"无效的尺寸值: {value!r}")
    return width, height


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"无效的布尔值: {value!r}")
    text = _normalize_string(value).lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"无效的布尔值: {value!r}")


def apply_img2img_resize_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Force ``resizeFlag`` from size/width/height for image-editing workflows.

    Several AIO image-edit graphs only resize the reference image when this
    flag is true. A size (or a complete width/height pair) turns it on; an
    incomplete pair is dropped so it cannot half-apply.
    """

    out = dict(params or {})
    has_size = bool(_normalize_string(out.get("size")))
    has_width = out.get("width") not in (None, "")
    has_height = out.get("height") not in (None, "")
    if has_size or (has_width and has_height):
        out["resizeFlag"] = True
        return out
    out["resizeFlag"] = False
    if has_width != has_height:
        out.pop("width", None)
        out.pop("height", None)
    return out


def normalize_parameter_value(definition: WorkflowParameterDefinition, raw_value: Any) -> Any:
    ptype = definition.type
    if ptype == "size":
        value: Any = _parse_size(raw_value)
        if definition.minimum is not None:
            min_width, min_height = _parse_size(definition.minimum)
            if value[0] < min_width or value[1] < min_height:
                raise ValueError(f"参数 {definition.name!r} 必须 >= {definition.minimum}")
        if definition.maximum is not None:
            max_width, max_height = _parse_size(definition.maximum)
            if value[0] > max_width or value[1] > max_height:
                raise ValueError(f"参数 {definition.name!r} 必须 <= {definition.maximum}")
        return value
    if ptype == "int":
        try:
            value = int(raw_value)
        except OverflowError as exc:
            raise ValueError(f"参数 {definition.name!r} 必须是有限整数") from exc
    elif ptype == "float":
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"参数 {definition.name!r} 必须是有限数字")
    elif ptype == "bool":
        value = _parse_bool(raw_value)
    elif ptype in {"string", "image"}:
        value = _normalize_string(raw_value)
    else:
        raise ValueError(f"不支持的参数类型: {ptype}")

    if definition.minimum is not None and value < definition.minimum:
        raise ValueError(f"参数 {definition.name!r} 必须 >= {definition.minimum}")
    if definition.maximum is not None and value > definition.maximum:
        raise ValueError(f"参数 {definition.name!r} 必须 <= {definition.maximum}")
    return value


def _parse_selector(
    raw: Any, *, parameter_name: str, map_index: int
) -> WorkflowParamSelector | None:
    if raw is None:
        return None
    obj = _as_mapping(raw, context=f"parameters.{parameter_name}.maps[{map_index}].selector")
    selector = WorkflowParamSelector(
        class_type=_normalize_string(obj.get("class_type")),
        title=_normalize_string(obj.get("title")),
        input_key=_normalize_string(obj.get("input_key")),
    )
    if not selector.class_type and not selector.title and not selector.input_key:
        raise ValueError(
            f"parameters.{parameter_name}.maps[{map_index}].selector 不能为空"
        )
    return selector


def _parse_target(
    raw: Any, *, parameter_name: str, map_index: int
) -> tuple[str, WorkflowParamSelector | None]:
    if isinstance(raw, str):
        return _normalize_string(raw), None
    if raw is None:
        return "", None
    obj = _as_mapping(raw, context=f"parameters.{parameter_name}.maps[{map_index}].target")
    return _normalize_string(obj.get("ref")), _parse_selector(
        obj.get("selector"), parameter_name=parameter_name, map_index=map_index
    )


def _parse_map(raw: Any, *, parameter_name: str, map_index: int) -> WorkflowParamTarget:
    obj = _as_mapping(raw, context=f"parameters.{parameter_name}.maps[{map_index}]")
    ref = _normalize_string(obj.get("ref"))
    selector = _parse_selector(
        obj.get("selector"), parameter_name=parameter_name, map_index=map_index
    )

    target_value = obj.get("target")
    if target_value is not None:
        target_ref, target_selector = _parse_target(
            target_value, parameter_name=parameter_name, map_index=map_index
        )
        if target_ref:
            ref = target_ref
        if target_selector is not None:
            selector = target_selector

    if not ref and selector is None:
        raise ValueError(f"parameters.{parameter_name}.maps[{map_index}] 需要 target 或 ref")

    part = _normalize_string(obj.get("part")).lower()
    if part and part not in {"width", "height"}:
        raise ValueError(f"参数 {parameter_name!r} 不支持 part={part!r}")

    transform = _normalize_string(obj.get("transform") or "direct").lower()
    if transform not in {"direct", "seconds_to_frames"}:
        raise ValueError(f"参数 {parameter_name!r} 不支持 transform={transform!r}")

    round_mode = _normalize_string(obj.get("round") or "round").lower()
    if round_mode not in {"round", "ceil", "floor", "int"}:
        raise ValueError(f"参数 {parameter_name!r} 不支持 round={round_mode!r}")

    return WorkflowParamTarget(
        ref=ref,
        selector=selector,
        part=part,
        transform=transform,
        fps_param=_normalize_string(obj.get("fps_param") or "fps") or "fps",
        round_mode=round_mode,
    )


def load_workflow_parameter_spec(
    *,
    workflows_dir: Path,
    workflow_path: Path,
    expected_kind: str = "",
    expected_kinds: Iterable[str] = (),
) -> WorkflowParameterSpec | None:
    path = parameter_sidecar_path(workflows_dir, workflow_path)
    if not path.exists():
        return None

    obj = _as_mapping(read_json(path), context=str(path))
    version = int(obj.get("version") or 1)
    if version != 1:
        raise ValueError(f"不支持的参数映射版本: {version}")

    expected_tuple = tuple(expected_kinds)
    fallback_kind = expected_kind or (expected_tuple[0] if expected_tuple else "")
    kind = _normalize_string(obj.get("kind") or fallback_kind or "unknown")
    allowed_kinds = {expected_kind} if expected_kind else set(expected_tuple)
    if kind and allowed_kinds and kind not in allowed_kinds:
        raise ValueError(
            f"参数映射 kind 不匹配: 期望 {', '.join(sorted(allowed_kinds))}, 实际 {kind!r}"
        )

    raw_parameters = _as_mapping(obj.get("parameters") or {}, context=f"{path}.parameters")
    parameters: dict[str, WorkflowParameterDefinition] = {}
    for name, raw_definition in raw_parameters.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("参数名必须是非空字符串")
        param_name = name.strip()
        definition_obj = _as_mapping(raw_definition, context=f"parameters.{param_name}")
        ptype = _normalize_string(definition_obj.get("type") or "string").lower()
        maps_raw = definition_obj.get("maps") or []
        if not isinstance(maps_raw, list) or not maps_raw:
            raise ValueError(f"parameters.{param_name}.maps 必须是非空数组")
        maps = tuple(
            _parse_map(item, parameter_name=param_name, map_index=index)
            for index, item in enumerate(maps_raw)
        )
        definition = WorkflowParameterDefinition(
            name=param_name,
            type=ptype,
            default=definition_obj.get("default"),
            description=_normalize_string(definition_obj.get("description")),
            required=bool(definition_obj.get("required") is True),
            minimum=definition_obj.get("minimum"),
            maximum=definition_obj.get("maximum"),
            maps=maps,
        )
        if definition.default is not None:
            normalize_parameter_value(definition, definition.default)
        parameters[param_name] = definition

    return WorkflowParameterSpec(
        version=version,
        kind=kind,
        parameters=parameters,
        path=path.resolve(),
        prompt_node=_normalize_string(obj.get("prompt_node")),
        negative_prompt_node=_normalize_string(obj.get("negative_prompt_node")),
        negative_prompt_node_with_image=_normalize_string(
            obj.get("negative_prompt_node_with_image")
        ),
        image_node=_normalize_string(obj.get("image_node")),
        audio_node=_normalize_string(obj.get("audio_node")),
        model=_normalize_string(obj.get("model")),
        display_name=_normalize_string(obj.get("display_name")),
    )


def _ordered_parameter_names(parameter_names: Iterable[str]) -> list[str]:
    names = list(parameter_names)
    ordered = [name for name in STANDARD_PARAMETER_ORDER if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _selector_matches(node: dict[str, Any], selector: WorkflowParamSelector) -> bool:
    if selector.class_type and as_str(node.get("class_type")).lower() != selector.class_type.lower():
        return False
    if selector.title and get_node_title(node).lower() != selector.title.lower():
        return False
    if selector.input_key:
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or selector.input_key not in inputs:
            return False
    return True


def _resolve_selector_target(
    prompt: dict[str, Any], selector: WorkflowParamSelector
) -> tuple[str, str]:
    matches: list[tuple[str, str]] = []
    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            continue
        if not _selector_matches(node, selector):
            continue
        input_key = selector.input_key
        if not input_key:
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            candidates = [key for key in inputs.keys() if isinstance(key, str)]
            if len(candidates) != 1:
                continue
            input_key = candidates[0]
        matches.append((node_id, input_key))

    if not matches:
        raise KeyError("没有节点匹配参数 selector")
    if len(matches) > 1:
        lines = [f"{node_id}.{input_key}" for node_id, input_key in matches[:12]]
        raise KeyError("参数 selector 匹配到多个节点: " + ", ".join(lines))
    return matches[0]


def _resolve_target(prompt: dict[str, Any], mapping: WorkflowParamTarget) -> tuple[str, str]:
    if mapping.ref:
        node_id, input_key = parse_node_input_ref(mapping.ref, default_input="value")
        node = prompt.get(node_id)
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict) and input_key in inputs:
            return node_id, input_key
        if mapping.selector is None:
            return node_id, input_key
    if mapping.selector is None:
        raise KeyError("缺少参数目标 selector")
    return _resolve_selector_target(prompt, mapping.selector)


def _round_number(value: float, mode: str) -> int:
    if mode == "ceil":
        return int(math.ceil(value))
    if mode == "floor":
        return int(math.floor(value))
    if mode == "int":
        return int(value)
    return int(round(value))


def _mapped_value(
    *,
    parameter_name: str,
    parameter_value: Any,
    mapping: WorkflowParamTarget,
    resolved_params: Mapping[str, Any],
) -> Any:
    if mapping.transform == "seconds_to_frames":
        fps_value = resolved_params.get(mapping.fps_param or "fps")
        if fps_value is None:
            raise ValueError(
                f"参数 {parameter_name!r} 使用 seconds_to_frames 时必须同时提供 {mapping.fps_param!r}"
            )
        return _round_number(float(parameter_value) * float(fps_value), mapping.round_mode)
    if mapping.part:
        width, height = _parse_size(parameter_value)
        return width if mapping.part == "width" else height
    return parameter_value


def normalize_standard_params(
    *,
    spec: WorkflowParameterSpec | None,
    request_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = dict(request_params or {})
    if spec is None or not spec.parameters:
        return params
    for name, raw_value in list(params.items()):
        definition = spec.parameters.get(name)
        if definition is None:
            continue
        try:
            params[name] = normalize_parameter_value(definition, raw_value)
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError(f"参数 '{name}' 无效: {exc}") from exc
    return params


def resolve_standard_overrides(
    *,
    workflow_obj: Any,
    spec: WorkflowParameterSpec | None,
    request_params: Mapping[str, Any] | None,
) -> list[tuple[str, str, Any]]:
    if spec is None or not spec.parameters:
        return []

    prompt, _extra = extract_prompt_and_extra(workflow_obj)
    resolved_params: dict[str, Any] = {}
    for name, definition in spec.parameters.items():
        if definition.default is not None:
            resolved_params[name] = normalize_parameter_value(definition, definition.default)
    resolved_params.update(normalize_standard_params(spec=spec, request_params=request_params))

    overrides: list[tuple[str, str, Any]] = []
    for name in _ordered_parameter_names(spec.parameters.keys()):
        if name not in resolved_params:
            continue
        definition = spec.parameters[name]
        value = resolved_params[name]
        for mapping in definition.maps:
            node_id, input_key = _resolve_target(prompt, mapping)
            mapped_value = _mapped_value(
                parameter_name=name,
                parameter_value=value,
                mapping=mapping,
                resolved_params=resolved_params,
            )
            overrides.append((node_id, input_key, mapped_value))
    return overrides


def public_parameter_spec(spec: WorkflowParameterSpec | None) -> dict[str, Any]:
    if spec is None:
        return {
            "version": 1,
            "kind": None,
            "path": None,
            "input_targets": {
                "prompt_node": None,
                "negative_prompt_node": None,
                "negative_prompt_node_with_image": None,
                "image_node": None,
                "audio_node": None,
            },
            "parameters": [],
        }

    items = []
    for name in _ordered_parameter_names(spec.parameters.keys()):
        definition = spec.parameters[name]
        items.append(
            {
                "name": name,
                "type": definition.type,
                "default": definition.default,
                "description": definition.description
                or PARAMETER_DESCRIPTIONS.get(name, ""),
                "label": PARAMETER_LABELS.get(name, name),
                "required": definition.required,
                "minimum": definition.minimum,
                "maximum": definition.maximum,
                "maps": [
                    {
                        "ref": item.ref or None,
                        "selector": (
                            {
                                "class_type": item.selector.class_type or None,
                                "title": item.selector.title or None,
                                "input_key": item.selector.input_key or None,
                            }
                            if item.selector
                            else None
                        ),
                        "part": item.part or None,
                        "transform": item.transform,
                        "fps_param": (
                            item.fps_param if item.transform == "seconds_to_frames" else None
                        ),
                    }
                    for item in definition.maps
                ],
            }
        )

    return {
        "version": spec.version,
        "kind": spec.kind,
        "path": str(spec.path),
        "input_targets": {
            "prompt_node": spec.prompt_node or None,
            "negative_prompt_node": spec.negative_prompt_node or None,
            "negative_prompt_node_with_image": spec.negative_prompt_node_with_image or None,
            "image_node": spec.image_node or None,
            "audio_node": spec.audio_node or None,
        },
        "parameters": items,
    }
