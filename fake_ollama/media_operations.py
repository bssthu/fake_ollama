"""Public media-operation schemas derived from ComfyUI workflow bindings.

The ComfyUI workflow remains the execution source of truth.  This module turns
its logical bindings into a transport-neutral parameter description consumed
by ``/playground/api/models``.  A parameter is advertised only when the selected
workflow actually binds it (apart from API orchestration fields such as ``n``),
which prevents controls that silently do nothing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .comfyui_presets import OperationPreset, WorkflowSpec, nearest_ratio, resolve_workflows


def _workflow_specs(
    target: Any, operation_id: str
) -> Tuple[List[str], List[WorkflowSpec]]:
    workflows = resolve_workflows(target.workflow_config())
    mode_names = {
        "image_generation": ("t2i",),
        "image_edit": ("i2i",),
        "video_generation": ("video", "i2v", "fl2va", "l2va"),
    }.get(operation_id, ())
    configured_modes = [mode for mode in mode_names if workflows.get(mode) is not None]
    return configured_modes, [workflows[mode] for mode in configured_modes]


def _bound_parameters(specs: Iterable[WorkflowSpec]) -> set[str]:
    return {
        parameter
        for spec in specs
        for parameter, placements in spec.bindings.items()
        if placements
    }


def _ratio_size(ratio: str) -> str:
    """Return a harmless divisible-by-eight size carrying ``ratio``.

    Ratio-based workflows ignore the pixel count and map width/height to the
    nearest native bucket.  Keeping the longest edge at 1024 makes the public
    request readable without pretending that it controls the model's pixels.
    """

    left, _, right = ratio.partition(":")
    a, b = max(1, int(left)), max(1, int(right))
    if a >= b:
        width = 1024
        height = max(8, round((1024 * b / a) / 8) * 8)
    else:
        height = 1024
        width = max(8, round((1024 * a / b) / 8) * 8)
    return f"{width}x{height}"


def _parameter(
    name: str,
    label: str,
    kind: str,
    default: Any,
    *,
    advanced: bool = False,
    description: str = "",
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None,
    nullable: bool = False,
    choices: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": name,
        "label": label,
        "type": kind,
        "default": default,
        "advanced": advanced,
    }
    if description:
        result["description"] = description
    if minimum is not None:
        result["min"] = minimum
    if maximum is not None:
        result["max"] = maximum
    if step is not None:
        result["step"] = step
    if nullable:
        result["nullable"] = True
    if choices:
        result["choices"] = choices
    return result


def _public_presets(specs: Iterable[WorkflowSpec]) -> List[Dict[str, Any]]:
    by_id: Dict[str, OperationPreset] = {}
    for spec in specs:
        for preset in spec.operation_presets:
            by_id.setdefault(preset.id, preset)
    return [
        {
            "id": preset.id,
            "label": preset.label,
            "description": preset.description,
            "recommended": preset.recommended,
            "values": dict(preset.values),
        }
        for preset in by_id.values()
    ]


def _image_ref_limit(spec: WorkflowSpec) -> Optional[int]:
    """Mirror the client's structural reference-image capacity."""

    if spec.max_image_refs is not None:
        return max(0, int(spec.max_image_refs))
    if spec.binds("images"):
        return None
    limit = 1 if spec.binds("image") else 0
    for index in range(1, 5):
        if spec.binds(f"image_{index}"):
            limit = max(limit, index)
    return limit


def describe_comfyui_operation(target: Any, operation_id: str) -> Dict[str, Any]:
    """Describe one configured ComfyUI operation for discovery clients."""

    workflow_modes, specs = _workflow_specs(target, operation_id)
    bound = _bound_parameters(specs)
    parameters: List[Dict[str, Any]] = []
    default_size = f"{target.default_width}x{target.default_height}"

    if {"width", "height", "size_ratio"}.intersection(bound):
        ratio_options = next(
            (spec.size_ratio_options for spec in specs if spec.size_ratio_options),
            (),
        )
        if ratio_options:
            default_ratio = nearest_ratio(
                target.default_width, target.default_height, ratio_options
            )
            parameters.append(
                _parameter(
                    "size",
                    "宽高比",
                    "select",
                    _ratio_size(default_ratio),
                    description="模型按原生像素档位输出；这里选择最接近的宽高比。",
                    choices=[
                        {"value": _ratio_size(ratio), "label": ratio}
                        for ratio in ratio_options
                    ],
                )
            )
        else:
            min_width = int(getattr(target, "min_width", 8) or 8)
            min_height = int(getattr(target, "min_height", 8) or 8)
            width_modulo = int(getattr(target, "width_modulo", 8) or 8)
            height_modulo = int(getattr(target, "height_modulo", 8) or 8)
            max_width = getattr(target, "max_width", None)
            max_height = getattr(target, "max_height", None)
            resolution = _parameter(
                "size",
                "尺寸",
                "resolution",
                default_size,
                description=(
                    "格式为 WIDTHxHEIGHT；"
                    f"宽至少 {min_width} 且为 {width_modulo} 的倍数，"
                    f"高至少 {min_height} 且为 {height_modulo} 的倍数。"
                ),
            )
            resolution.update(
                {
                    "min_width": min_width,
                    "min_height": min_height,
                    "width_modulo": width_modulo,
                    "height_modulo": height_modulo,
                }
            )
            if max_width is not None:
                resolution["max_width"] = int(max_width)
            if max_height is not None:
                resolution["max_height"] = int(max_height)
            parameters.append(
                resolution
            )

    max_batch = max(1, int(target.max_batch_size))
    if max_batch > 1:
        parameters.append(
            _parameter("n", "数量", "integer", 1, minimum=1, maximum=max_batch, step=1)
        )

    if "seed" in bound:
        seed_default = (
            int(target.seed)
            if str(target.seed_mode).lower() in {"fixed", "increment"}
            else None
        )
        parameters.append(
            _parameter(
                "seed",
                "Seed",
                "integer",
                seed_default,
                advanced=True,
                minimum=0,
                step=1,
                nullable=True,
                description="留空时按 target 的 seed_mode 生成。",
            )
        )
    if "steps" in bound:
        parameters.append(
            _parameter(
                "steps", "Steps", "integer", target.default_steps,
                advanced=True, minimum=1, step=1,
            )
        )
    if "cfg" in bound:
        parameters.append(
            _parameter(
                "cfg", "CFG", "number", target.default_cfg,
                advanced=True, minimum=0, step=0.1,
            )
        )
    if "sampler_name" in bound:
        parameters.append(
            _parameter(
                "sampler_name", "Sampler", "string", target.default_sampler_name,
                advanced=True,
            )
        )
    if "scheduler" in bound:
        parameters.append(
            _parameter(
                "scheduler", "Scheduler", "string", target.default_scheduler,
                advanced=True,
            )
        )
    if operation_id == "image_edit" and "denoise" in bound:
        parameters.append(
            _parameter(
                "denoise", "Denoise", "number", target.default_edit_denoise,
                advanced=True, minimum=0, maximum=1, step=0.05,
            )
        )
    if "num_frames" in bound:
        frame_parameter = _parameter(
            "num_frames",
            "帧数",
            "integer",
            target.default_num_frames,
            minimum=getattr(target, "min_num_frames", 1),
            maximum=target.max_num_frames,
            step=max(1, int(target.num_frames_modulo)),
            description=(
                f"需满足 (帧数 - {target.num_frames_offset}) % "
                f"{target.num_frames_modulo} = 0。"
            ),
        )
        frame_parameter.update(
            {
                "offset": int(target.num_frames_offset),
                "modulo": int(target.num_frames_modulo),
            }
        )
        parameters.append(frame_parameter)
    if "frame_rate" in bound:
        parameters.append(
            _parameter(
                "fps",
                "FPS",
                "number",
                target.default_frame_rate,
                minimum=getattr(target, "min_frame_rate", 1.0),
                maximum=getattr(target, "max_frame_rate", 120.0),
                step=1,
            )
        )
    if "prefetch_count" in bound:
        parameters.append(
            _parameter(
                "prefetch_count",
                "Prefetch count",
                "integer",
                target.default_prefetch_count,
                advanced=True,
                minimum=0,
                maximum=getattr(target, "max_prefetch_count", 48),
                step=1,
                description="模型层预取/卸载参数；具体含义由 workflow 节点定义。",
            )
        )
    if "enable_tile" in bound:
        parameters.append(
            _parameter(
                "enable_tile",
                "分块解码",
                "boolean",
                getattr(target, "default_enable_tile", False),
                advanced=True,
                description="降低视频 VAE 解码峰值显存，通常会增加耗时。",
            )
        )
    if "enable_streaming" in bound:
        parameters.append(
            _parameter(
                "enable_streaming",
                "模型流式加载",
                "boolean",
                getattr(target, "default_enable_streaming", False),
                advanced=True,
                description="控制模型内部权重加载，不是 HTTP 流式响应。",
            )
        )

    defaults = {parameter["name"]: parameter.get("default") for parameter in parameters}
    defaults.setdefault("n", 1)
    presets = _public_presets(specs)
    recommended = next(
        (preset["id"] for preset in presets if preset.get("recommended")), None
    )
    matching_presets = [
        preset
        for preset in presets
        if preset.get("values")
        and all(defaults.get(name) == value for name, value in preset["values"].items())
    ]
    default_preset = next(
        (
            preset["id"]
            for preset in matching_presets
            if preset.get("recommended")
        ),
        matching_presets[0]["id"] if matching_presets else None,
    )

    image_specs = [
        spec
        for mode, spec in zip(workflow_modes, specs)
        if mode in {"i2i", "i2v", "fl2va", "l2va"}
    ]
    reference_limits = [_image_ref_limit(spec) for spec in image_specs]
    max_references = (
        None
        if any(limit is None for limit in reference_limits)
        else (max(reference_limits) if reference_limits else 0)
    )
    accepts_images = operation_id == "image_edit" or bool(image_specs)
    requires_images = operation_id == "image_edit" or (
        operation_id == "video_generation"
        and any(mode in workflow_modes for mode in ("i2v", "fl2va", "l2va"))
        and "video" not in workflow_modes
    )
    multiple_images = accepts_images and (
        max_references is None or max_references > 1
    )

    limits: Dict[str, Any] = {"max_batch_size": max_batch}
    if max_references:
        limits["max_reference_images"] = max_references
    if operation_id == "video_generation":
        limits.update(
            {
                "min_num_frames": getattr(target, "min_num_frames", 1),
                "max_num_frames": target.max_num_frames,
                "num_frames_offset": target.num_frames_offset,
                "num_frames_modulo": target.num_frames_modulo,
            }
        )

    return {
        "configured": bool(specs),
        "workflow_modes": workflow_modes,
        "accepts_images": accepts_images,
        "requires_images": requires_images,
        "multiple_images": multiple_images,
        "parameters": parameters,
        "defaults": defaults,
        "limits": limits,
        "presets": presets,
        "recommended_preset": recommended,
        "default_preset": default_preset,
    }
