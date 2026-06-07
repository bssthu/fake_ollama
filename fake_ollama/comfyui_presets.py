"""Declarative ComfyUI workflow presets and parameter bindings.

A ComfyUI image model is described by one or two :class:`WorkflowSpec`
objects (text-to-image and, optionally, image-to-image). A spec is just an
API-format workflow JSON plus a *binding table* that maps logical request
parameters (``prompt``, ``seed``, ``steps`` …) onto concrete
``(node_id, input_name)`` slots in that workflow, and a ``static_inputs``
table for fixed values (model file names, sampler knobs that are not
request-driven, …).

This decouples :mod:`fake_ollama.comfyui_client` from any specific node
graph: the client only knows how to drop values into bound slots, so a new
model is added by shipping a workflow JSON + a preset entry (or by supplying
``bindings`` / ``static_inputs`` straight from config), never by editing the
client. The legacy Z-Image-Turbo layout is reproduced by
:func:`z_image_workflows`, which synthesises a spec from the per-field node
ids that older configs still set.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"

# Logical, request-driven parameters a binding table may reference. ``image``
# carries the uploaded reference filename (image-to-image only); ``image_1`` /
# ``image_2`` ... are populated when a client sends multiple reference images.
# ``size_ratio`` is the nearest aspect-ratio bucket for models that pick
# resolution from a combo (e.g. SenseNova) instead of explicit width/height.
DYNAMIC_PARAMS: Tuple[str, ...] = (
    "prompt",
    "seed",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "denoise",
    "width",
    "height",
    "batch_size",
    "image",
    "images",
    "image_1",
    "image_2",
    "image_3",
    "image_4",
    "image_count",
    "size_ratio",
    "num_frames",
    "frame_rate",
    "prefetch_count",
)

Placement = Tuple[str, str]  # (node_id, input_name)


@dataclass(frozen=True)
class WorkflowSpec:
    """One API-format workflow plus its parameter bindings."""

    path: Path
    bindings: Dict[str, List[Placement]]
    static_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Aspect-ratio buckets for ``size_ratio`` (empty when the model takes
    # explicit width/height instead of a ratio combo).
    size_ratio_options: Tuple[str, ...] = ()

    def binds(self, param: str) -> bool:
        return bool(self.bindings.get(param))


@dataclass(frozen=True)
class Preset:
    name: str
    t2i: WorkflowSpec
    i2i: Optional[WorkflowSpec] = None


# ---------------------------------------------------------------------------
# Aspect-ratio bucketing (for ratio-combo models like SenseNova)
# ---------------------------------------------------------------------------

SENSENOVA_RATIOS: Tuple[str, ...] = (
    "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:2", "2:1", "1:3", "3:1",
)


def _ratio_value(ratio: str) -> float:
    a, _, b = ratio.partition(":")
    return float(a) / float(b)


def nearest_ratio(width: int, height: int, options: Tuple[str, ...]) -> str:
    """Pick the aspect-ratio bucket closest to ``width:height``.

    Compares in log-space so e.g. 2:1 and 1:2 are equidistant from 1:1.
    """
    import math

    if not options:
        return ""
    if height <= 0 or width <= 0:
        return options[0]
    target = math.log(width / height)
    return min(options, key=lambda r: abs(math.log(_ratio_value(r)) - target))


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

# Qwen-Image-Edit (e.g. the "Qwen-Rapid-AIO" all-in-one checkpoints) loaded via
# CheckpointLoaderSimple. text-to-image uses a plain CLIPTextEncode; the edit
# graph swaps in TextEncodeQwenImageEditPlus fed by a LoadImage node so the
# reference image is VAE-encoded into the conditioning (denoise stays 1.0).
_QWEN_T2I = WorkflowSpec(
    path=WORKFLOW_DIR / "qwen_image_edit_aio_t2i.json",
    bindings={
        "prompt": [("27", "text")],
        "seed": [("3", "seed")],
        "steps": [("3", "steps")],
        "cfg": [("3", "cfg")],
        "sampler_name": [("3", "sampler_name")],
        "scheduler": [("3", "scheduler")],
        "denoise": [("3", "denoise")],
        "width": [("13", "width")],
        "height": [("13", "height")],
        "batch_size": [("13", "batch_size")],
    },
)
_QWEN_I2I = WorkflowSpec(
    path=WORKFLOW_DIR / "qwen_image_edit_aio_i2i.json",
    bindings={
        "prompt": [("27", "prompt")],
        "image": [("12", "image")],
        "seed": [("3", "seed")],
        "steps": [("3", "steps")],
        "cfg": [("3", "cfg")],
        "sampler_name": [("3", "sampler_name")],
        "scheduler": [("3", "scheduler")],
        "denoise": [("3", "denoise")],
        "width": [("13", "width")],
        "height": [("13", "height")],
        "batch_size": [("13", "batch_size")],
    },
)

# SenseNova-U1: one fused model node + one fused sampler node (custom node pack
# ComfyUI_SenseNova_U1). Resolution comes from an aspect-ratio combo, not
# width/height, and the sampler batches internally. img_mode stays "edit": with
# no image it falls through to text-to-image, with an image it edits.
_SENSENOVA_T2I = WorkflowSpec(
    path=WORKFLOW_DIR / "sensenova_u1_t2i.json",
    bindings={
        "prompt": [("10", "prompt")],
        "seed": [("10", "seed")],
        "steps": [("10", "steps")],
        "cfg": [("10", "cfg")],
        "size_ratio": [("10", "target_pixels")],
        "batch_size": [("10", "batch_size")],
    },
    size_ratio_options=SENSENOVA_RATIOS,
)
_SENSENOVA_I2I = WorkflowSpec(
    path=WORKFLOW_DIR / "sensenova_u1_i2i.json",
    bindings={
        "prompt": [("10", "prompt")],
        "image": [("12", "image")],
        "seed": [("10", "seed")],
        "steps": [("10", "steps")],
        "cfg": [("10", "cfg")],
        "size_ratio": [("10", "target_pixels")],
        "batch_size": [("10", "batch_size")],
    },
    size_ratio_options=SENSENOVA_RATIOS,
)

PRESETS: Dict[str, Preset] = {
    "qwen_image_edit_aio": Preset(
        name="qwen_image_edit_aio", t2i=_QWEN_T2I, i2i=_QWEN_I2I
    ),
    "sensenova_u1": Preset(
        name="sensenova_u1", t2i=_SENSENOVA_T2I, i2i=_SENSENOVA_I2I
    ),
}

# Presets resolved entirely from config (z_image) rather than the table above.
_FIELD_PRESETS = frozenset({"z_image_turbo"})
_CUSTOM_PRESETS = frozenset({"custom", "comfyui_api"})


def preset_names() -> List[str]:
    return ["z_image_turbo", *PRESETS.keys(), *_CUSTOM_PRESETS]


def is_known_preset(name: str) -> bool:
    return name in _FIELD_PRESETS or name in PRESETS or name in _CUSTOM_PRESETS


# ---------------------------------------------------------------------------
# Z-Image-Turbo: spec synthesised from the per-field node ids/model files that
# the original ComfyUITarget schema exposes (keeps old configs working as-is).
# ---------------------------------------------------------------------------


def z_image_workflows(fields: Dict[str, Any]) -> Dict[str, Optional[WorkflowSpec]]:
    """Build {t2i, i2i} specs from the legacy per-field Z-Image config."""

    def nid(key: str, default: str) -> str:
        return str(fields.get(key) or default)

    unet = nid("unet_node_id", "28")
    clip = nid("clip_node_id", "30")
    vae = nid("vae_node_id", "29")
    prompt_n = nid("prompt_node_id", "27")
    latent = nid("latent_node_id", "13")
    sampling = nid("sampling_node_id", "11")
    ksampler = nid("ksampler_node_id", "3")
    save = nid("save_image_node_id", "9")
    load_image = nid("load_image_node_id", "12")
    image_scale = nid("image_scale_node_id", "14")

    def loaders_static() -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if fields.get("diffusion_model") is not None:
            out.setdefault(unet, {})["unet_name"] = fields["diffusion_model"]
        if fields.get("diffusion_weight_dtype") is not None:
            out.setdefault(unet, {})["weight_dtype"] = fields["diffusion_weight_dtype"]
        if fields.get("text_encoder_model") is not None:
            out.setdefault(clip, {})["clip_name"] = fields["text_encoder_model"]
        if fields.get("text_encoder_type") is not None:
            out.setdefault(clip, {})["type"] = fields["text_encoder_type"]
        if fields.get("text_encoder_device") is not None:
            out.setdefault(clip, {})["device"] = fields["text_encoder_device"]
        if fields.get("vae_model") is not None:
            out.setdefault(vae, {})["vae_name"] = fields["vae_model"]
        if fields.get("default_shift") is not None:
            out.setdefault(sampling, {})["shift"] = fields["default_shift"]
        return out

    sampler_bindings = {
        "seed": [(ksampler, "seed")],
        "steps": [(ksampler, "steps")],
        "cfg": [(ksampler, "cfg")],
        "sampler_name": [(ksampler, "sampler_name")],
        "scheduler": [(ksampler, "scheduler")],
        "denoise": [(ksampler, "denoise")],
        "prompt": [(prompt_n, "text")],
    }

    t2i = WorkflowSpec(
        path=Path(fields["text_to_image_workflow_path"])
        if fields.get("text_to_image_workflow_path")
        else WORKFLOW_DIR / "z_image_turbo_t2i.json",
        bindings={
            **sampler_bindings,
            "width": [(latent, "width")],
            "height": [(latent, "height")],
            "batch_size": [(latent, "batch_size")],
        },
        static_inputs=loaders_static(),
    )

    i2i_static = loaders_static()
    if fields.get("image_upscale_method") is not None:
        i2i_static.setdefault(image_scale, {})["upscale_method"] = fields[
            "image_upscale_method"
        ]
    if fields.get("image_crop") is not None:
        i2i_static.setdefault(image_scale, {})["crop"] = fields["image_crop"]

    i2i = WorkflowSpec(
        path=Path(fields["image_to_image_workflow_path"])
        if fields.get("image_to_image_workflow_path")
        else WORKFLOW_DIR / "z_image_turbo_i2i.json",
        bindings={
            **sampler_bindings,
            "image": [(load_image, "image")],
            "width": [(image_scale, "width")],
            "height": [(image_scale, "height")],
        },
        static_inputs=i2i_static,
    )
    return {"t2i": t2i, "i2i": i2i}


# ---------------------------------------------------------------------------
# Resolution: turn a target's preset + overrides into {t2i, i2i} specs
# ---------------------------------------------------------------------------


def _merge_static(
    base: Dict[str, Dict[str, Any]], extra: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    out = {node: dict(inputs) for node, inputs in base.items()}
    for node, inputs in (extra or {}).items():
        if not isinstance(inputs, dict):
            continue
        out.setdefault(str(node), {}).update(inputs)
    return out


def _merge_bindings(
    base: Dict[str, List[Placement]], extra: Optional[Dict[str, Any]]
) -> Dict[str, List[Placement]]:
    out = {param: list(places) for param, places in base.items()}
    for param, places in (extra or {}).items():
        norm: List[Placement] = []
        for place in places or []:
            if isinstance(place, (list, tuple)) and len(place) == 2:
                norm.append((str(place[0]), str(place[1])))
            elif isinstance(place, dict) and "node" in place and "input" in place:
                norm.append((str(place["node"]), str(place["input"])))
        out[str(param)] = norm
    return out


def resolve_workflows(
    fields: Dict[str, Any],
) -> Dict[str, Optional[WorkflowSpec]]:
    """Resolve a ComfyUITarget's fields into {t2i, i2i} WorkflowSpecs.

    ``fields`` is the flat dict produced by ``ComfyUITarget.workflow_config()``
    (preset name, per-field node ids/model files for z_image, plus optional
    ``bindings`` / ``static_inputs`` / workflow-path overrides).
    """
    preset_name = str(fields.get("preset") or "z_image_turbo")
    save_node = str(fields.get("save_image_node_id") or "9")
    output_prefix = fields.get("output_prefix")
    override_bindings = fields.get("bindings") or {}
    override_static = fields.get("static_inputs") or {}
    t2i_path = fields.get("text_to_image_workflow_path")
    i2i_path = fields.get("image_to_image_workflow_path")
    video_path = fields.get("video_workflow_path")

    if preset_name in _FIELD_PRESETS:
        base = z_image_workflows(fields)
    elif preset_name in PRESETS:
        preset = PRESETS[preset_name]
        base = {
            "t2i": preset.t2i,
            "i2i": preset.i2i,
        }
    elif preset_name in _CUSTOM_PRESETS:
        base = {
            "t2i": (
                WorkflowSpec(path=Path(t2i_path), bindings={}, static_inputs={})
                if t2i_path
                else None
            ),
            "i2i": (
                WorkflowSpec(path=Path(i2i_path), bindings={}, static_inputs={})
                if i2i_path
                else None
            ),
            "video": (
                WorkflowSpec(path=Path(video_path), bindings={}, static_inputs={})
                if video_path
                else None
            ),
        }
    else:
        raise ValueError(
            f"unknown comfyui preset {preset_name!r}; known presets: {preset_names()}"
        )

    resolved: Dict[str, Optional[WorkflowSpec]] = {}
    for mode in ("t2i", "i2i", "video"):
        spec = base.get(mode)
        if spec is None:
            resolved[mode] = None
            continue
        path = spec.path
        if mode == "t2i" and t2i_path:
            path = Path(t2i_path)
        if mode == "i2i" and i2i_path:
            path = Path(i2i_path)
        if mode == "video" and video_path:
            path = Path(video_path)
        static_inputs = _merge_static(spec.static_inputs, override_static.get(mode))
        # Output filename prefix is a per-target, non-request value; route it
        # to the SaveImage node uniformly across presets.
        if output_prefix is not None:
            static_inputs = _merge_static(
                static_inputs, {save_node: {"filename_prefix": output_prefix}}
            )
        bindings = _merge_bindings(spec.bindings, override_bindings.get(mode))
        resolved[mode] = WorkflowSpec(
            path=path,
            bindings=bindings,
            static_inputs=static_inputs,
            size_ratio_options=spec.size_ratio_options,
        )
    return resolved
