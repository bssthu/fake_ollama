"""Request parsing, media parameters and per-model limits."""

from __future__ import annotations

import json
import math
import base64
import secrets
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, Request
from .config import H3ContextIRProfile, Settings, estimate_tokens_from_anthropic_payload
from .media_operations import describe_comfyui_operation

async def _image_payload_from_request(
    request: Request,
) -> tuple[Dict[str, Any], List[Tuple[bytes, str]]]:
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        payload: Dict[str, Any] = {}
        image_inputs: List[Tuple[bytes, str]] = []
        for key, value in form.multi_items():
            if hasattr(value, "read") and hasattr(value, "filename"):
                upload = value
                # OpenAI's images/edits multipart convention sends the input
                # image as "image"; with multiple reference images (and what
                # the AI SDK's OpenAICompatibleImageModel emits) it is "image[]"
                # / "image[0]". Preserve all of them in request order so
                # multi-reference ComfyUI graphs can bind image_1/image_2/...
                if key == "image" or key.startswith("image["):
                    filename = getattr(upload, "filename", None) or "input.png"
                    image_inputs.append((await upload.read(), filename))
                continue
            payload[key] = value
        return payload, image_inputs

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    filenames = payload.get("filenames")
    fallback_filename = payload.get("filename")

    def filename_for(idx: int) -> str:
        if isinstance(filenames, list) and idx < len(filenames):
            return str(filenames[idx] or f"input-{idx + 1}.png")
        if isinstance(fallback_filename, list) and idx < len(fallback_filename):
            return str(fallback_filename[idx] or f"input-{idx + 1}.png")
        if isinstance(fallback_filename, str) and fallback_filename:
            return fallback_filename if idx == 0 else f"input-{idx + 1}.png"
        return "input.png" if idx == 0 else f"input-{idx + 1}.png"

    raw_values: List[Any] = []
    raw_image = payload.get("image")
    if isinstance(raw_image, list):
        raw_values.extend(raw_image)
    elif isinstance(raw_image, str) and raw_image.strip():
        raw_values.append(raw_image)
    raw_images = payload.get("images")
    if isinstance(raw_images, list):
        raw_values.extend(raw_images)
    elif isinstance(raw_images, str) and raw_images.strip():
        raw_values.append(raw_images)

    image_inputs: List[Tuple[bytes, str]] = []
    for idx, raw in enumerate(raw_values):
        if isinstance(raw, str) and raw.strip():
            image_inputs.append((_decode_base64_image(raw), filename_for(idx)))
    return payload, image_inputs


def _decode_base64_image(raw: str) -> bytes:
    data = raw.strip()
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid base64 image") from exc


def _coerce_int_payload(
    payload: Dict[str, Any],
    keys: tuple[str, ...],
    default: int,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    value: Any = None
    for key in keys:
        if payload.get(key) not in (None, ""):
            value = payload.get(key)
            break
    if value is None:
        value = default
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{keys[0]} must be an integer") from exc
    if out < minimum or (maximum is not None and out > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise HTTPException(
            status_code=400,
            detail=f"{keys[0]} must be >= {minimum}{upper}",
        )
    return out


def _coerce_float_payload(
    payload: Dict[str, Any],
    keys: tuple[str, ...],
    default: float,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:
    value: Any = None
    for key in keys:
        if payload.get(key) not in (None, ""):
            value = payload.get(key)
            break
    if value is None:
        value = default
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{keys[0]} must be a number") from exc
    if not math.isfinite(out):
        raise HTTPException(status_code=400, detail=f"{keys[0]} must be finite")
    if out < minimum or (maximum is not None and out > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise HTTPException(
            status_code=400,
            detail=f"{keys[0]} must be >= {minimum}{upper}",
        )
    return out


def _coerce_bool_payload(
    payload: Dict[str, Any], keys: tuple[str, ...], default: bool
) -> bool:
    value: Any = None
    for key in keys:
        if payload.get(key) not in (None, ""):
            value = payload.get(key)
            break
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on"}:
            return True
        if normalised in {"0", "false", "no", "off"}:
            return False
    raise HTTPException(status_code=400, detail=f"{keys[0]} must be a boolean")


def _parse_image_size(payload: Dict[str, Any], target: Any) -> tuple[int, int]:
    width = payload.get("width")
    height = payload.get("height")
    size = payload.get("size")
    if (width in (None, "")) and (height in (None, "")) and isinstance(size, str):
        if size.lower() not in ("", "auto"):
            if "x" not in size.lower():
                raise HTTPException(
                    status_code=400,
                    detail="size must be 'WIDTHxHEIGHT', for example '1024x1024'",
                )
            left, _, right = size.lower().partition("x")
            width, height = left, right
    if width in (None, ""):
        width = target.default_width
    if height in (None, ""):
        height = target.default_height
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="width and height must be integers") from exc
    for axis, value in (("width", w), ("height", h)):
        minimum = int(getattr(target, f"min_{axis}", 8) or 8)
        maximum = getattr(target, f"max_{axis}", None)
        modulo = int(getattr(target, f"{axis}_modulo", 8) or 8)
        if value < minimum:
            raise HTTPException(
                status_code=400,
                detail=f"{axis} must be >= {minimum} for this ComfyUI workflow",
            )
        if maximum is not None and value > int(maximum):
            raise HTTPException(
                status_code=400,
                detail=f"{axis} must be <= {maximum} for this ComfyUI workflow",
            )
        if value % modulo:
            raise HTTPException(
                status_code=400,
                detail=f"{axis} must be divisible by {modulo} for this ComfyUI workflow",
            )
    return w, h


def _resolve_image_seed(
    payload: Dict[str, Any], target: Any, app: Any, *, count: int
) -> int:
    """Pick the KSampler seed for one request.

    An explicit ``seed`` in the request body always wins. Otherwise the
    per-target ``seed_mode`` decides: ``fixed`` reuses ``target.seed``,
    ``increment`` advances a per-target counter (by ``count``, the number of
    images this request consumes) starting from ``target.seed``, and
    ``random`` (default) draws a fresh random seed.
    """
    seed_raw = payload.get("seed")
    if seed_raw not in (None, "", "random"):
        return _coerce_int_payload(payload, ("seed",), 0, minimum=0)

    mode = (getattr(target, "seed_mode", "random") or "random").lower()
    base = int(getattr(target, "seed", 0) or 0)
    if mode == "fixed":
        return base
    if mode == "increment":
        counters = getattr(app.state, "comfyui_seed_counters", None)
        if counters is None:
            counters = {}
            app.state.comfyui_seed_counters = counters
        cur = counters.get(target.name, base)
        counters[target.name] = cur + max(1, int(count))
        return cur
    # Keep random seeds within signed 32-bit range (0 .. 2**31-1). ComfyUI's
    # KSampler accepts up to 2**64-1, but some custom sampler nodes cap the seed
    # at INT32_MAX (e.g. SenseNova_SM_Sampler rejects anything larger), and any
    # float64 consumer loses precision above 2**53-1 (JS Number.MAX_SAFE_INTEGER,
    # e.g. dropping the output PNG back into ComfyUI's web UI to reproduce it).
    # 2**31-1 satisfies every backend.
    return secrets.randbits(31)


def _image_request_params(
    payload: Dict[str, Any],
    target: Any,
    *,
    edit: bool,
    app: Any,
) -> Dict[str, Any]:
    width, height = _parse_image_size(payload, target)
    n = _coerce_int_payload(payload, ("n",), 1)
    if n > target.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"n={n} exceeds comfyui target max_batch_size={target.max_batch_size}",
        )
    seed = _resolve_image_seed(payload, target, app, count=n)
    return {
        "width": width,
        "height": height,
        "n": n,
        "seed": seed,
        "negative_prompt": str(payload.get("negative_prompt") or ""),
        "steps": _coerce_int_payload(
            payload, ("steps", "num_inference_steps"), target.default_steps
        ),
        "cfg": _coerce_float_payload(
            payload, ("cfg", "guidance_scale"), target.default_cfg
        ),
        "sampler_name": str(
            payload.get("sampler_name") or target.default_sampler_name
        ),
        "scheduler": str(payload.get("scheduler") or target.default_scheduler),
        "denoise": _coerce_float_payload(
            payload,
            ("denoise",),
            target.default_edit_denoise if edit else target.default_denoise,
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _enforce_reference_image_limit(
    target: Any,
    operation_id: str,
    image_inputs: List[Tuple[bytes, str]],
) -> None:
    """Reject public requests that exceed the workflow's advertised capacity."""

    operation = describe_comfyui_operation(target, operation_id)
    maximum = operation.get("limits", {}).get("max_reference_images")
    if maximum is not None and len(image_inputs) > int(maximum):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{operation_id} accepts at most {maximum} reference image(s) "
                f"for comfyui target {getattr(target, 'name', '')!r}; "
                f"received {len(image_inputs)}"
            ),
        )


def _video_request_params(
    payload: Dict[str, Any],
    target: Any,
    *,
    app: Any,
) -> Dict[str, Any]:
    params = _image_request_params(payload, target, edit=False, app=app)
    min_num_frames = int(getattr(target, "min_num_frames", 1) or 1)
    max_num_frames = int(getattr(target, "max_num_frames", 241) or 241)
    params["num_frames"] = _coerce_int_payload(
        payload,
        ("num_frames", "frames"),
        getattr(target, "default_num_frames", 121),
        minimum=min_num_frames,
        maximum=max_num_frames,
    )
    num_frames_modulo = int(getattr(target, "num_frames_modulo", 1) or 1)
    num_frames_offset = int(getattr(target, "num_frames_offset", 0) or 0)
    if (params["num_frames"] - num_frames_offset) % num_frames_modulo != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"num_frames must satisfy "
                f"(num_frames - {num_frames_offset}) % {num_frames_modulo} == 0 "
                f"for comfyui target {getattr(target, 'name', '')!r}"
            ),
        )
    params["frame_rate"] = _coerce_float_payload(
        payload,
        ("frame_rate", "fps"),
        getattr(target, "default_frame_rate", 24.0),
        minimum=float(getattr(target, "min_frame_rate", 1.0) or 1.0),
        maximum=float(getattr(target, "max_frame_rate", 120.0) or 120.0),
    )
    params["prefetch_count"] = _coerce_int_payload(
        payload,
        ("prefetch_count",),
        getattr(target, "default_prefetch_count", 1),
        minimum=0,
        maximum=int(getattr(target, "max_prefetch_count", 48) or 48),
    )
    params["enable_tile"] = _coerce_bool_payload(
        payload,
        ("enable_tile",),
        getattr(target, "default_enable_tile", False),
    )
    params["enable_streaming"] = _coerce_bool_payload(
        payload,
        ("enable_streaming",),
        getattr(target, "default_enable_streaming", False),
    )
    return params


def _context_ir_duration(
    payload: Dict[str, Any],
    profile: H3ContextIRProfile,
    *,
    video_params: Optional[Dict[str, Any]] = None,
) -> float:
    raw = payload.get("duration_seconds", payload.get("duration"))
    if raw is not None and str(raw).strip():
        try:
            duration = float(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="duration_seconds must be numeric"
            ) from exc
    elif video_params:
        frame_count = float(video_params.get("num_frames") or 0)
        frame_rate = float(video_params.get("frame_rate") or 0)
        duration = (
            frame_count / frame_rate
            if frame_count > 0 and frame_rate > 0
            else profile.default_duration_seconds
        )
    else:
        duration = profile.default_duration_seconds
    duration = round(duration, 2)
    if not 4 <= duration <= 15:
        raise HTTPException(
            status_code=400,
            detail="H3 Context-IR duration_seconds must be between 4 and 15",
        )
    return duration


def _enforce_limits(
    settings: Settings,
    ollama_model: str,
    upstream_payload: Dict[str, Any],
) -> None:
    """Apply per-model max_output_tokens default + context guardrail.

    Mutates ``upstream_payload`` in place. Raises HTTPException(400) when the
    estimated input + ``max_tokens`` exceeds the model's context window and
    ``FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT`` is enabled.
    """
    profile = settings.profile_for(ollama_model)
    # Treat profile.max_output_tokens as both a floor and a cap. Many BYOK
    # clients (notably VS Code Copilot Chat) send a small ``max_tokens``
    # default that easily produces ``finish_reason=length`` on long file
    # edits; downstream the client then refuses the whole reply with
    # "Response too long". Lifting to the per-model output ceiling lets the
    # upstream actually finish.
    if profile.max_output_tokens:
        cur = int(upstream_payload.get("max_tokens") or 0)
        if cur <= 0 or cur < profile.max_output_tokens:
            upstream_payload["max_tokens"] = profile.max_output_tokens
        else:
            upstream_payload["max_tokens"] = min(cur, profile.max_output_tokens)

    if not settings.enforce_context_limit:
        return
    estimated_input = estimate_tokens_from_anthropic_payload(upstream_payload)
    requested_output = int(upstream_payload.get("max_tokens") or 0)
    total = estimated_input + requested_output
    if total > profile.context_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"request exceeds context window for model {ollama_model!r}: "
                f"estimated_input_tokens={estimated_input} + max_tokens={requested_output} "
                f"= {total} > context_length={profile.context_length}. "
                "Reduce the prompt or lower max_tokens / num_predict. "
                "Set FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT=false to disable this guardrail."
            ),
        )


def _apply_thinking_config(
    settings: Settings,
    ollama_model: str,
    upstream_payload: Dict[str, Any],
) -> None:
    """Inject the per-model `thinking` directive into the upstream payload.

    Explicit profile modes are authoritative. Client-provided ``thinking``
    is only honoured when the profile mode is ``auto``. For ``auto`` we also
    force ``disabled`` if the profile sets ``show_thinking=False`` and the
    client did not make an explicit request.
    """
    profile = settings.profile_for(ollama_model)
    mode = profile.thinking_mode
    if mode == "enabled":
        upstream_payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": profile.thinking_budget_tokens,
        }
    elif mode == "disabled":
        upstream_payload["thinking"] = {"type": "disabled"}
    elif "thinking" in upstream_payload:
        return
    elif not profile.show_thinking:
        upstream_payload["thinking"] = {"type": "disabled"}


def _apply_ollama_thinking_config(
    settings: Settings,
    display_model: str,
    anthropic_payload: Dict[str, Any],
    ollama_payload: Dict[str, Any],
) -> None:
    """Map Anthropic/profile thinking preferences onto Ollama's API.

    Newer Ollama models may stream reasoning on ``message.thinking`` before
    visible ``message.content``. If a model profile disables thinking, pass
    ``think: false`` so clients do not see an apparently empty response when
    their output budget is consumed by hidden reasoning.
    """
    profile = settings.profile_for(display_model)
    if profile.thinking_mode == "disabled":
        ollama_payload["think"] = False
        return
    if profile.thinking_mode == "enabled":
        ollama_payload["think"] = True
        return

    if "think" in ollama_payload:
        return

    requested = anthropic_payload.get("thinking")
    if isinstance(requested, dict):
        req_type = str(requested.get("type") or "").lower()
        if req_type == "disabled":
            ollama_payload["think"] = False
            return
        if req_type == "enabled":
            ollama_payload["think"] = True
            return

    if not profile.show_thinking:
        ollama_payload["think"] = False


def _apply_reverse_output_limits(
    settings: Settings,
    display_model: str,
    ollama_payload: Dict[str, Any],
) -> None:
    """Lift ``options.num_predict`` to the per-profile ``max_output_tokens``.

    Some upstream clients (notably VS Code Copilot Chat's BYOK provider) send
    a small ``max_tokens`` that easily truncates long file edits. When the
    upstream returns ``done_reason="length"`` Copilot rejects the whole
    response with "Response too long" (see Copilot extension.js: it bails on
    ``finishReason === "length"``). To let users opt in to a higher floor per
    model, when ``profile.max_output_tokens`` is configured and is greater
    than what the converter wrote into ``num_predict``, we raise it.

    This mirrors the floor half of ``_enforce_limits`` but is intentionally
    floor-only on the reverse path: we never lower a client-requested cap.
    """
    profile = settings.profile_for(display_model)
    if not profile.max_output_tokens:
        return
    options = ollama_payload.setdefault("options", {})
    if not isinstance(options, dict):
        return
    cur = int(options.get("num_predict") or 0)
    if cur < int(profile.max_output_tokens):
        options["num_predict"] = int(profile.max_output_tokens)


def _apply_llama_cpp_thinking_config(
    settings: Settings,
    display_model: str,
    source_payload: Dict[str, Any],
    openai_payload: Dict[str, Any],
) -> None:
    """Map thinking preferences onto llama.cpp's chat-template kwargs.

    Recent llama.cpp server builds support per-request
    ``chat_template_kwargs.enable_thinking`` for Qwen-style templates. This
    mirrors the Ollama ``think`` mapping so low ``max_tokens`` requests do not
    spend their entire budget on hidden reasoning when a profile disables it.
    """
    existing = openai_payload.get("chat_template_kwargs")
    profile = settings.profile_for(display_model)

    enabled: bool | None = None
    if profile.thinking_mode == "disabled":
        enabled = False
    elif profile.thinking_mode == "enabled":
        enabled = True
    else:
        if isinstance(existing, dict) and "enable_thinking" in existing:
            return
        requested = source_payload.get("thinking")
        if isinstance(requested, dict):
            req_type = str(requested.get("type") or "").lower()
            if req_type == "disabled":
                enabled = False
            elif req_type == "enabled":
                enabled = True
        if enabled is None and not profile.show_thinking:
            enabled = False

    if enabled is None:
        return
    kwargs = dict(existing) if isinstance(existing, dict) else {}
    kwargs["enable_thinking"] = enabled
    openai_payload["chat_template_kwargs"] = kwargs


_COUNT_TOKENS_BODY_KEYS = {
    "cache_control",
    "messages",
    "model",
    "output_config",
    "system",
    "thinking",
    "tool_choice",
    "tools",
}


def _count_tokens_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Build an Anthropic count_tokens body, dropping generation-only fields."""
    out = {k: payload[k] for k in _COUNT_TOKENS_BODY_KEYS if k in payload}
    out["model"] = model
    return out
