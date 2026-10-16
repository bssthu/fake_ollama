"""Image, video and H3 planning orchestration."""

from __future__ import annotations

import base64
import mimetypes
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from .anthropic_client import AnthropicClient
from .comfyui_client import ComfyUIClient
from .comfyui_presets import resolve_workflows
from .h3_media import video_media_from_request
from .config import H3ContextIRProfile, H3ContextIRProvider, LlamaCppTarget, Settings
from .h3_context_ir import SYSTEM_PROMPT as H3_CONTEXT_IR_SYSTEM_PROMPT, build_planning_request, build_repair_request, fallback_plan, is_structured_base_prompt, parse_and_validate_plan, render_base_prompt, resolve_base_mode
from .openai_client import OpenAIClient
from .playground_support import (
    ExternalPlannerConnection,
    _context_ir_provider_is_exposed,
    _external_planner_connection,
    _external_planner_http_client,
)
from .proxy_errors import (
    _log_upstream_error,
    _read_error_text,
    _upstream_error,
)
from .request_validation import (
    _apply_llama_cpp_thinking_config,
    _context_ir_duration,
    _enforce_reference_image_limit,
    _image_payload_from_request,
    _image_request_params,
    _video_request_params,
)
from .routing import (
    _authorise_interface,
    _backend_client,
    _dispatch,
    _interface_for,
    _model_interface_for_request,
)

import logging

logger = logging.getLogger("fake_ollama")

def _default_image_model(request: Request, settings: Settings) -> str:
    iface = _authorise_interface(request)
    candidates: List[str] = []
    for exposure in iface.exposed_models:
        try:
            backend, _ = settings.resolve_request(
                exposure.public_id, interface_name=iface.name
            )
        except ValueError:
            continue
        if backend.protocol == "comfyui":
            candidates.append(exposure.public_id)
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="no ComfyUI image model is exposed on this interface",
        )
    if len(candidates) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "missing 'model'; multiple ComfyUI image models are exposed: "
                + ", ".join(candidates)
            ),
        )
    return candidates[0]


def _dispatch_image_backend(
    request: Request, settings: Settings, payload: Dict[str, Any]
):
    requested_model = str(payload.get("model") or "").strip()
    if not requested_model:
        requested_model = _default_image_model(request, settings)
        payload["model"] = requested_model
    backend, real_model = _dispatch(request, settings, requested_model)
    if backend.protocol != "comfyui":
        raise HTTPException(
            status_code=400,
            detail=(
                f"model {requested_model!r} is not a ComfyUI image model; "
                "use a model exposed from comfyui_targets"
            ),
        )
    return backend, real_model, requested_model


def _image_response(
    images: List[Any],
    payload: Dict[str, Any],
    *,
    revised_prompt: Optional[str] = None,
) -> JSONResponse:
    response_format = str(payload.get("response_format") or "b64_json")
    if response_format not in ("b64_json", "url"):
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'b64_json' or 'url'",
        )
    data = []
    for image in images:
        b64 = image.b64_json
        if response_format == "url":
            item = {
                "url": f"data:{image.mime_type};base64,{b64}",
                "mime_type": image.mime_type,
                "filename": image.filename,
            }
        else:
            item = {
                "b64_json": b64,
                "mime_type": image.mime_type,
                "filename": image.filename,
            }
        if revised_prompt is not None:
            item["revised_prompt"] = revised_prompt
        data.append(item)
    return JSONResponse(
        {
            "created": int(datetime.now(timezone.utc).timestamp()),
            "data": data,
        }
    )


async def _handle_openai_image_generation(request: Request) -> Any:
    app = request.app
    settings: Settings = app.state.settings
    payload, _ = await _image_payload_from_request(request)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing 'prompt'")
    backend, real_model, public_model = _dispatch_image_backend(
        request, settings, payload
    )
    target = backend.source
    params = _image_request_params(payload, target, edit=False, app=app)
    client: ComfyUIClient = _backend_client(app, backend)
    profile = settings.profile_for(f"{real_model}@{backend.name}")
    try:
        images = await client.generate_image(
            model=target.resolve_model(real_model),
            prompt=prompt,
            estimated_vram_gb=profile.estimated_vram_gb,
            estimated_memory_gb=profile.estimated_memory_gb,
            request_vram_headroom_gb=profile.request_vram_headroom_gb,
            min_free_vram_gb=profile.min_free_vram_gb,
            vram_cleanup_policy=profile.vram_cleanup_policy,
            exclusive_gpu=profile.exclusive_gpu,
            **params,
        )
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, payload)
        return _upstream_error(exc)
    return _image_response(images, payload)


async def _handle_openai_image_edit(request: Request) -> Any:
    app = request.app
    settings: Settings = app.state.settings
    payload, image_inputs = await _image_payload_from_request(request)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing 'prompt'")
    if not image_inputs:
        raise HTTPException(
            status_code=400,
            detail="missing image; send multipart field 'image' or JSON base64 'image'",
        )
    backend, real_model, public_model = _dispatch_image_backend(
        request, settings, payload
    )
    target = backend.source
    _enforce_reference_image_limit(target, "image_edit", image_inputs)
    params = _image_request_params(payload, target, edit=True, app=app)
    client: ComfyUIClient = _backend_client(app, backend)
    profile = settings.profile_for(f"{real_model}@{backend.name}")
    image_bytes, filename = image_inputs[0]
    try:
        images = await client.edit_image(
            model=target.resolve_model(real_model),
            prompt=prompt,
            image_bytes=image_bytes,
            filename=filename,
            image_inputs=image_inputs,
            estimated_vram_gb=profile.estimated_vram_gb,
            estimated_memory_gb=profile.estimated_memory_gb,
            request_vram_headroom_gb=profile.request_vram_headroom_gb,
            min_free_vram_gb=profile.min_free_vram_gb,
            vram_cleanup_policy=profile.vram_cleanup_policy,
            exclusive_gpu=profile.exclusive_gpu,
            **params,
        )
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, payload)
        return _upstream_error(exc)
    return _image_response(images, payload)


def _select_context_ir_provider(
    settings: Settings,
    profile: H3ContextIRProfile,
    requested: Any,
    *,
    has_images: bool,
    interface_name: str,
    external_connection: Optional[ExternalPlannerConnection] = None,
) -> H3ContextIRProvider:
    name = str(requested or "auto").strip()
    if name in ("", "auto"):
        if has_images and profile.default_multimodal_provider:
            name = profile.default_multimodal_provider
        else:
            name = profile.default_text_provider or profile.providers[0].name
    if name == "external":
        if not profile.allow_external_api:
            raise HTTPException(
                status_code=400,
                detail="this H3 Context-IR profile does not allow external APIs",
            )
        if external_connection is None:
            raise HTTPException(
                status_code=400,
                detail="external API connection details are required",
            )
        return H3ContextIRProvider(
            name="external",
            model=external_connection.model,
            target=f"external:{external_connection.protocol}",
            modalities=list(external_connection.modalities),
            json_mode=False,
        )
    if name.startswith("model:"):
        if not profile.allow_compatible_models:
            raise HTTPException(
                status_code=400,
                detail="this H3 Context-IR profile does not allow compatible model selection",
            )
        public_model_id = name[len("model:") :].strip()
        if not public_model_id:
            raise HTTPException(
                status_code=400,
                detail="compatible Planner selection is missing a public model id",
            )
        try:
            backend, display_model = settings.resolve_request(
                public_model_id, interface_name=interface_name
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if backend.protocol not in ("anthropic", "openai", "ollama"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"model {public_model_id!r} is not a compatible chat Planner"
                ),
            )
        model_profile = settings.profile_for(
            f"{display_model}@{backend.name}"
        )
        capabilities = set(model_profile.capabilities)
        if "completion" not in capabilities:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"model {public_model_id!r} does not declare the completion capability"
                ),
            )
        modalities = ["text"]
        if "vision" in capabilities:
            modalities.append("image")
        return H3ContextIRProvider(
            name=name,
            model=display_model,
            target=backend.name,
            modalities=modalities,
            json_mode=False,
        )
    provider = profile.provider_by_name(name)
    if provider is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown H3 Context-IR provider {name!r}; available: "
                f"{[item.name for item in profile.providers]}"
            ),
        )
    if not _context_ir_provider_is_exposed(settings, interface_name, provider):
        raise HTTPException(403, "Planner model is not exposed on this interface")
    return provider


def _image_media_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename or "")
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def _neutral_context_ir_messages(
    planning_request: str,
    images: List[Tuple[bytes, str]],
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "text": H3_CONTEXT_IR_SYSTEM_PROMPT, "images": []},
        {"role": "user", "text": planning_request, "images": images},
    ]


def _openai_context_ir_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages:
        images = message.get("images") or []
        if images:
            content: Any = [{"type": "text", "text": message["text"]}]
            for data, filename in images:
                media_type = _image_media_type(filename)
                encoded = base64.b64encode(data).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{encoded}"
                        },
                    }
                )
        else:
            content = message["text"]
        out.append({"role": message["role"], "content": content})
    return out


def _anthropic_context_ir_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts = [
        message["text"] for message in messages if message["role"] == "system"
    ]
    out: List[Dict[str, Any]] = []
    for message in messages:
        if message["role"] == "system":
            continue
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": message["text"]}
        ]
        for data, filename in message.get("images") or []:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _image_media_type(filename),
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        out.append({"role": message["role"], "content": content})
    return "\n\n".join(system_parts), out


def _ollama_context_ir_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages:
        item: Dict[str, Any] = {
            "role": message["role"],
            "content": message["text"],
        }
        images = message.get("images") or []
        if images:
            item["images"] = [
                base64.b64encode(data).decode("ascii") for data, _ in images
            ]
        out.append(item)
    return out


def _text_from_openai_response(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("OpenAI-compatible provider returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
        if text.strip():
            return text
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    raise ValueError("OpenAI-compatible provider returned empty content")


def _text_from_anthropic_response(data: Dict[str, Any]) -> str:
    content = data.get("content") or []
    text = "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )
    if not text.strip():
        raise ValueError("Anthropic-compatible provider returned empty content")
    return text


async def _invoke_context_ir_provider(
    app: FastAPI,
    profile: H3ContextIRProfile,
    provider: H3ContextIRProvider,
    messages: List[Dict[str, Any]],
    external_connection: Optional[ExternalPlannerConnection] = None,
) -> Tuple[str, Dict[str, Any]]:
    if provider.name == "external":
        if external_connection is None:
            raise ValueError("external Planner connection is unavailable")
        async with _external_planner_http_client(app) as http_client:
            if external_connection.protocol == "anthropic":
                client = AnthropicClient(
                    external_connection.base_url,
                    external_connection.auth_token,
                    timeout=app.state.settings.timeout_seconds,
                    trust_env=app.state.settings.use_system_proxy,
                    client=http_client,
                )
                system, wire_messages = _anthropic_context_ir_messages(messages)
                response = await client.messages(
                    {
                        "model": external_connection.model,
                        "system": system,
                        "messages": wire_messages,
                        "max_tokens": profile.max_output_tokens,
                        "temperature": profile.temperature,
                        "stream": False,
                    }
                )
                return _text_from_anthropic_response(response), dict(
                    response.get("usage") or {}
                )
            client = OpenAIClient(
                external_connection.base_url,
                auth_token=external_connection.auth_token,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
                upstream_name="playground-external",
                client=http_client,
            )
            response = await client.chat(
                {
                    "model": external_connection.model,
                    "messages": _openai_context_ir_messages(messages),
                    "temperature": profile.temperature,
                    "max_tokens": profile.max_output_tokens,
                    "stream": False,
                }
            )
            return _text_from_openai_response(response), dict(
                response.get("usage") or {}
            )

    settings: Settings = app.state.settings
    backend = settings.backend_by_name(provider.target)
    if backend is None:
        raise ValueError(f"context IR target {provider.target!r} is unavailable")
    wire_model = backend.source.resolve_model(provider.model)
    model_profile = settings.profile_for(f"{provider.model}@{provider.target}")
    client = _backend_client(app, backend)

    if backend.protocol == "anthropic":
        system, wire_messages = _anthropic_context_ir_messages(messages)
        response = await client.messages(
            {
                "model": wire_model,
                "system": system,
                "messages": wire_messages,
                "max_tokens": profile.max_output_tokens,
                "temperature": profile.temperature,
                "stream": False,
            }
        )
        return _text_from_anthropic_response(response), dict(response.get("usage") or {})

    if backend.protocol == "ollama":
        payload = {
            "model": wire_model,
            "messages": _ollama_context_ir_messages(messages),
            "stream": False,
            "think": False,
            "options": {
                "temperature": profile.temperature,
                "num_predict": profile.max_output_tokens,
            },
        }
        response = await client.chat(
            payload,
            estimated_vram_gb=model_profile.estimated_vram_gb,
            estimated_memory_gb=model_profile.estimated_memory_gb,
        )
        message = response.get("message") or {}
        text = str(message.get("content") or message.get("thinking") or "")
        if not text.strip():
            raise ValueError("Ollama provider returned empty content")
        usage = {
            "prompt_tokens": response.get("prompt_eval_count"),
            "completion_tokens": response.get("eval_count"),
        }
        return text, {key: value for key, value in usage.items() if value is not None}

    if backend.protocol == "openai":
        payload = {
            "model": wire_model,
            "messages": _openai_context_ir_messages(messages),
            "temperature": profile.temperature,
            "max_tokens": profile.max_output_tokens,
            "stream": False,
        }
        if provider.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if backend.kind == "local":
            if isinstance(backend.source, LlamaCppTarget):
                _apply_llama_cpp_thinking_config(
                    settings,
                    f"{provider.model}@{provider.target}",
                    {"thinking": {"type": "disabled"}},
                    payload,
                )
            response = await client.chat(
                payload,
                estimated_vram_gb=model_profile.estimated_vram_gb,
                estimated_memory_gb=model_profile.estimated_memory_gb,
            )
        else:
            response = await client.chat(payload)
        return _text_from_openai_response(response), dict(response.get("usage") or {})

    raise ValueError(
        f"context IR provider cannot use backend protocol {backend.protocol!r}"
    )


async def _run_h3_context_ir(
    app: FastAPI,
    profile: H3ContextIRProfile,
    *,
    prompt: str,
    image_inputs: List[Tuple[bytes, str]],
    requested_mode: str,
    duration_seconds: float,
    interface_name: str,
    requested_provider: Any = None,
    external_connection: Optional[ExternalPlannerConnection] = None,
) -> Dict[str, Any]:
    try:
        mode = resolve_base_mode(requested_mode, len(image_inputs))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    provider = _select_context_ir_provider(
        settings=app.state.settings,
        profile=profile,
        requested=requested_provider,
        has_images=bool(image_inputs),
        interface_name=interface_name,
        external_connection=external_connection,
    )
    warnings: List[str] = []
    provider_images = image_inputs
    if image_inputs and not provider.accepts_images:
        provider_images = []
        warnings.append(
            f"provider {provider.name!r} is text-only; reference images were not "
            "shown to the planner, but H3 picture labels were preserved"
        )
    planning_request = build_planning_request(
        prompt,
        mode=mode,
        duration_seconds=duration_seconds,
        image_count=len(image_inputs),
    )
    messages = _neutral_context_ir_messages(planning_request, provider_images)
    plan = None
    raw_response = ""
    usage: Dict[str, Any] = {}
    attempts = 0
    last_error = ""
    upstream_error = False
    for attempts in range(1, profile.max_attempts + 1):
        try:
            raw_response, usage = await _invoke_context_ir_provider(
                app,
                profile,
                provider,
                messages,
                external_connection=external_connection,
            )
            plan = parse_and_validate_plan(
                raw_response,
                expected_mode=mode,
                expected_duration_seconds=duration_seconds,
            )
            break
        except httpx.HTTPError as exc:
            last_error = _read_error_text(exc)
            upstream_error = True
            break
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
            if attempts < profile.max_attempts:
                messages.extend(
                    [
                        {"role": "assistant", "text": raw_response, "images": []},
                        {
                            "role": "user",
                            "text": build_repair_request(last_error),
                            "images": [],
                        },
                    ]
                )

    used_fallback = plan is None
    if plan is None:
        if profile.failure_mode == "error":
            raise HTTPException(
                status_code=502 if upstream_error else 422,
                detail=f"H3 Context-IR provider failed: {last_error}",
            )
        warnings.append(
            "planner failed validation; used a lossless one-shot fallback: "
            + (last_error or "unknown error")
        )
        plan = fallback_plan(
            prompt, mode=mode, duration_seconds=duration_seconds
        )

    rendered = render_base_prompt(plan)
    return {
        "id": "h3cir_" + secrets.token_hex(12),
        "object": "video.context_ir",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": profile.public_model_id,
        "provider": {
            "name": provider.name,
            "model": provider.model,
            "target": provider.target,
            "modalities": provider.modalities,
        },
        "mode": mode,
        "duration_seconds": duration_seconds,
        "content": {"prompt": rendered},
        "ir": plan.model_dump(),
        "fallback": used_fallback,
        "attempts": attempts,
        "warnings": warnings,
        "usage": usage,
    }


async def _release_context_ir_provider_for_video(
    app: FastAPI,
    provider_info: Dict[str, Any],
) -> bool:
    """Release a managed local Planner before admitting the H3 video model.

    H3 needs almost the entire VRAM budget of a 24 GiB card.  Waiting for the
    normal idle-reclaim window after a local Planner call makes the default
    ``prompt_mode=auto`` path fail even though the two models only need to run
    sequentially.  The regular participant release hook is concurrency-aware
    and refuses to stop an unowned or still-active backend.
    """

    target = str(provider_info.get("target") or "").strip()
    if not target or target == "request-scoped":
        return False
    backend = app.state.settings.backend_by_name(target)
    if backend is None or backend.kind != "local":
        return False
    client = _backend_client(app, backend)
    release = getattr(client, "release_for_vram", None)
    if not callable(release):
        return False
    try:
        released = bool(await release())
    except Exception:
        logger.warning(
            "failed to release local H3 Context-IR provider %s before video handoff",
            target,
            exc_info=True,
        )
        return False
    if released:
        logger.info(
            "released local H3 Context-IR provider %s before video model admission",
            target,
        )
    else:
        logger.info(
            "local H3 Context-IR provider %s was not releasable; video resource "
            "admission will use the current free-resource reading",
            target,
        )
    return released


async def _handle_h3_context_ir(request: Request) -> JSONResponse:
    settings, iface = _model_interface_for_request(request)
    payload, image_inputs = await _image_payload_from_request(request)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing 'prompt'")
    requested_profile = str(
        payload.get("model") or payload.get("profile") or ""
    ).strip()
    profile = settings.h3_context_ir_profile_by_name(requested_profile)
    if profile is None or not profile.enabled:
        raise HTTPException(
            status_code=404,
            detail=f"H3 Context-IR profile {requested_profile!r} not found",
        )
    if len(image_inputs) > 2:
        raise HTTPException(
            status_code=400,
            detail="H3 base Context-IR accepts at most two reference images",
        )
    for data, filename in image_inputs:
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"reference image {filename!r} exceeds 20 MiB",
            )
    duration = _context_ir_duration(payload, profile)
    requested_provider = payload.get("provider")
    external_connection = _external_planner_connection(
        request,
        profile,
        payload,
        requested_provider,
    )
    result = await _run_h3_context_ir(
        request.app,
        profile,
        prompt=prompt,
        image_inputs=image_inputs,
        requested_mode=str(payload.get("mode") or "auto"),
        duration_seconds=duration,
        interface_name=iface.name,
        requested_provider=requested_provider,
        external_connection=external_connection,
    )
    return JSONResponse(result)


async def _handle_openai_video_generation(request: Request) -> Any:
    app = request.app
    settings: Settings = app.state.settings
    payload, image_inputs = await _image_payload_from_request(request)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing 'prompt'")
    backend, real_model, public_model = _dispatch_image_backend(
        request, settings, payload
    )
    iface = _interface_for(request)
    if iface is None:  # pragma: no cover - dispatch already requires an interface
        raise HTTPException(status_code=404, detail="unknown listener")
    target = backend.source
    params = _video_request_params(payload, target, app=app)
    try:
        image_inputs, video_inputs, keyframe_inputs = await video_media_from_request(
            request, payload, image_inputs, frames=params["num_frames"], fps=params["frame_rate"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _enforce_reference_image_limit(target, "video_generation", image_inputs)
    revised_prompt: Optional[str] = None
    context_ir_profile_name = getattr(target, "context_ir_profile", None)
    video_mode = "auto"
    requested_mode = str(payload.get("video_mode") or payload.get("context_ir_mode") or "auto").strip().lower()
    legacy_mode = str(payload.get("context_ir_mode") or "auto").strip().lower()
    if requested_mode != "auto" and legacy_mode not in ("auto", requested_mode):
        raise HTTPException(status_code=400, detail="video_mode conflicts with context_ir_mode")
    if requested_mode == "auto":
        requested_mode = legacy_mode
    raw_media = requested_mode == "ref2va" or bool(keyframe_inputs)
    workflows = resolve_workflows(target.workflow_config())
    if raw_media:
        if getattr(target, "preset", "") != "minimax_h3":
            raise HTTPException(status_code=400, detail="Ref2VA/keyframes require the minimax_h3 preset")
        if str(payload.get("prompt_mode") or "").lower() != "raw":
            raise HTTPException(status_code=400, detail="Ref2VA/keyframes require prompt_mode=raw; Context-IR supports base modes only")
    if video_inputs and requested_mode != "ref2va":
        raise HTTPException(status_code=400, detail="reference videos require video_mode=ref2va")
    if requested_mode == "ref2va":
        if workflows.get("ref2va") is None or not (image_inputs or video_inputs):
            raise HTTPException(status_code=400, detail="Ref2VA requires a configured workflow and at least one reference image or video")
        video_mode = "ref2va"
    elif context_ir_profile_name or getattr(target, "preset", "") == "minimax_h3":
        try:
            video_mode = resolve_base_mode(
                requested_mode,
                len(image_inputs),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif requested_mode != "auto":
        raise HTTPException(status_code=400, detail="video_mode is only supported for H3 targets")
    if keyframe_inputs:
        occupied = set()
        if video_mode in {"i2va", "fl2va"}:
            occupied.add(0)
        if video_mode in {"l2va", "fl2va"}:
            occupied.add(params["num_frames"] - 1)
        if any(index in occupied for index, _, _ in keyframe_inputs):
            raise HTTPException(status_code=400, detail="keyframe overlaps a first/last frame constraint")
    if context_ir_profile_name and not raw_media:
        context_profile = settings.h3_context_ir_profile_by_name(
            context_ir_profile_name
        )
        if context_profile is None or not context_profile.enabled:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"configured H3 Context-IR profile {context_ir_profile_name!r} "
                    "is unavailable"
                ),
            )
        prompt_mode = str(
            payload.get("prompt_mode")
            or getattr(target, "context_ir_prompt_mode", "auto")
        ).strip().lower()
        if prompt_mode not in ("raw", "auto", "enhance"):
            raise HTTPException(
                status_code=400,
                detail="prompt_mode must be raw|auto|enhance",
            )
        should_enhance = prompt_mode == "enhance" or (
            prompt_mode == "auto" and not is_structured_base_prompt(prompt)
        )
        if should_enhance:
            requested_provider = payload.get("context_ir_provider")
            external_connection = _external_planner_connection(
                request,
                context_profile,
                payload,
                requested_provider,
            )
            context_result = await _run_h3_context_ir(
                app,
                context_profile,
                prompt=prompt,
                image_inputs=image_inputs,
                requested_mode=video_mode,
                duration_seconds=_context_ir_duration(
                    payload, context_profile, video_params=params
                ),
                interface_name=iface.name,
                requested_provider=requested_provider,
                external_connection=external_connection,
            )
            revised_prompt = str(context_result["content"]["prompt"])
            prompt = revised_prompt
            await _release_context_ir_provider_for_video(
                app,
                dict(context_result.get("provider") or {}),
            )
    client: ComfyUIClient = _backend_client(app, backend)
    profile = settings.profile_for(f"{real_model}@{backend.name}")
    image_bytes = None
    filename = None
    if image_inputs:
        image_bytes, filename = image_inputs[0]
    try:
        videos = await client.generate_video(
            model=target.resolve_model(real_model),
            prompt=prompt,
            image_bytes=image_bytes,
            filename=filename,
            image_inputs=image_inputs or None,
            video_mode=video_mode,
            video_inputs=video_inputs or None,
            keyframe_inputs=keyframe_inputs or None,
            estimated_vram_gb=profile.estimated_vram_gb,
            estimated_memory_gb=profile.estimated_memory_gb,
            request_vram_headroom_gb=profile.request_vram_headroom_gb,
            min_free_vram_gb=profile.min_free_vram_gb,
            vram_cleanup_policy=profile.vram_cleanup_policy,
            exclusive_gpu=profile.exclusive_gpu,
            **params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, payload)
        return _upstream_error(exc)
    return _image_response(videos, payload, revised_prompt=revised_prompt)
