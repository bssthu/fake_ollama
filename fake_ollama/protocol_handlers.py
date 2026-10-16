"""Chat protocol orchestration over configured backend clients."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from .config import LlamaCppTarget, GenericOpenAITarget, Settings, estimate_tokens_from_anthropic_payload
from .converters import AnthropicStreamTranslator, OpenAIChatStreamTranslator, anthropic_to_ollama_chat, anthropic_to_ollama_generate, anthropic_to_openai_chat, ollama_chat_to_anthropic, ollama_generate_to_anthropic, openai_chat_to_anthropic
from .llama_cpp_client import LlamaCppClient
from .generic_openai_client import GenericOpenAIClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .stream_state import AnthropicStreamState, checked_openai_lines
from .protocol_controls import include_usage_requested
from .protocol_usage import token_count
from .protocol_tools import UpstreamProtocolError
from .vram import LocalTargetResourceError
from .reverse_converters import anthropic_to_ollama_chat as anthropic_to_ollama_chat_payload, anthropic_to_openai_chat as anthropic_to_openai_chat_payload, ollama_chat_to_anthropic as ollama_chat_to_anthropic_response, ollama_stream_to_anthropic_events, ollama_stream_to_anthropic_sse, openai_chat_to_anthropic as openai_chat_to_anthropic_response, openai_stream_to_anthropic_sse
from .proxy_errors import (
    _anthropic_error_response,
    _log_upstream_error,
    _read_error_text,
    _upstream_error,
)
from .request_validation import (
    _apply_llama_cpp_thinking_config,
    _apply_ollama_thinking_config,
    _apply_reverse_output_limits,
    _apply_thinking_config,
    _count_tokens_payload,
    _enforce_limits,
)
from .routing import (
    _backend_client,
    _dispatch,
)

async def _openai_upstream_messages(
    oc: OpenAIClient,
    anthropic_payload: Dict[str, Any],
    *,
    anthropic_model: str,
    target_model: str,
    default_max_tokens: int,
    show_thinking: bool,
) -> Dict[str, Any]:
    """Round-trip an Anthropic-shape payload through an OpenAI upstream."""
    openai_payload = anthropic_to_openai_chat_payload(
        anthropic_payload,
        target_model=target_model,
        default_max_tokens=default_max_tokens,
    )
    openai_resp = await oc.chat(openai_payload)
    return openai_chat_to_anthropic_response(
        openai_resp,
        anthropic_model=anthropic_model,
        show_thinking=show_thinking,
    )


async def _openai_upstream_stream_messages(
    oc: OpenAIClient,
    anthropic_payload: Dict[str, Any],
    *,
    anthropic_model: str,
    target_model: str,
    default_max_tokens: int,
    show_thinking: bool,
) -> AsyncIterator[tuple[str, Dict[str, Any]]]:
    """Stream an Anthropic-shape request through an OpenAI upstream as events."""
    from .reverse_converters import openai_stream_to_anthropic_events

    openai_payload = anthropic_to_openai_chat_payload(
        anthropic_payload,
        target_model=target_model,
        default_max_tokens=default_max_tokens,
    )
    lines = oc.stream_chat(openai_payload)
    async for event_type, data in openai_stream_to_anthropic_events(
        lines, anthropic_model=anthropic_model, show_thinking=show_thinking
    ):
        yield event_type, data


async def _handle_ollama_via_openai_upstream(
    request: Request,
    payload: Dict[str, Any],
    backend: Any,
    real_model: str,
    *,
    mode: str,
) -> Any:
    """Ollama frontend (``/api/chat`` or ``/api/generate``) over an OpenAI upstream.

    Pipeline: Ollama JSON -> Anthropic canonical -> OpenAI Chat
    Completions -> Anthropic canonical -> Ollama JSON. The two
    conversion steps re-use the existing forward/reverse converters so
    new wire formats only need an Anthropic adapter pair to plug into
    every existing surface.
    """
    app = request.app
    settings: Settings = app.state.settings
    ollama_model = payload.get("model") or ""
    upstream_model = backend.source.resolve_model(real_model)
    oc = _backend_client(app, backend)
    profile = settings.profile_for(ollama_model)

    if mode == "chat":
        anthropic_payload = ollama_chat_to_anthropic(
            payload,
            upstream_model=ollama_model,
            default_max_tokens=settings.default_max_tokens,
        )
    else:
        anthropic_payload = ollama_generate_to_anthropic(
            payload,
            upstream_model=ollama_model,
            default_max_tokens=settings.default_max_tokens,
        )
    stream = bool(payload.get("stream", True))
    anthropic_payload["stream"] = stream
    _apply_thinking_config(settings, ollama_model, anthropic_payload)
    _enforce_limits(settings, ollama_model, anthropic_payload)

    if not stream:
        try:
            anthropic_resp = await _openai_upstream_messages(
                oc,
                anthropic_payload,
                anthropic_model=ollama_model,
                target_model=upstream_model,
                default_max_tokens=settings.default_max_tokens,
                show_thinking=profile.show_thinking,
            )
        except httpx.HTTPError as exc:
            _log_upstream_error(request, exc, anthropic_payload)
            return _upstream_error(exc)
        if mode == "chat":
            return JSONResponse(
                anthropic_to_ollama_chat(
                    anthropic_resp,
                    ollama_model=ollama_model,
                    show_thinking=profile.show_thinking,
                )
            )
        return JSONResponse(
            anthropic_to_ollama_generate(
                anthropic_resp,
                ollama_model=ollama_model,
                show_thinking=profile.show_thinking,
            )
        )

    async def body() -> AsyncIterator[bytes]:
        translator = AnthropicStreamTranslator(
            ollama_model, mode=mode, show_thinking=profile.show_thinking
        )
        try:
            async for event_type, data in _openai_upstream_stream_messages(
                oc,
                anthropic_payload,
                anthropic_model=ollama_model,
                target_model=upstream_model,
                default_max_tokens=settings.default_max_tokens,
                show_thinking=profile.show_thinking,
            ):
                for chunk in translator.feed_event(event_type, data):
                    yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
            for chunk in translator.finish():
                yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception as exc:
            _log_upstream_error(request, exc, anthropic_payload)
            err_chunk = {
                "model": ollama_model,
                "created_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "done": True,
                "error": _read_error_text(exc),
            }
            yield (json.dumps(err_chunk, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(body(), media_type="application/x-ndjson")


async def _handle(request: Request, *, mode: str) -> Any:
    app = request.app
    settings: Settings = app.state.settings

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    requested = payload.get("model") or ""
    if not requested:
        raise HTTPException(status_code=400, detail="missing 'model'")
    backend, real_model = _dispatch(request, settings, requested)
    # ``ollama_model`` is the composite id (e.g. ``llama3.1@my-target``)
    # which we keep echoing back to the client in response payloads so
    # they keep seeing the same identifier they sent us.
    ollama_model = requested

    if backend.protocol == "openai" and backend.kind == "remote":
        return await _handle_ollama_via_openai_upstream(
            request, payload, backend, real_model, mode=mode
        )
    if backend.protocol != "anthropic":
        raise HTTPException(
            status_code=400,
            detail=(
                f"model {requested!r} resolves to {backend.protocol}/{backend.kind} "
                f"backend; the Ollama /api/chat surface only forwards to "
                f"anthropic upstreams or openai remote upstreams. Use "
                f"/v1/chat/completions or /v1/messages instead."
            ),
        )

    upstream_model = backend.source.resolve_model(real_model)
    client = _backend_client(app, backend)

    if mode == "chat":
        upstream_payload = ollama_chat_to_anthropic(
            payload,
            upstream_model=upstream_model,
            default_max_tokens=settings.default_max_tokens,
        )
    else:
        upstream_payload = ollama_generate_to_anthropic(
            payload,
            upstream_model=upstream_model,
            default_max_tokens=settings.default_max_tokens,
        )

    stream = bool(payload.get("stream", True))
    upstream_payload["stream"] = stream
    _apply_thinking_config(settings, ollama_model, upstream_payload)
    _enforce_limits(settings, ollama_model, upstream_payload)

    profile = settings.profile_for(ollama_model)

    if not stream:
        try:
            data = await client.messages(upstream_payload)
        except httpx.HTTPError as exc:
            _log_upstream_error(request, exc, upstream_payload)
            return _upstream_error(exc)
        if mode == "chat":
            return JSONResponse(
                anthropic_to_ollama_chat(
                    data, ollama_model=ollama_model, show_thinking=profile.show_thinking
                )
            )
        return JSONResponse(
            anthropic_to_ollama_generate(
                data, ollama_model=ollama_model, show_thinking=profile.show_thinking
            )
        )

    async def body() -> AsyncIterator[bytes]:
        translator = AnthropicStreamTranslator(
            ollama_model, mode=mode, show_thinking=profile.show_thinking
        )
        try:
            async for event_type, data in client.stream_messages(upstream_payload):
                for chunk in translator.feed_event(event_type, data):
                    yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
            for chunk in translator.finish():
                yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception as exc:
            _log_upstream_error(request, exc, upstream_payload)
            err_chunk = {
                "model": ollama_model,
                "created_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "done": True,
                "error": _read_error_text(exc),
            }
            yield (json.dumps(err_chunk, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(body(), media_type="application/x-ndjson")


async def _handle_openai_chat(request: Request) -> Any:
    app = request.app
    settings: Settings = app.state.settings

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    openai_model = payload.get("model") or ""
    if not openai_model:
        raise HTTPException(status_code=400, detail="missing 'model'")

    backend, real_model = _dispatch(request, settings, openai_model)
    profile = settings.profile_for(openai_model)
    if backend.protocol == "comfyui":
        raise HTTPException(
            status_code=400,
            detail=(
                f"model {openai_model!r} is a ComfyUI image model; use "
                "/v1/images/generations or /v1/images/edits"
            ),
        )

    # Map the resolved backend back onto the four legacy local variables
    # the per-protocol branches below still reference. ``real_model`` is
    # the bare model name the backend advertises (no ``@target``
    # suffix); ``openai_model`` keeps the composite id so response
    # payloads echo back what the client sent.
    target = backend.source if (backend.protocol == "ollama") else None
    llama_target = (
        backend.source
        if isinstance(backend.source, LlamaCppTarget)
        else None
    )
    generic_openai_target = (
        backend.source
        if isinstance(backend.source, GenericOpenAITarget)
        else None
    )
    openai_up_backend = (
        backend if (backend.protocol == "openai" and backend.kind == "remote") else None
    )
    anthropic_backend = backend if backend.protocol == "anthropic" else None

    if target is not None:
        oc: OllamaClient = app.state.ollama_clients.get(target.name)
        if oc is None:
            raise HTTPException(
                status_code=503,
                detail=f"ollama_target '{target.name}' is not initialised",
            )
        anthropic_payload = openai_chat_to_anthropic(
            payload,
            upstream_model=openai_model,
            default_max_tokens=settings.default_max_tokens,
        )
        stream = bool(payload.get("stream", False))
        anthropic_payload["stream"] = stream
        ollama_payload = anthropic_to_ollama_chat_payload(
            anthropic_payload,
            target_model=target.resolve_model(real_model),
            default_max_tokens=settings.default_max_tokens,
        )
        _apply_ollama_thinking_config(
            settings, openai_model, anthropic_payload, ollama_payload
        )
        _apply_reverse_output_limits(settings, openai_model, ollama_payload)
        if not stream:
            try:
                resp = await oc.chat(
                    ollama_payload,
                    estimated_vram_gb=profile.estimated_vram_gb,
                    estimated_memory_gb=profile.estimated_memory_gb,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, ollama_payload)
                return _upstream_error(exc)
            anthropic_resp = ollama_chat_to_anthropic_response(
                resp, anthropic_model=openai_model
            )
            return JSONResponse(
                anthropic_to_openai_chat(
                    anthropic_resp,
                    openai_model=openai_model,
                    show_thinking=profile.show_thinking,
                )
            )

        async def body_local() -> AsyncIterator[bytes]:
            translator = OpenAIChatStreamTranslator(
                openai_model, show_thinking=profile.show_thinking, include_usage=include_usage_requested(payload)
            )
            try:
                lines = oc.stream_chat(
                    ollama_payload,
                    estimated_vram_gb=profile.estimated_vram_gb,
                    estimated_memory_gb=profile.estimated_memory_gb,
                )
                async for event_type, data in ollama_stream_to_anthropic_events(
                    lines, anthropic_model=openai_model
                ):
                    for frame in translator.feed_event(event_type, data):
                        yield (
                            "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
                        ).encode("utf-8")
                for frame in translator.finish():
                    yield ("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode("utf-8")
            except Exception as exc:
                translator._state.failed = True
                _log_upstream_error(request, exc, ollama_payload)
                err_frame = {"error": {"type": "upstream_error", "message": _read_error_text(exc)}}
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
            if not translator._state.failed:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(body_local(), media_type="text/event-stream")

    if llama_target is not None:
        lc: LlamaCppClient = app.state.llama_cpp_clients.get(llama_target.name)
        if lc is None:
            raise HTTPException(
                status_code=503,
                detail=f"llama_cpp_target '{llama_target.name}' is not initialised",
            )
        llama_payload = dict(payload)
        llama_payload["model"] = llama_target.resolve_model(real_model)
        stream = bool(payload.get("stream", False))
        llama_payload["stream"] = stream
        _apply_llama_cpp_thinking_config(
            settings, openai_model, payload, llama_payload
        )
        estimated_vram_gb = profile.estimated_vram_gb
        estimated_memory_gb = profile.estimated_memory_gb
        if not stream:
            try:
                resp = await lc.chat(
                    llama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, llama_payload)
                return _upstream_error(exc)
            if isinstance(resp, dict):
                resp = dict(resp)
                resp["model"] = openai_model
            return JSONResponse(resp)

        async def body_llama_openai() -> AsyncIterator[bytes]:
            try:
                async for line in checked_openai_lines(lc.stream_chat(
                    llama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )):
                    if line.startswith("data:"):
                        yield (line + "\n\n").encode("utf-8")
                    else:
                        yield ("data: " + line + "\n\n").encode("utf-8")
            except Exception as exc:
                _log_upstream_error(request, exc, llama_payload)
                err_frame = {"error": {"type": "upstream_error", "message": _read_error_text(exc)}}
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                return

        return StreamingResponse(body_llama_openai(), media_type="text/event-stream")

    if generic_openai_target is not None:
        loc: GenericOpenAIClient = app.state.generic_openai_clients.get(
            generic_openai_target.name
        )
        if loc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"generic_openai_target '{generic_openai_target.name}' is not "
                    "initialised"
                ),
            )
        forward_payload = dict(payload)
        forward_payload["model"] = generic_openai_target.resolve_model(real_model)
        stream = bool(payload.get("stream", False))
        forward_payload["stream"] = stream
        estimated_vram_gb = profile.estimated_vram_gb
        estimated_memory_gb = profile.estimated_memory_gb
        if not stream:
            try:
                resp = await loc.chat(
                    forward_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, forward_payload)
                return _upstream_error(exc)
            if isinstance(resp, dict):
                resp = dict(resp)
                resp["model"] = openai_model
            return JSONResponse(resp)

        async def body_generic_openai() -> AsyncIterator[bytes]:
            try:
                async for line in checked_openai_lines(loc.stream_chat(
                    forward_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )):
                    if line.startswith("data:"):
                        yield (line + "\n\n").encode("utf-8")
                    else:
                        yield ("data: " + line + "\n\n").encode("utf-8")
            except Exception as exc:
                _log_upstream_error(request, exc, forward_payload)
                err_frame = {"error": {"type": "upstream_error", "message": _read_error_text(exc)}}
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                return

        return StreamingResponse(body_generic_openai(), media_type="text/event-stream")

    if openai_up_backend is not None:
        oc_remote = _backend_client(app, openai_up_backend)
        forward_payload = dict(payload)
        forward_payload["model"] = openai_up_backend.source.resolve_model(real_model)
        stream = bool(payload.get("stream", False))
        forward_payload["stream"] = stream
        # No thinking_config / _enforce_limits here: remote OpenAI gateways
        # are passthrough; the user already controls those knobs in the
        # request they send us.
        if not stream:
            try:
                resp = await oc_remote.chat(forward_payload)
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, forward_payload)
                return _upstream_error(exc)
            if isinstance(resp, dict):
                resp = dict(resp)
                resp["model"] = openai_model
            return JSONResponse(resp)

        async def body_openai_upstream() -> AsyncIterator[bytes]:
            try:
                async for line in checked_openai_lines(oc_remote.stream_chat(forward_payload)):
                    if line.startswith("data:"):
                        yield (line + "\n\n").encode("utf-8")
                    else:
                        yield ("data: " + line + "\n\n").encode("utf-8")
            except Exception as exc:
                _log_upstream_error(request, exc, forward_payload)
                err_frame = {"error": {"type": "upstream_error", "message": _read_error_text(exc)}}
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                return

        return StreamingResponse(body_openai_upstream(), media_type="text/event-stream")

    # Fallback: anthropic backend
    if anthropic_backend is None:
        raise HTTPException(
            status_code=500,
            detail=f"unhandled backend {backend.name!r} ({backend.protocol}/{backend.kind})",
        )
    upstream_model = anthropic_backend.source.resolve_model(real_model)
    client = _backend_client(app, anthropic_backend)

    upstream_payload = openai_chat_to_anthropic(
        payload,
        upstream_model=upstream_model,
        default_max_tokens=settings.default_max_tokens,
    )
    stream = bool(payload.get("stream", False))
    upstream_payload["stream"] = stream
    _apply_thinking_config(settings, openai_model, upstream_payload)
    _enforce_limits(settings, openai_model, upstream_payload)

    if not stream:
        try:
            data = await client.messages(upstream_payload)
        except httpx.HTTPError as exc:
            _log_upstream_error(request, exc, upstream_payload)
            return _upstream_error(exc)
        return JSONResponse(
            anthropic_to_openai_chat(
                data, openai_model=openai_model, show_thinking=profile.show_thinking
            )
        )

    async def body() -> AsyncIterator[bytes]:
        translator = OpenAIChatStreamTranslator(
            openai_model, show_thinking=profile.show_thinking, include_usage=include_usage_requested(payload)
        )
        try:
            async for event_type, data in client.stream_messages(upstream_payload):
                for frame in translator.feed_event(event_type, data):
                    yield (
                        "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
                    ).encode("utf-8")
            for frame in translator.finish():
                yield ("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode("utf-8")
        except Exception as exc:
            translator._state.failed = True
            _log_upstream_error(request, exc, upstream_payload)
            err_frame = {"error": {"type": "upstream_error", "message": _read_error_text(exc)}}
            yield (
                "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
            ).encode("utf-8")
        if not translator._state.failed:
            yield b"data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


async def _handle_anthropic_count_tokens(request: Request) -> Any:
    """Handle ``POST /v1/messages/count_tokens``.

    Anthropic returns ``{"input_tokens": int}``. For local targets we return
    the project's conservative local estimate without starting the model. For
    upstream passthrough models, use the upstream count_tokens endpoint so
    official Anthropic-compatible servers can provide exact counts.
    """
    app = request.app
    settings: Settings = app.state.settings

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    anth_model = payload.get("model")
    if not anth_model:
        raise HTTPException(status_code=400, detail="missing 'model'")

    backend, real_model = _dispatch(request, settings, anth_model)

    # Only Anthropic upstreams expose a real ``/v1/messages/count_tokens``
    # endpoint. For every other backend (Ollama daemons, llama.cpp
    # servers, remote OpenAI-compatible upstreams) we return the
    # project's conservative local estimate without spinning the model
    # up or making a wire call.
    if backend.protocol != "anthropic":
        return JSONResponse(
            {"input_tokens": estimate_tokens_from_anthropic_payload(payload)}
        )

    # Anthropic upstream: forward to its count_tokens endpoint.
    upstream_payload = _count_tokens_payload(
        payload,
        model=backend.source.resolve_model(real_model),
    )
    client = _backend_client(app, backend)
    try:
        data = await client.count_tokens(
            upstream_payload,
            params=dict(request.query_params),
        )
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, upstream_payload)
        return _upstream_error(exc)
    count = token_count(data.get("input_tokens"))
    if count is None:
        return _anthropic_error_response(UpstreamProtocolError("Upstream omitted a valid input token count"))
    return JSONResponse({"input_tokens": count})


async def _handle_anthropic_messages(request: Request) -> Any:
    """Handle ``POST /v1/messages``.

    If the requested model is served by an ``ollama_targets`` entry, convert
    the Anthropic request to Ollama format and call the local daemon.
    Otherwise, pass it through to the matching upstream Anthropic-compatible
    server (so this same endpoint also doubles as a transparent proxy /
    auth shim).
    """
    app = request.app
    settings: Settings = app.state.settings

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    anth_model = payload.get("model")
    if not anth_model:
        raise HTTPException(status_code=400, detail="missing 'model'")
    stream = bool(payload.get("stream", False))

    backend, real_model = _dispatch(request, settings, anth_model)
    if backend.protocol == "comfyui":
        raise HTTPException(
            status_code=400,
            detail=(
                f"model {anth_model!r} is a ComfyUI image model; use "
                "/v1/images/generations or /v1/images/edits"
            ),
        )
    target = backend.source if backend.protocol == "ollama" else None
    llama_target = (
        backend.source
        if isinstance(backend.source, LlamaCppTarget)
        else None
    )
    generic_openai_target = (
        backend.source
        if isinstance(backend.source, GenericOpenAITarget)
        else None
    )
    openai_up_backend = (
        backend if (backend.protocol == "openai" and backend.kind == "remote") else None
    )
    anthropic_backend = backend if backend.protocol == "anthropic" else None

    if target is not None:
        oc: OllamaClient = app.state.ollama_clients.get(target.name)
        if oc is None:
            raise HTTPException(
                status_code=503,
                detail=f"ollama_target '{target.name}' is not initialised",
            )
        ollama_payload = anthropic_to_ollama_chat_payload(
            payload,
            target_model=target.resolve_model(real_model),
            default_max_tokens=settings.default_max_tokens,
        )
        _apply_ollama_thinking_config(settings, anth_model, payload, ollama_payload)
        _apply_reverse_output_limits(settings, anth_model, ollama_payload)
        _anth_ollama_profile = settings.profile_for(anth_model)
        estimated_vram_gb = _anth_ollama_profile.estimated_vram_gb
        estimated_memory_gb = _anth_ollama_profile.estimated_memory_gb
        if not stream:
            try:
                resp = await oc.chat(
                    ollama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, ollama_payload)
                return _anthropic_error_response(exc)
            return JSONResponse(
                ollama_chat_to_anthropic_response(resp, anthropic_model=anth_model)
            )

        async def body() -> AsyncIterator[bytes]:
            try:
                lines = oc.stream_chat(
                    ollama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
                async for chunk in ollama_stream_to_anthropic_sse(
                    lines, anthropic_model=anth_model
                ):
                    yield chunk
            except Exception as exc:
                _log_upstream_error(request, exc, ollama_payload)
                error_type = (
                    exc.error_type
                    if isinstance(exc, LocalTargetResourceError)
                    else "upstream_error"
                )
                err = {
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": _read_error_text(exc),
                    },
                }
                yield (
                    "event: error\ndata: "
                    + json.dumps(err, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")

        return StreamingResponse(body(), media_type="text/event-stream")

    if llama_target is not None:
        lc: LlamaCppClient = app.state.llama_cpp_clients.get(llama_target.name)
        if lc is None:
            raise HTTPException(
                status_code=503,
                detail=f"llama_cpp_target '{llama_target.name}' is not initialised",
            )
        llama_payload = anthropic_to_openai_chat_payload(
            payload,
            target_model=llama_target.resolve_model(real_model),
            default_max_tokens=settings.default_max_tokens,
        )
        _apply_llama_cpp_thinking_config(settings, anth_model, payload, llama_payload)
        profile = settings.profile_for(anth_model)
        estimated_vram_gb = profile.estimated_vram_gb
        estimated_memory_gb = profile.estimated_memory_gb
        if not stream:
            try:
                resp = await lc.chat(
                    llama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, llama_payload)
                return _anthropic_error_response(exc)
            return JSONResponse(
                openai_chat_to_anthropic_response(
                    resp,
                    anthropic_model=anth_model,
                    show_thinking=profile.show_thinking,
                )
            )

        async def body_llama_anthropic() -> AsyncIterator[bytes]:
            try:
                lines = lc.stream_chat(
                    llama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
                async for chunk in openai_stream_to_anthropic_sse(
                    lines,
                    anthropic_model=anth_model,
                    show_thinking=profile.show_thinking,
                ):
                    yield chunk
            except Exception as exc:
                _log_upstream_error(request, exc, llama_payload)
                error_type = (
                    exc.error_type
                    if isinstance(exc, LocalTargetResourceError)
                    else "upstream_error"
                )
                err = {
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": _read_error_text(exc),
                    },
                }
                yield (
                    "event: error\ndata: "
                    + json.dumps(err, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")

        return StreamingResponse(body_llama_anthropic(), media_type="text/event-stream")

    if generic_openai_target is not None:
        loc: GenericOpenAIClient = app.state.generic_openai_clients.get(
            generic_openai_target.name
        )
        if loc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"generic_openai_target '{generic_openai_target.name}' is not "
                    "initialised"
                ),
            )
        local_payload = anthropic_to_openai_chat_payload(
            payload,
            target_model=generic_openai_target.resolve_model(real_model),
            default_max_tokens=settings.default_max_tokens,
        )
        profile = settings.profile_for(anth_model)
        estimated_vram_gb = profile.estimated_vram_gb
        estimated_memory_gb = profile.estimated_memory_gb
        if not stream:
            try:
                resp = await loc.chat(
                    local_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, local_payload)
                return _anthropic_error_response(exc)
            return JSONResponse(
                openai_chat_to_anthropic_response(
                    resp,
                    anthropic_model=anth_model,
                    show_thinking=profile.show_thinking,
                )
            )

        async def body_generic_openai_anthropic() -> AsyncIterator[bytes]:
            try:
                lines = loc.stream_chat(
                    local_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                )
                async for chunk in openai_stream_to_anthropic_sse(
                    lines,
                    anthropic_model=anth_model,
                    show_thinking=profile.show_thinking,
                ):
                    yield chunk
            except Exception as exc:
                _log_upstream_error(request, exc, local_payload)
                error_type = (
                    exc.error_type
                    if isinstance(exc, LocalTargetResourceError)
                    else "upstream_error"
                )
                err = {
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": _read_error_text(exc),
                    },
                }
                yield (
                    "event: error\ndata: "
                    + json.dumps(err, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")

        return StreamingResponse(
            body_generic_openai_anthropic(), media_type="text/event-stream"
        )

    if openai_up_backend is not None:
        oc_remote = _backend_client(app, openai_up_backend)
        upstream_model = openai_up_backend.source.resolve_model(real_model)
        profile = settings.profile_for(anth_model)
        if not stream:
            try:
                anthropic_resp = await _openai_upstream_messages(
                    oc_remote,
                    payload,
                    anthropic_model=anth_model,
                    target_model=upstream_model,
                    default_max_tokens=settings.default_max_tokens,
                    show_thinking=profile.show_thinking,
                )
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, payload)
                return _anthropic_error_response(exc)
            return JSONResponse(anthropic_resp)

        async def body_openai_anthropic() -> AsyncIterator[bytes]:
            from .reverse_converters import _sse  # type: ignore

            try:
                async for event_type, data in _openai_upstream_stream_messages(
                    oc_remote,
                    payload,
                    anthropic_model=anth_model,
                    target_model=upstream_model,
                    default_max_tokens=settings.default_max_tokens,
                    show_thinking=profile.show_thinking,
                ):
                    yield _sse(event_type, data)
            except Exception as exc:
                _log_upstream_error(request, exc, payload)
                err = {
                    "type": "error",
                    "error": {
                        "type": "upstream_error",
                        "message": _read_error_text(exc),
                    },
                }
                yield (
                    "event: error\ndata: "
                    + json.dumps(err, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")

        return StreamingResponse(body_openai_anthropic(), media_type="text/event-stream")

    # Fallback: Anthropic upstream pass-through.
    if anthropic_backend is None:
        raise HTTPException(
            status_code=500,
            detail=f"unhandled backend {backend.name!r} ({backend.protocol}/{backend.kind})",
        )
    upstream_payload = dict(payload)
    upstream_model = anthropic_backend.source.resolve_model(real_model)
    upstream_payload["model"] = upstream_model
    upstream_payload["stream"] = stream
    client = _backend_client(app, anthropic_backend)
    if not stream:
        try:
            data = await client.messages(upstream_payload)
        except httpx.HTTPError as exc:
            _log_upstream_error(request, exc, upstream_payload)
            return _upstream_error(exc)
        return JSONResponse(data)

    async def body_passthrough() -> AsyncIterator[bytes]:
        state = AnthropicStreamState()
        try:
            async for event_type, data in client.stream_messages(upstream_payload):
                state.consume(event_type, data)
                yield (
                    f"event: {event_type}\ndata: "
                    + json.dumps(data, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")
                if state.terminal:
                    return
            raise UpstreamProtocolError("Upstream stream ended before completion")
        except Exception as exc:
            _log_upstream_error(request, exc, upstream_payload)
            err = {
                "type": "error",
                "error": {
                    "type": "upstream_error",
                    "message": _read_error_text(exc),
                },
            }
            yield (
                "event: error\ndata: "
                + json.dumps(err, ensure_ascii=False)
                + "\n\n"
            ).encode("utf-8")

    return StreamingResponse(body_passthrough(), media_type="text/event-stream")
