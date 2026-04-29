"""FastAPI app exposing an Ollama-compatible interface."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .anthropic_client import AnthropicClient
from .config import Settings, estimate_tokens_from_anthropic_payload, get_settings
from .converters import (
    AnthropicStreamTranslator,
    OpenAIChatStreamTranslator,
    anthropic_to_ollama_chat,
    anthropic_to_ollama_generate,
    anthropic_to_openai_chat,
    ollama_chat_to_anthropic,
    ollama_generate_to_anthropic,
    openai_chat_to_anthropic,
)

logger = logging.getLogger("fake_ollama")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_client = False
        if not getattr(app.state, "client", None):
            app.state.client = AnthropicClient(
                settings.upstream_url,
                settings.anthropic_auth_token,
                timeout=settings.timeout_seconds,
                trust_env=settings.use_system_proxy,
            )
            owns_client = True
        try:
            yield
        finally:
            if owns_client:
                client: AnthropicClient | None = getattr(app.state, "client", None)
                if client is not None:
                    await client.aclose()

    app = FastAPI(title="fake-ollama", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    @app.get("/")
    async def root() -> str:
        return "Ollama is running"

    @app.head("/")
    async def root_head() -> str:
        return ""

    @app.get("/api/version")
    async def version() -> Dict[str, str]:
        settings: Settings = app.state.settings
        return {"version": settings.advertised_version}

    @app.get("/api/tags")
    async def tags() -> Dict[str, Any]:
        settings: Settings = app.state.settings
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        models = []
        for name in settings.models:
            profile = settings.profile_for(name)
            models.append(
                {
                    "name": name,
                    "model": name,
                    "modified_at": now,
                    "size": 0,
                    "digest": "sha256:" + "0" * 64,
                    "details": {
                        "parent_model": "",
                        "format": "anthropic-proxy",
                        "family": "claude",
                        "families": ["claude"],
                        "parameter_size": "unknown",
                        "quantization_level": "none",
                    },
                    "capabilities": list(profile.capabilities),
                    "context_length": profile.context_length,
                }
            )
        return {"models": models}

    @app.get("/api/ps")
    async def ps() -> Dict[str, Any]:
        return {"models": []}

    @app.post("/api/show")
    async def show(payload: Dict[str, Any]) -> Dict[str, Any]:
        settings: Settings = app.state.settings
        name = payload.get("name") or payload.get("model") or ""
        profile = settings.profile_for(name)
        # Some clients (e.g. the GitHub Copilot VS Code extension's Ollama
        # integration) silently drop models whose /api/show response does not
        # advertise the capabilities they need ("completion" for chat,
        # "tools" for tool-calling, "vision" for image input). We surface
        # the per-model profile here so users can configure exactly what each
        # model claims to support.
        capabilities = list(profile.capabilities)
        ctx_len = profile.context_length
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return {
            "license": "proprietary (proxied)",
            "modelfile": f"# Proxied via fake-ollama -> {name}",
            "parameters": f"num_ctx                        {ctx_len}",
            "template": "{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "anthropic-proxy",
                "family": "claude",
                "families": ["claude"],
                "parameter_size": "unknown",
                "quantization_level": "none",
            },
            "model_info": {
                "general.architecture": "claude",
                "general.basename": name,
                "general.parameter_count": 0,
                "general.context_length": ctx_len,
                "claude.context_length": ctx_len,
            },
            "capabilities": capabilities,
            "context_length": ctx_len,
            "modified_at": now,
        }

    @app.post("/api/chat")
    async def chat(request: Request) -> Any:
        return await _handle(request, mode="chat")

    @app.post("/api/generate")
    async def generate(request: Request) -> Any:
        return await _handle(request, mode="generate")

    @app.post("/api/embeddings")
    @app.post("/api/embed")
    async def embeddings(payload: Dict[str, Any]) -> JSONResponse:
        # Anthropic does not provide embeddings; expose a clear error.
        raise HTTPException(
            status_code=501,
            detail="Embeddings are not supported by the upstream Anthropic API.",
        )

    # ---- OpenAI-compatible endpoints (Ollama also implements these) -----

    @app.get("/v1/models")
    async def openai_models() -> Dict[str, Any]:
        settings: Settings = app.state.settings
        now = int(datetime.now(timezone.utc).timestamp())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now,
                    "owned_by": "fake-ollama",
                    "context_length": settings.profile_for(name).context_length,
                    "capabilities": list(settings.profile_for(name).capabilities),
                }
                for name in settings.models
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Any:
        return await _handle_openai_chat(request)

    @app.post("/v1/embeddings")
    async def openai_embeddings(payload: Dict[str, Any]) -> JSONResponse:
        raise HTTPException(
            status_code=501,
            detail="Embeddings are not supported by the upstream Anthropic API.",
        )


def _read_error_text(exc: httpx.HTTPStatusError) -> str:
    try:
        return exc.response.text
    except Exception:  # pragma: no cover
        return str(exc)


def _log_upstream_error(
    exc: httpx.HTTPStatusError, upstream_payload: Dict[str, Any]
) -> None:
    """Log enough context to debug 4xx/5xx responses from the upstream."""
    body = _read_error_text(exc)
    # Avoid dumping huge prompts; keep the diagnostic relevant fields only.
    summary = {
        "model": upstream_payload.get("model"),
        "stream": upstream_payload.get("stream"),
        "max_tokens": upstream_payload.get("max_tokens"),
        "thinking": upstream_payload.get("thinking"),
        "tool_choice": upstream_payload.get("tool_choice"),
        "n_messages": len(upstream_payload.get("messages") or []),
        "n_tools": len(upstream_payload.get("tools") or []),
        "has_system": bool(upstream_payload.get("system")),
        "temperature": upstream_payload.get("temperature"),
        "top_p": upstream_payload.get("top_p"),
        "top_k": upstream_payload.get("top_k"),
    }
    logger.warning(
        "upstream %s: %s | request=%s",
        exc.response.status_code,
        body,
        json.dumps(summary, ensure_ascii=False),
    )


def _upstream_error(exc: httpx.HTTPStatusError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.response.status_code,
        content={"error": _read_error_text(exc)},
    )


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
    # Prefer per-model max_output_tokens default when client didn't pin one.
    if profile.max_output_tokens:
        cur = int(upstream_payload.get("max_tokens") or 0)
        if cur <= 0 or cur == settings.default_max_tokens:
            upstream_payload["max_tokens"] = profile.max_output_tokens
    # Cap any over-large request to model's max_output_tokens.
    if profile.max_output_tokens:
        upstream_payload["max_tokens"] = min(
            int(upstream_payload.get("max_tokens") or profile.max_output_tokens),
            profile.max_output_tokens,
        )

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

    Honours the client's explicit ``thinking`` field if already present so
    the user can override per-request. Only acts when the profile mode is
    `enabled` or `disabled`; `auto` leaves the payload unchanged.
    """
    if "thinking" in upstream_payload:
        return
    profile = settings.profile_for(ollama_model)
    mode = profile.thinking_mode
    if mode == "enabled":
        upstream_payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": profile.thinking_budget_tokens,
        }
    elif mode == "disabled":
        upstream_payload["thinking"] = {"type": "disabled"}


async def _handle(request: Request, *, mode: str) -> Any:
    app = request.app
    settings: Settings = app.state.settings
    client: AnthropicClient = app.state.client

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    ollama_model = payload.get("model") or (settings.models[0] if settings.models else "")
    if not ollama_model:
        raise HTTPException(status_code=400, detail="missing 'model'")
    upstream_model = settings.resolve_model(ollama_model)

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
        except httpx.HTTPStatusError as exc:
            _log_upstream_error(exc, upstream_payload)
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
        except httpx.HTTPStatusError as exc:
            _log_upstream_error(exc, upstream_payload)
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
    client: AnthropicClient = app.state.client

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    openai_model = payload.get("model") or (settings.models[0] if settings.models else "")
    if not openai_model:
        raise HTTPException(status_code=400, detail="missing 'model'")
    upstream_model = settings.resolve_model(openai_model)

    upstream_payload = openai_chat_to_anthropic(
        payload,
        upstream_model=upstream_model,
        default_max_tokens=settings.default_max_tokens,
    )
    stream = bool(payload.get("stream", False))
    upstream_payload["stream"] = stream
    _apply_thinking_config(settings, openai_model, upstream_payload)
    _enforce_limits(settings, openai_model, upstream_payload)

    profile = settings.profile_for(openai_model)

    if not stream:
        try:
            data = await client.messages(upstream_payload)
        except httpx.HTTPStatusError as exc:
            _log_upstream_error(exc, upstream_payload)
            return _upstream_error(exc)
        return JSONResponse(
            anthropic_to_openai_chat(
                data, openai_model=openai_model, show_thinking=profile.show_thinking
            )
        )

    async def body() -> AsyncIterator[bytes]:
        translator = OpenAIChatStreamTranslator(
            openai_model, show_thinking=profile.show_thinking
        )
        try:
            async for event_type, data in client.stream_messages(upstream_payload):
                for frame in translator.feed_event(event_type, data):
                    yield (
                        "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
                    ).encode("utf-8")
        except httpx.HTTPStatusError as exc:
            _log_upstream_error(exc, upstream_payload)
            err_frame = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": openai_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"[upstream error: {_read_error_text(exc)}]"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield (
                "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
            ).encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")



