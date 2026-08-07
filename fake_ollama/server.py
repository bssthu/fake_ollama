"""FastAPI app exposing an Ollama-compatible interface."""

from __future__ import annotations

import json
import logging
import asyncio
import time
import base64
import mimetypes
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from . import __version__
from .anthropic_client import AnthropicClient
from .comfyui_client import ComfyUIClient
from .config import (
    FORWARDED_BY_HEADER,
    INSTANCE_ID,
    ComfyUITarget,
    H3ContextIRProfile,
    H3ContextIRProvider,
    LlamaCppTarget,
    GenericOpenAITarget,
    Settings,
    estimate_tokens_from_anthropic_payload,
    get_settings,
    outbound_cycle_headers,
    parse_forwarded_chain,
    reset_inbound_forwarded_chain,
    set_inbound_forwarded_chain,
)
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
from .llama_cpp_client import LlamaCppClient
from .media_operations import describe_comfyui_operation
from .h3_context_ir import (
    SYSTEM_PROMPT as H3_CONTEXT_IR_SYSTEM_PROMPT,
    build_planning_request,
    build_repair_request,
    fallback_plan,
    is_structured_base_prompt,
    parse_and_validate_plan,
    render_base_prompt,
    resolve_base_mode,
)
from .generic_openai_client import GenericOpenAIClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .request_data_log import RequestDataLogMiddleware
from .vram import LocalTargetResourceError, MemoryCoordinator, VramCoordinator
from .dashboard import DashboardState, RequestMetrics, run_runtime_monitor
from .reverse_converters import (
    anthropic_to_ollama_chat as anthropic_to_ollama_chat_payload,
    anthropic_to_openai_chat as anthropic_to_openai_chat_payload,
    ollama_chat_to_anthropic as ollama_chat_to_anthropic_response,
    ollama_stream_to_anthropic_events,
    ollama_stream_to_anthropic_sse,
    openai_chat_to_anthropic as openai_chat_to_anthropic_response,
    openai_stream_to_anthropic_sse,
)

logger = logging.getLogger("fake_ollama")


EXTERNAL_PLANNER_TOKEN_HEADER = "x-playground-upstream-key"
_EXTERNAL_PLANNER_PROTOCOLS = ("openai", "anthropic")


@dataclass(frozen=True)
class ExternalPlannerConnection:
    """Validated, request-scoped credentials for a Playground Planner."""

    protocol: str
    base_url: str
    auth_token: str
    model: str
    modalities: tuple[str, ...]

    @property
    def accepts_images(self) -> bool:
        return "image" in self.modalities


class ForwardedCycleMiddleware:
    """Reject inbound requests that already passed through this process.

    On every ``/api/*`` or ``/v1/*`` inbound request:

    * Parse the ``x-fake-ollama-forwarded-by`` header into a chain of
      instance ids. If our own :data:`fake_ollama.config.INSTANCE_ID`
      is already present the request has looped back to us → respond
      508 ``Loop Detected``.
    * Otherwise stash the chain in a ContextVar so each outbound client
      can append our id to it when it stamps the same header on the
      onward request.

    Other paths (``/admin/*``, ``/dashboard/*``, ``/``) bypass the
    middleware entirely so internal tooling can still talk to us.
    """

    _MODEL_PREFIXES = ("/api/", "/v1/")

    def __init__(self, app):  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if not any(path.startswith(p) for p in self._MODEL_PREFIXES):
            await self.app(scope, receive, send)
            return
        chain: tuple[str, ...] = ()
        for raw_name, raw_value in scope.get("headers") or ():
            if raw_name.decode("latin-1").lower() == FORWARDED_BY_HEADER:
                chain = parse_forwarded_chain(raw_value.decode("latin-1"))
                break
        if INSTANCE_ID in chain:
            body = json.dumps(
                {
                    "error": (
                        "loop detected: request already passed through this "
                        "fake_ollama instance"
                    ),
                    "instance_id": INSTANCE_ID,
                    "chain": list(chain),
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 508,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (
                            FORWARDED_BY_HEADER.encode("latin-1"),
                            ",".join(chain).encode("latin-1"),
                        ),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        token = set_inbound_forwarded_chain(chain)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_inbound_forwarded_chain(token)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_names: list[str] = []
        owned_target_names: list[str] = []
        owned_llama_cpp_names: list[str] = []
        owned_generic_openai_names: list[str] = []
        owned_openai_names: list[str] = []
        owned_comfyui_names: list[str] = []
        if not getattr(app.state, "clients", None):
            app.state.clients = {}
        if not getattr(app.state, "ollama_clients", None):
            app.state.ollama_clients = {}
        if not getattr(app.state, "llama_cpp_clients", None):
            app.state.llama_cpp_clients = {}
        if not getattr(app.state, "generic_openai_clients", None):
            app.state.generic_openai_clients = {}
        if not getattr(app.state, "openai_clients", None):
            app.state.openai_clients = {}
        if not getattr(app.state, "comfyui_clients", None):
            app.state.comfyui_clients = {}
        if not getattr(app.state, "vram_coordinator", None):
            app.state.vram_coordinator = VramCoordinator()
        if not getattr(app.state, "memory_coordinator", None):
            app.state.memory_coordinator = MemoryCoordinator()
        if not getattr(app.state, "dashboard_state", None):
            app.state.dashboard_state = DashboardState()
        if not getattr(app.state, "request_metrics", None):
            app.state.request_metrics = RequestMetrics()
        for up in app.state.settings.anthropic_upstreams:
            if up.name in app.state.clients:
                continue
            app.state.clients[up.name] = AnthropicClient(
                up.base_url,
                up.auth_token,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
            )
            owned_names.append(up.name)
        for up in app.state.settings.openai_upstreams:
            if up.name in app.state.openai_clients:
                continue
            app.state.openai_clients[up.name] = OpenAIClient(
                up.base_url,
                auth_token=up.auth_token,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
                upstream_name=up.name,
            )
            owned_openai_names.append(up.name)
        for tgt in app.state.settings.ollama_targets:
            if tgt.name in app.state.ollama_clients:
                continue
            app.state.ollama_clients[tgt.name] = OllamaClient(
                tgt.base_url,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
                auto_start=tgt.auto_start,
                start_command=tgt.start_command,
                stop_command=tgt.stop_command,
                idle_timeout_seconds=tgt.idle_timeout_seconds,
                startup_timeout_seconds=tgt.startup_timeout_seconds,
                health_path=tgt.health_path,
                cwd=tgt.cwd,
                target_name=tgt.name,
                vram_coordinator=app.state.vram_coordinator,
                memory_coordinator=app.state.memory_coordinator,
            )
            owned_target_names.append(tgt.name)
        for raw_tgt in app.state.settings.llama_cpp_targets:
            tgt = app.state.settings.effective_llama_cpp_target(raw_tgt)
            if tgt.name in app.state.llama_cpp_clients:
                continue
            app.state.llama_cpp_clients[tgt.name] = LlamaCppClient(
                tgt.base_url,
                auth_token=tgt.auth_token,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
                auto_start=tgt.auto_start,
                start_command=tgt.start_command,
                start_argv=tgt.synthesize_start_argv(),
                stop_command=tgt.stop_command,
                idle_timeout_seconds=tgt.idle_timeout_seconds,
                startup_timeout_seconds=tgt.startup_timeout_seconds,
                health_path=tgt.health_path,
                cwd=tgt.cwd,
                launch_env=tgt.effective_env(),
                target_name=tgt.name,
                vram_coordinator=app.state.vram_coordinator,
                memory_coordinator=app.state.memory_coordinator,
                max_concurrent_requests=tgt.effective_max_concurrent_requests,
                request_read_timeout_seconds=tgt.request_read_timeout_seconds,
            )
            owned_llama_cpp_names.append(tgt.name)
        for tgt in app.state.settings.generic_openai_targets:
            if tgt.name in app.state.generic_openai_clients:
                continue
            app.state.generic_openai_clients[tgt.name] = GenericOpenAIClient(
                tgt.base_url,
                auth_token=tgt.auth_token,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
                auto_start=tgt.auto_start,
                start_command=tgt.start_command,
                stop_command=tgt.stop_command,
                idle_timeout_seconds=tgt.idle_timeout_seconds,
                startup_timeout_seconds=tgt.startup_timeout_seconds,
                health_path=tgt.health_path,
                cwd=tgt.cwd,
                target_name=tgt.name,
                vram_coordinator=app.state.vram_coordinator,
                memory_coordinator=app.state.memory_coordinator,
                max_concurrent_requests=tgt.max_concurrent_requests,
                request_read_timeout_seconds=tgt.request_read_timeout_seconds,
            )
            owned_generic_openai_names.append(tgt.name)
        for tgt in app.state.settings.comfyui_targets:
            if tgt.name in app.state.comfyui_clients:
                continue
            app.state.comfyui_clients[tgt.name] = ComfyUIClient(
                tgt.base_url,
                auth_token=tgt.auth_token,
                timeout=app.state.settings.timeout_seconds,
                trust_env=app.state.settings.use_system_proxy,
                auto_start=tgt.auto_start,
                start_command=tgt.start_command,
                stop_command=tgt.stop_command,
                idle_timeout_seconds=tgt.idle_timeout_seconds,
                startup_timeout_seconds=tgt.startup_timeout_seconds,
                health_path=tgt.health_path,
                cwd=tgt.cwd,
                target_name=tgt.name,
                workflow_config=tgt.workflow_config(),
                vram_coordinator=app.state.vram_coordinator,
                memory_coordinator=app.state.memory_coordinator,
            )
            owned_comfyui_names.append(tgt.name)
        _sync_local_target_idle_monitor(app)
        _sync_runtime_monitor(app)
        try:
            yield
        finally:
            runtime_monitor = getattr(app.state, "runtime_monitor", None)
            if runtime_monitor is not None:
                runtime_monitor.cancel()
                try:
                    await runtime_monitor
                except asyncio.CancelledError:
                    pass
                app.state.runtime_monitor = None
            idle_monitor = getattr(app.state, "local_target_idle_monitor", None)
            if idle_monitor is not None:
                idle_monitor.cancel()
                try:
                    await idle_monitor
                except asyncio.CancelledError:
                    pass
                app.state.local_target_idle_monitor = None
            for name in owned_names:
                client = app.state.clients.pop(name, None)
                if client is not None:
                    await client.aclose()
            for name in owned_target_names:
                oc = app.state.ollama_clients.pop(name, None)
                if oc is not None:
                    await oc.aclose()
            for name in owned_llama_cpp_names:
                lc = app.state.llama_cpp_clients.pop(name, None)
                if lc is not None:
                    await lc.aclose()
            for name in owned_generic_openai_names:
                lc2 = app.state.generic_openai_clients.pop(name, None)
                if lc2 is not None:
                    await lc2.aclose()
            for name in owned_openai_names:
                oc2 = app.state.openai_clients.pop(name, None)
                if oc2 is not None:
                    await oc2.aclose()
            for name in owned_comfyui_names:
                cc = app.state.comfyui_clients.pop(name, None)
                if cc is not None:
                    await cc.aclose()

    app = FastAPI(title="fake-ollama", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.shutdown_requested = False
    app.state.vram_coordinator = VramCoordinator()
    app.state.memory_coordinator = MemoryCoordinator()
    app.state.dashboard_state = DashboardState()
    app.state.request_metrics = RequestMetrics()
    app.state.ensure_local_target_idle_monitor = _sync_local_target_idle_monitor
    app.state.ensure_runtime_monitor = _sync_runtime_monitor
    _install_port_router(app)
    app.add_middleware(RequestDataLogMiddleware)
    app.add_middleware(ForwardedCycleMiddleware)
    _register_routes(app)
    return app


def request_shutdown(app: FastAPI) -> None:
    """Mark the app as shutting down and disable local target auto-starts.

    Also schedules ``stop_if_owned`` for every owned local target client so
    in-flight requests against a still-loading llama.cpp / ollama child get
    aborted (taskkill /T /F on Windows). Without this the upstream HTTP
    connection from fake-ollama to the local server stays open until the
    huge model finishes loading, which prevents uvicorn from exiting after
    CTRL+C ("Waiting for connections to close").
    """
    if getattr(app.state, "shutdown_requested", False):
        return
    app.state.shutdown_requested = True
    owned_clients: list[Any] = []
    for client_group in (
        getattr(app.state, "ollama_clients", {}),
        getattr(app.state, "llama_cpp_clients", {}),
        getattr(app.state, "generic_openai_clients", {}),
        getattr(app.state, "comfyui_clients", {}),
    ):
        for client in list(client_group.values()):
            begin_shutdown = getattr(client, "begin_shutdown", None)
            if callable(begin_shutdown):
                begin_shutdown()
            if getattr(client, "stop_if_owned", None) is not None:
                owned_clients.append(client)

    if not owned_clients:
        return

    async def _stop_all() -> None:
        for client in owned_clients:
            try:
                await client.stop_if_owned()
            except Exception:
                logger.exception(
                    "failed to stop owned local target during shutdown"
                )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.create_task(_stop_all())
    else:
        # No running loop (e.g. uvicorn already finished serving). Run the
        # cleanup synchronously so the caller still kills the children.
        try:
            asyncio.run(_stop_all())
        except RuntimeError:
            pass


def _openai_client_for(app: FastAPI, upstream_name: str) -> OpenAIClient:
    """Resolve the configured :class:`OpenAIClient` for an upstream name."""
    clients: Dict[str, OpenAIClient] = getattr(app.state, "openai_clients", {})
    client = clients.get(upstream_name)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai_upstream '{upstream_name}' is not initialised",
        )
    return client


def _interface_for(request: Request):
    """Return the ``OllamaInterface``/``ApiInterface`` that owns this request.

    Looks up the interface by local listener port. Returns ``None`` when
    the request landed on the admin or dashboard listener (or any other
    listener that does not handle model traffic).
    """
    settings: Settings = request.app.state.settings
    local_port = _local_port(request)
    if local_port is not None:
        if (
            settings.playground_listener_enabled
            and local_port == settings.playground_port
        ):
            return _playground_interface_for(request, settings)
        for it in settings.ollama_interfaces:
            if it.port == local_port:
                return it
        for it in settings.api_interfaces:
            if it.port == local_port:
                return it
    # No matching listener (TestClient / ASGI direct call, or unknown port):
    # best effort fallback so unit tests can hit /api/* and /v1/* without
    # binding real sockets.
    if settings.ollama_interfaces:
        return settings.ollama_interfaces[0]
    if settings.api_interfaces:
        return settings.api_interfaces[0]
    return None


def _dispatch(
    request: Request, settings: Settings, requested_model: str
) -> tuple["Backend", str]:  # type: ignore[name-defined]
    """Resolve a client-requested public model id for ``request``.

    * Identifies which interface owns the request (by local port).
    * Performs the per-interface auth check (when access_tokens is set).
    * Returns ``(backend, real_model)`` where ``real_model`` is the
      source-level display name (alias_or_name); callers further wrap
      it via ``backend.source.resolve_model(real_model)`` to get the
      wire-side id.
    * Raises ``HTTPException`` with 400/401/404 on bad input.
    """
    iface = _interface_for(request)
    if iface is None:
        raise HTTPException(
            status_code=404,
            detail="this listener does not serve model traffic",
        )
    if iface.auth_required:
        token = _bearer_or_api_key(request)
        if not iface.is_valid_token(token):
            raise HTTPException(
                status_code=401,
                detail=(
                    "missing or invalid api token (send via x-api-key "
                    "or Authorization: Bearer header)"
                ),
            )
    try:
        backend, model = settings.resolve_request(
            requested_model, interface_name=iface.name
        )
    except ValueError as exc:
        msg = str(exc)
        if (
            "is not exposed" in msg
            or "unknown target" in msg
            or "does not serve" in msg
        ):
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc
    metrics: Optional[RequestMetrics] = getattr(
        request.app.state, "request_metrics", None
    )
    rid = getattr(request.state, "request_metric_id", None)
    if metrics is not None and rid is not None:
        metrics.set_target(rid, backend.name)
    return backend, model


def _backend_client(app: FastAPI, backend) -> Any:  # type: ignore[no-untyped-def]
    """Return the protocol-appropriate client for ``backend``.

    Knows about the four (protocol, kind) pairs:
    * ``anthropic`` + ``remote`` → :class:`AnthropicClient`
    * ``openai`` + ``remote`` → :class:`OpenAIClient`
    * ``ollama`` + ``local`` → :class:`OllamaClient`
    * ``openai`` + ``local`` → :class:`LlamaCppClient`
    """
    if backend.protocol == "anthropic":
        client = app.state.clients.get(backend.name)
        if client is None:
            # Fall back to the first available client, mirroring the
            # legacy ``_client_for`` behaviour used by some tests.
            clients = app.state.clients
            if clients:
                return next(iter(clients.values()))
            raise HTTPException(
                status_code=503,
                detail=f"anthropic upstream '{backend.name}' is not initialised",
            )
        return client
    if backend.protocol == "openai" and backend.kind == "remote":
        return _openai_client_for(app, backend.name)
    if backend.protocol == "ollama":
        client = app.state.ollama_clients.get(backend.name)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"ollama_target '{backend.name}' is not initialised",
            )
        return client
    if backend.protocol == "openai" and backend.kind == "local":
        if isinstance(backend.source, GenericOpenAITarget):
            client = app.state.generic_openai_clients.get(backend.name)
            label = "generic_openai_target"
        else:
            client = app.state.llama_cpp_clients.get(backend.name)
            label = "llama_cpp_target"
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"{label} '{backend.name}' is not initialised",
            )
        return client
    if backend.protocol == "comfyui":
        client = app.state.comfyui_clients.get(backend.name)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"comfyui_target '{backend.name}' is not initialised",
            )
        return client
    raise HTTPException(
        status_code=500,
        detail=f"no client wiring for backend {backend.name!r} ({backend.protocol}/{backend.kind})",
    )


def _profile_for_public_id(settings: Settings, iface: Any, public_id: str) -> Any:
    entry = iface.exposure_for_public_id(public_id) if iface is not None else None
    if entry is not None:
        return settings.profile_for(f"{entry.model}@{entry.target}")
    return settings.profile_for(public_id)


def _openai_model_entry(public_id: str, created: int) -> Dict[str, Any]:
    """Return the exact OpenAI Model object shape."""

    return {
        "id": public_id,
        "object": "model",
        "created": created,
        "owned_by": "fake-ollama",
    }


def _anthropic_model_capabilities(profile: Any) -> Dict[str, Any]:
    """Map the local profile conservatively to Anthropic's capability schema."""

    capability_set = set(profile.capabilities)

    def support(enabled: bool = False) -> Dict[str, bool]:
        return {"supported": enabled}

    thinking_enabled = profile.thinking_mode == "enabled"
    return {
        "batch": support(),
        "citations": support(),
        "code_execution": support(),
        "context_management": {
            "clear_thinking_20251015": support(),
            "clear_tool_uses_20250919": support(),
            "compact_20260112": support(),
            "supported": False,
        },
        "effort": {
            "high": support(),
            "low": support(),
            "max": support(),
            "medium": support(),
            "supported": False,
            "xhigh": support(),
        },
        "image_input": support("vision" in capability_set),
        "pdf_input": support(),
        "structured_outputs": support(),
        "thinking": {
            "supported": thinking_enabled,
            "types": {
                "adaptive": support(),
                "enabled": support(thinking_enabled),
            },
        },
    }


def _anthropic_model_entry(
    settings: Settings, iface: Any, public_id: str
) -> Dict[str, Any]:
    """Return the Anthropic ModelInfo shape for one exposed model."""

    profile = _profile_for_public_id(settings, iface, public_id)
    return {
        "id": public_id,
        "capabilities": _anthropic_model_capabilities(profile),
        "created_at": "1970-01-01T00:00:00Z",
        "display_name": public_id,
        "max_input_tokens": profile.context_length,
        "max_tokens": profile.max_output_tokens or profile.context_length,
        "type": "model",
    }


def _playground_model_entry(
    settings: Settings, iface: Any, public_id: str
) -> Dict[str, Any]:
    """Describe one model for fake-ollama's versioned Playground API."""

    profile = _profile_for_public_id(settings, iface, public_id)
    exposure = iface.exposure_for_public_id(public_id) if iface is not None else None
    source = settings.source_by_name(exposure.target) if exposure is not None else None
    capabilities = list(profile.capabilities)
    capability_set = set(capabilities)
    operations: List[Dict[str, Any]] = []
    if capability_set.intersection({"completion", "tools", "vision"}):
        operations.append(
            {
                "id": "chat",
                "endpoint": "/v1/chat/completions",
                "stream": True,
                "history_mode": "conversation",
                "accepts_images": "vision" in capability_set,
                "tool_calling": "tools" in capability_set,
            }
        )
    if "image_generation" in capability_set:
        operations.append(
            {
                "id": "image_generation",
                "endpoint": "/v1/images/generations",
                "stream": False,
                "history_mode": "single_turn",
                "accepts_images": False,
            }
        )
    if "image_edit" in capability_set:
        operations.append(
            {
                "id": "image_edit",
                "endpoint": "/v1/images/edits",
                "stream": False,
                "history_mode": "single_turn",
                "accepts_images": True,
                "requires_images": True,
                "multiple_images": True,
            }
        )
    if "video_generation" in capability_set:
        operations.append(
            {
                "id": "video_generation",
                "endpoint": "/v1/videos/generations",
                "stream": False,
                "history_mode": "single_turn",
                "accepts_images": True,
                "requires_images": False,
                "multiple_images": True,
            }
        )
    if "video_understanding" in capability_set:
        operations.append(
            {
                "id": "video_analysis",
                "endpoint": "/v1/chat/completions",
                "stream": True,
                "history_mode": "single_turn",
                "accepts_videos": True,
                "requires_videos": True,
                "multiple_videos": False,
                "limits": {
                    "max_videos": 1,
                    "max_video_bytes": 64 * 1024 * 1024,
                },
                "live_camera": {
                    "supported": True,
                    "capture_mode": "windowed_media_recorder",
                    "max_pending_segments": 1,
                },
                "parameters": [
                    {
                        "name": "segment_seconds",
                        "label": "时间窗口（秒）",
                        "type": "number",
                        "default": 8,
                        "min": 2,
                        "max": 60,
                        "step": 1,
                        "description": "逐段分析的视频窗口长度。",
                    },
                    {
                        "name": "frames_per_segment",
                        "label": "每段抽帧数",
                        "type": "integer",
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "description": "FFmpeg 在每个时间窗口内均匀抽取的帧数。",
                    },
                    {
                        "name": "max_segments",
                        "label": "最多分析段数",
                        "type": "integer",
                        "default": 12,
                        "min": 1,
                        "max": 120,
                        "step": 1,
                        "description": "长视频超过上限时沿完整时间轴均匀选段；增加段数只增加串行耗时。",
                    },
                    {
                        "name": "include_summary",
                        "label": "生成整体总结",
                        "type": "boolean",
                        "default": False,
                        "description": "逐段分析后再调用一次模型汇总，耗时会增加。",
                    },
                ],
            }
        )
    if isinstance(source, ComfyUITarget):
        for operation in operations:
            if operation["id"] not in {
                "image_generation", "image_edit", "video_generation"
            }:
                continue
            operation.update(describe_comfyui_operation(source, operation["id"]))
            if operation["id"] == "video_generation" and source.context_ir_profile:
                ir_profile = settings.h3_context_ir_profile_by_name(
                    source.context_ir_profile
                )
                if ir_profile is None or not ir_profile.enabled:
                    operation["configured"] = False
                    operation["context_ir_profile"] = source.context_ir_profile
                    continue
                provider_choices = [
                    {
                        "value": "auto",
                        "label": "Auto by input modality",
                        "group": "推荐 Planner",
                        "selection_kind": "auto",
                    }
                ]
                for provider in ir_profile.providers:
                    provider_choices.append(
                        _context_ir_provider_choice(
                            settings,
                            provider,
                            value=provider.name,
                            label=(
                                f"{provider.name} "
                                f"({'/'.join(provider.modalities)}; "
                                f"{provider.model}@{provider.target})"
                            ),
                            group="推荐 Planner",
                            selection_kind="recommended",
                        )
                    )
                provider_choices.extend(
                    _compatible_context_ir_provider_choices(
                        settings, iface, ir_profile
                    )
                )
                external_choice = _external_context_ir_provider_choice(ir_profile)
                if external_choice is not None:
                    provider_choices.append(external_choice)
                operation["context_ir_profile"] = ir_profile.name
                operation["planner_defaults"] = {
                    "text": ir_profile.default_text_provider,
                    "image": (
                        ir_profile.default_multimodal_provider
                        or ir_profile.default_text_provider
                    ),
                }
                if ir_profile.allow_external_api:
                    operation["external_planner_api"] = {
                        "models_endpoint": "/playground/api/external-models",
                        "protocols": list(_EXTERNAL_PLANNER_PROTOCOLS),
                    }
                operation.setdefault("parameters", []).extend(
                    [
                        {
                            "name": "prompt_mode",
                            "label": "Prompt enhancement",
                            "type": "select",
                            "default": source.context_ir_prompt_mode,
                            "choices": [
                                {"value": "raw", "label": "Raw / bypass"},
                                {
                                    "value": "auto",
                                    "label": "Auto (bypass structured prompts)",
                                },
                                {"value": "enhance", "label": "Always enhance"},
                            ],
                            "description": (
                                "Runs the configured H3 Context-IR-fake profile "
                                "before submitting the ComfyUI workflow."
                            ),
                        },
                        {
                            "name": "context_ir_provider",
                            "label": "Context-IR provider",
                            "type": "select",
                            "default": "auto",
                            "choices": provider_choices,
                            "advanced": True,
                        },
                        {
                            "name": "context_ir_mode",
                            "label": "H3 base mode",
                            "type": "select",
                            "default": "auto",
                            "choices": [
                                {"value": "auto", "label": "Auto by image count"},
                                {"value": "t2va", "label": "T2VA"},
                                {"value": "i2va", "label": "I2VA"},
                                {"value": "fl2va", "label": "FL2VA"},
                                {"value": "l2va", "label": "L2VA"},
                            ],
                            "advanced": True,
                        },
                    ]
                )
    return {
        "id": public_id,
        "context_length": profile.context_length,
        "max_output_tokens": profile.max_output_tokens,
        "estimated_vram_gb": profile.estimated_vram_gb,
        "estimated_memory_gb": profile.estimated_memory_gb,
        "capabilities": capabilities,
        "operations": operations,
    }


def _context_ir_provider_choice(
    settings: Settings,
    provider: H3ContextIRProvider,
    *,
    value: str,
    label: str,
    group: str,
    selection_kind: str,
) -> Dict[str, Any]:
    model_profile = settings.profile_for(
        f"{provider.model}@{provider.target}"
    )
    backend = settings.backend_by_name(provider.target)
    return {
        "value": value,
        "label": label,
        "group": group,
        "selection_kind": selection_kind,
        "model": provider.model,
        "target": provider.target,
        "modalities": list(provider.modalities),
        "backend_kind": backend.kind if backend is not None else None,
        "protocol": backend.protocol if backend is not None else None,
        "context_length": model_profile.context_length,
        "max_output_tokens": model_profile.max_output_tokens,
        "estimated_vram_gb": model_profile.estimated_vram_gb,
        "estimated_memory_gb": model_profile.estimated_memory_gb,
    }


def _external_context_ir_provider_choice(
    profile: H3ContextIRProfile,
) -> Optional[Dict[str, Any]]:
    if not profile.allow_external_api:
        return None
    return {
        "value": "external",
        "label": "临时第三方 API（URL + Token）",
        "group": "第三方 API",
        "selection_kind": "external",
        "model": None,
        "target": "request-scoped",
        "modalities": ["text"],
        "backend_kind": "remote",
        "protocol": None,
        "context_length": None,
        "max_output_tokens": profile.max_output_tokens,
        "estimated_vram_gb": None,
        "estimated_memory_gb": None,
    }


def _external_planner_protocol(value: Any) -> str:
    protocol = str(value or "openai").strip().lower()
    if protocol not in _EXTERNAL_PLANNER_PROTOCOLS:
        raise HTTPException(
            status_code=400,
            detail="external API protocol must be openai|anthropic",
        )
    return protocol


def _normalize_external_planner_base_url(value: Any) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        raise HTTPException(status_code=400, detail="external API base_url is required")
    if len(raw) > 2048:
        raise HTTPException(status_code=400, detail="external API base_url is too long")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="external API base_url must be an absolute http(s) URL",
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="external API credentials must use the token field, not URL userinfo",
        )
    if parsed.query or parsed.fragment:
        raise HTTPException(
            status_code=400,
            detail="external API base_url must not contain a query or fragment",
        )
    path = parsed.path.rstrip("/")
    lowered = path.lower()
    for suffix in ("/chat/completions", "/messages", "/models"):
        if lowered.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            lowered = path.lower()
            break
    if lowered.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _external_planner_token(request: Request) -> str:
    token = request.headers.get(EXTERNAL_PLANNER_TOKEN_HEADER, "").strip()
    if not token:
        raise HTTPException(
            status_code=400,
            detail="external API token is required in x-playground-upstream-key",
        )
    if len(token) > 16384:
        raise HTTPException(status_code=400, detail="external API token is too long")
    return token


def _external_planner_connection(
    request: Request,
    profile: H3ContextIRProfile,
    payload: Dict[str, Any],
    requested_provider: Any,
) -> Optional[ExternalPlannerConnection]:
    if str(requested_provider or "").strip() != "external":
        return None
    if not profile.allow_external_api:
        raise HTTPException(
            status_code=400,
            detail="this H3 Context-IR profile does not allow external APIs",
        )
    if _listener_name(request) != "playground":
        raise HTTPException(
            status_code=400,
            detail="request-scoped external APIs are only available on the Playground listener",
        )
    protocol = _external_planner_protocol(payload.get("external_api_protocol"))
    base_url = _normalize_external_planner_base_url(
        payload.get("external_api_base_url")
    )
    model = str(payload.get("external_api_model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="external API model is required")
    if len(model) > 512:
        raise HTTPException(status_code=400, detail="external API model is too long")
    raw_modalities = payload.get("external_api_modalities") or "text"
    if isinstance(raw_modalities, str):
        values = raw_modalities.replace(";", ",").split(",")
    elif isinstance(raw_modalities, list):
        values = raw_modalities
    else:
        values = [raw_modalities]
    modalities = ["text"]
    if any(str(value).strip().lower() == "image" for value in values):
        modalities.append("image")
    return ExternalPlannerConnection(
        protocol=protocol,
        base_url=base_url,
        auth_token=_external_planner_token(request),
        model=model,
        modalities=tuple(modalities),
    )


def _external_planner_headers(protocol: str, token: str) -> Dict[str, str]:
    headers = {"accept": "application/json"}
    if protocol == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    headers["authorization"] = f"Bearer {token}"
    headers["x-api-key"] = token
    headers.update(outbound_cycle_headers())
    return headers


@asynccontextmanager
async def _external_planner_http_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    injected = getattr(app.state, "external_planner_http_client", None)
    if injected is not None:
        yield injected
        return
    settings: Settings = app.state.settings
    async with httpx.AsyncClient(
        timeout=settings.timeout_seconds,
        trust_env=settings.use_system_proxy,
    ) as client:
        yield client


def _compatible_context_ir_provider_choices(
    settings: Settings,
    iface: Any,
    profile: H3ContextIRProfile,
) -> List[Dict[str, Any]]:
    """Return compatible chat models exposed to the authenticated interface."""

    if not profile.allow_compatible_models:
        return []
    recommended = {
        (provider.model, provider.target) for provider in profile.providers
    }
    choices: List[Dict[str, Any]] = []
    for exposure in iface.exposed_models:
        try:
            backend, display_model = settings.resolve_request(
                exposure.public_id, interface_name=iface.name
            )
        except ValueError:
            continue
        if backend.protocol not in ("anthropic", "openai", "ollama"):
            continue
        if (display_model, backend.name) in recommended:
            continue
        model_profile = settings.profile_for(
            f"{display_model}@{backend.name}"
        )
        capabilities = set(model_profile.capabilities)
        if "completion" not in capabilities:
            continue
        modalities = ["text"]
        if "vision" in capabilities:
            modalities.append("image")
        provider = H3ContextIRProvider(
            name=f"model:{exposure.public_id}",
            model=display_model,
            target=backend.name,
            modalities=modalities,
            # Plain JSON parsing is the portable choice for arbitrary
            # OpenAI-compatible APIs; recommended providers may opt in to
            # response_format=json_object explicitly in config.
            json_mode=False,
        )
        modality_label = "/".join(modalities)
        choices.append(
            _context_ir_provider_choice(
                settings,
                provider,
                value=provider.name,
                label=f"{exposure.public_id} ({modality_label})",
                group="自选兼容模型",
                selection_kind="compatible",
            )
        )
    return choices


def _playground_context_ir_entry(
    settings: Settings, iface: Any, profile: H3ContextIRProfile
) -> Dict[str, Any]:
    """Expose one orchestration profile as a Playground-only virtual model."""

    default_provider = profile.provider_by_name(
        profile.default_text_provider or profile.providers[0].name
    ) or profile.providers[0]
    model_profile = settings.profile_for(
        f"{default_provider.model}@{default_provider.target}"
    )
    provider_choices = [
        {
            "value": "auto",
            "label": "Auto by input modality",
            "group": "推荐 Planner",
            "selection_kind": "auto",
        }
    ]
    for provider in profile.providers:
        modalities = "/".join(provider.modalities)
        provider_choices.append(
            _context_ir_provider_choice(
                settings,
                provider,
                value=provider.name,
                label=(
                    f"{provider.name} ({modalities}; "
                    f"{provider.model}@{provider.target})"
                ),
                group="推荐 Planner",
                selection_kind="recommended",
            )
        )
    provider_choices.extend(
        _compatible_context_ir_provider_choices(settings, iface, profile)
    )
    external_choice = _external_context_ir_provider_choice(profile)
    if external_choice is not None:
        provider_choices.append(external_choice)
    operation = {
        "id": "h3_context_ir",
        "endpoint": "/v1/videos/context-ir",
        "stream": False,
        "history_mode": "single_turn",
        "accepts_images": True,
        "requires_images": False,
        "multiple_images": True,
        "limits": {"max_reference_images": 2},
        "planner_defaults": {
            "text": profile.default_text_provider,
            "image": (
                profile.default_multimodal_provider
                or profile.default_text_provider
            ),
        },
        "parameters": [
            {
                "name": "provider",
                "label": "Planner provider",
                "type": "select",
                "default": "auto",
                "choices": provider_choices,
                "description": (
                    "Auto uses the text default without images and the multimodal "
                    "default when reference images are attached. Recommended "
                    "providers are followed by compatible chat models exposed "
                    "on the selected interface."
                ),
            },
            {
                "name": "mode",
                "label": "H3 base mode",
                "type": "select",
                "default": "auto",
                "choices": [
                    {"value": "auto", "label": "Auto by image count"},
                    {"value": "t2va", "label": "T2VA (text)"},
                    {"value": "i2va", "label": "I2VA (first frame)"},
                    {"value": "fl2va", "label": "FL2VA (first + last)"},
                    {"value": "l2va", "label": "L2VA (last frame)"},
                ],
            },
            {
                "name": "duration_seconds",
                "label": "Duration (seconds)",
                "type": "number",
                "default": profile.default_duration_seconds,
                "min": 4,
                "max": 15,
                "step": 0.1,
            },
        ],
    }
    if profile.allow_external_api:
        operation["external_planner_api"] = {
            "models_endpoint": "/playground/api/external-models",
            "protocols": list(_EXTERNAL_PLANNER_PROTOCOLS),
        }
    return {
        "id": profile.public_model_id,
        "context_length": model_profile.context_length,
        "max_output_tokens": profile.max_output_tokens,
        "estimated_vram_gb": model_profile.estimated_vram_gb,
        "estimated_memory_gb": model_profile.estimated_memory_gb,
        "capabilities": ["h3_context_ir"],
        "operations": [operation],
    }


def _public_model_ids(iface: Any) -> List[str]:
    seen: Dict[str, None] = {}
    for public_id in iface.public_ids():
        seen.setdefault(public_id, None)
    return list(seen)


def _model_interface_for_request(request: Request) -> tuple[Settings, Any]:
    """Resolve and authenticate the interface used for model discovery."""

    settings: Settings = request.app.state.settings
    iface = _interface_for(request)
    if iface is None:
        raise HTTPException(status_code=404, detail="unknown listener")
    if iface.auth_required:
        token = _bearer_or_api_key(request)
        if not iface.is_valid_token(token):
            raise HTTPException(
                status_code=401,
                detail=(
                    "missing or invalid api token (send via x-api-key "
                    "or Authorization: Bearer header)"
                ),
            )
    return settings, iface


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


async def _local_target_idle_monitor(app: FastAPI) -> None:
    while True:
        try:
            await asyncio.sleep(5.0)
            client_groups = (
                getattr(app.state, "ollama_clients", {}),
                getattr(app.state, "llama_cpp_clients", {}),
                getattr(app.state, "generic_openai_clients", {}),
                getattr(app.state, "comfyui_clients", {}),
            )
            for clients in client_groups:
                for client in list(clients.values()):
                    stop_if_idle = getattr(client, "stop_if_idle", None)
                    if stop_if_idle is not None:
                        await stop_if_idle()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("local target idle lifecycle check failed")


def _sync_local_target_idle_monitor(app: FastAPI) -> None:
    if getattr(app.state, "shutdown_requested", False):
        task = getattr(app.state, "local_target_idle_monitor", None)
        if task is not None and not task.done():
            task.cancel()
        app.state.local_target_idle_monitor = None
        return

    client_groups = (
        getattr(app.state, "ollama_clients", {}),
        getattr(app.state, "llama_cpp_clients", {}),
        getattr(app.state, "generic_openai_clients", {}),
        getattr(app.state, "comfyui_clients", {}),
    )
    needs_monitor = any(
        getattr(c, "idle_timeout_seconds", None)
        for clients in client_groups
        for c in clients.values()
    )
    task = getattr(app.state, "local_target_idle_monitor", None)
    task_running = task is not None and not task.done()
    if needs_monitor and not task_running:
        app.state.local_target_idle_monitor = asyncio.create_task(
            _local_target_idle_monitor(app)
        )
    elif not needs_monitor and task_running:
        task.cancel()
        app.state.local_target_idle_monitor = None


def _sync_runtime_monitor(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    needs_monitor = (
        settings.dashboard_listener_enabled or settings.vram_low_free_reclaim_enabled
    )
    task = getattr(app.state, "runtime_monitor", None)
    task_running = task is not None and not task.done()
    if needs_monitor and not task_running:
        app.state.runtime_monitor = asyncio.create_task(run_runtime_monitor(app))
    elif not needs_monitor and task_running:
        task.cancel()
        app.state.runtime_monitor = None


def _bearer_or_api_key(request: Request) -> str:
    """Extract a token from ``x-api-key`` or ``Authorization: Bearer ...``."""
    tok = request.headers.get("x-api-key") or ""
    if tok:
        return tok.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _playground_interface_for(request: Request, settings: Settings):
    """Pick the model interface represented by a playground request.

    A matching token wins, which lets one playground port work with multiple
    configured API interfaces.  Without a token, prefer the first open API
    interface, then an open Ollama interface.  Falling back to the first
    configured interface preserves its normal 401 response for invalid keys.
    """
    candidates = [*settings.api_interfaces, *settings.ollama_interfaces]
    if not candidates:
        return None
    token = _bearer_or_api_key(request)
    if token:
        for iface in candidates:
            if iface.auth_required and iface.is_valid_token(token):
                return iface
    for iface in candidates:
        if not iface.auth_required:
            return iface
    return candidates[0]


# Path prefixes that are only served by the admin listener when admin_port is set.
_ADMIN_ONLY_PATH_PREFIXES = ("/admin",)
# Path prefixes that are only served by the dashboard listener.
_DASHBOARD_ONLY_PATH_PREFIXES = ("/dashboard",)
# Path prefixes that are only served by the model playground listener.
_PLAYGROUND_ONLY_PATH_PREFIXES = ("/playground",)
# Paths only served on api_interfaces (Anthropic + OpenAI public API).
_API_ONLY_PATH_PREFIXES = (
    "/v1/messages",
    "/v1/models",
    "/v1/images",
    "/v1/videos",
)
# Paths served by both ollama_interfaces and api_interfaces.
_SHARED_V1_PATH_PREFIXES = ("/v1/chat/completions",)
# Paths only served on ollama_interfaces (Ollama-compatible /api/*).
_OLLAMA_ONLY_PATH_PREFIXES = ("/api",)


def _has_path_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + "/") for p in prefixes)


def _local_port(request: Request) -> int | None:
    server = request.scope.get("server") or (None, None)
    return server[1] if isinstance(server, (tuple, list)) and len(server) > 1 else None


def _listener_name(request: Request) -> str:
    settings: Settings = request.app.state.settings
    local_port = _local_port(request)
    if settings.admin_listener_enabled and local_port == settings.admin_port:
        return "admin"
    if settings.dashboard_listener_enabled and local_port == settings.dashboard_port:
        return "dashboard"
    if settings.playground_listener_enabled and local_port == settings.playground_port:
        return "playground"
    iface = _interface_for(request)
    if iface is not None:
        from .config import OllamaInterface

        prefix = "ollama" if isinstance(iface, OllamaInterface) else "api"
        return f"{prefix}:{iface.name}"
    if local_port is None:
        return "unknown"
    return f"port-{local_port}"


def _request_surface(path: str) -> str:
    if _has_path_prefix(path, _ADMIN_ONLY_PATH_PREFIXES):
        return "admin"
    if _has_path_prefix(path, _DASHBOARD_ONLY_PATH_PREFIXES):
        return "dashboard"
    if _has_path_prefix(path, _PLAYGROUND_ONLY_PATH_PREFIXES):
        return "playground"
    if path == "/v1/messages/count_tokens" or path.startswith("/v1/messages/count_tokens/"):
        return "anthropic-count-tokens"
    if _has_path_prefix(path, _API_ONLY_PATH_PREFIXES):
        if path.startswith("/v1/images"):
            return "images"
        if path.startswith("/v1/videos"):
            return "videos"
        return "anthropic" if path.startswith("/v1/messages") else "models"
    if _has_path_prefix(path, _SHARED_V1_PATH_PREFIXES):
        return "openai"
    if path == "/":
        return "root"
    if path.startswith("/api/"):
        return "ollama"
    return "other"


def _request_log_context(request: Request) -> Dict[str, Any]:
    local_port = _local_port(request)
    client = request.client.host if request.client is not None else "-"
    return {
        "listener": _listener_name(request),
        "port": local_port if local_port is not None else "-",
        "surface": _request_surface(request.url.path),
        "client": client,
        "method": request.method,
        "path": request.url.path,
    }


def _log_request_access(request: Request, status_code: int, duration_ms: float) -> None:
    ctx = _request_log_context(request)
    logger.info(
        "access listener=%s port=%s surface=%s client=%s method=%s path=%s status=%s duration_ms=%.2f",
        ctx["listener"],
        ctx["port"],
        ctx["surface"],
        ctx["client"],
        ctx["method"],
        ctx["path"],
        status_code,
        duration_ms,
    )


def _log_request_exception(request: Request, duration_ms: float) -> None:
    ctx = _request_log_context(request)
    logger.exception(
        "access listener=%s port=%s surface=%s client=%s method=%s path=%s status=500 duration_ms=%.2f",
        ctx["listener"],
        ctx["port"],
        ctx["surface"],
        ctx["client"],
        ctx["method"],
        ctx["path"],
        duration_ms,
    )


def _is_external_request(request: Request) -> bool:
    """True when the request landed on an ``api_interfaces[*]`` listener."""
    settings: Settings = request.app.state.settings
    local_port = _local_port(request)
    if local_port is None:
        return False
    return any(it.port == local_port for it in settings.api_interfaces)


def _install_port_router(app: FastAPI) -> None:
    """Route requests to the listener that owns each path.

    Every TCP listener fake_ollama owns sees the same FastAPI ``app``;
    this middleware enforces which paths are valid per listener:

    * admin port → only ``/admin/*``
    * dashboard port → only ``/dashboard/*`` (``/`` redirects to ``/dashboard/``)
    * playground port → ``/playground/*``, model discovery, chat, image and
      video generation surfaces used by the capability-aware debugger
    * ``ollama_interfaces[*].port`` → ``/`` plus ``/api/*`` plus ``/v1/chat/completions``
    * ``api_interfaces[*].port`` → ``/v1/messages*``, ``/v1/models``, ``/v1/images*``,
      ``/v1/videos*``, ``/v1/chat/completions``
    * any other port → 404
    """

    @app.middleware("http")
    async def _split(request: Request, call_next):
        settings: Settings = request.app.state.settings
        local_port = _local_port(request)
        path = request.url.path
        started_at = time.perf_counter()

        metrics: Optional[RequestMetrics] = getattr(
            request.app.state, "request_metrics", None
        )
        rid: Optional[int] = None
        if metrics is not None:
            ctx = _request_log_context(request)
            # Skip the management-plane surfaces — dashboard self-polls every
            # ~10s and admin is config traffic. Counting them buries real
            # model requests and (on a freshly-restarted process) makes the
            # Stats panel look like it only contains dashboard hits.
            if ctx["surface"] not in ("dashboard", "admin", "playground"):
                rid = metrics.begin(
                    listener=ctx["listener"],
                    port=ctx["port"],
                    surface=ctx["surface"],
                    client=ctx["client"],
                    method=ctx["method"],
                    path=ctx["path"],
                )
        request.state.request_metric_id = rid

        finalized = False

        def _finish(response):
            nonlocal finalized
            finalized = True
            _log_request_access(
                request,
                response.status_code,
                (time.perf_counter() - started_at) * 1000.0,
            )
            if rid is not None and metrics is not None:
                metrics.end(rid, status=response.status_code)
            return response

        try:
            if getattr(request.app.state, "shutdown_requested", False):
                return _finish(
                    JSONResponse({"detail": "server is shutting down"}, status_code=503)
                )

            # Determine which listener this request hit.
            is_admin = (
                settings.admin_listener_enabled and local_port == settings.admin_port
            )
            is_dashboard = (
                settings.dashboard_listener_enabled
                and local_port == settings.dashboard_port
            )
            is_playground = (
                settings.playground_listener_enabled
                and local_port == settings.playground_port
            )
            ollama_iface = next(
                (it for it in settings.ollama_interfaces if it.port == local_port),
                None,
            )
            api_iface = next(
                (it for it in settings.api_interfaces if it.port == local_port),
                None,
            )
            admin_only = _has_path_prefix(path, _ADMIN_ONLY_PATH_PREFIXES)
            dashboard_only = _has_path_prefix(path, _DASHBOARD_ONLY_PATH_PREFIXES)
            playground_only = _has_path_prefix(path, _PLAYGROUND_ONLY_PATH_PREFIXES)
            api_only = _has_path_prefix(path, _API_ONLY_PATH_PREFIXES)
            shared_v1 = _has_path_prefix(path, _SHARED_V1_PATH_PREFIXES)
            ollama_only = _has_path_prefix(path, _OLLAMA_ONLY_PATH_PREFIXES)

            # TestClient / ASGI direct calls present a synthetic local_port
            # (e.g. 80 from Starlette TestClient) that won't match any
            # configured listener. When no listener matches, fall back to
            # a configured interface so unit tests can hit /api/* and /v1/*
            # without binding real ports. Prefer an api_interface for
            # api-only paths, otherwise the first ollama_interface.
            if (
                not is_admin
                and not is_dashboard
                and not is_playground
                and ollama_iface is None
                and api_iface is None
            ):
                if api_only and settings.api_interfaces:
                    api_iface = settings.api_interfaces[0]
                elif settings.ollama_interfaces:
                    ollama_iface = settings.ollama_interfaces[0]
                elif settings.api_interfaces:
                    api_iface = settings.api_interfaces[0]

            # Admin listener: only /admin/*.
            if is_admin:
                if not admin_only:
                    return _finish(JSONResponse({"detail": "not found"}, status_code=404))
                return _finish(await call_next(request))

            # Dashboard listener: only /dashboard/* and a / redirect.
            if is_dashboard:
                if path == "/":
                    return _finish(RedirectResponse("/dashboard/"))
                if not dashboard_only:
                    return _finish(JSONResponse({"detail": "not found"}, status_code=404))
                return _finish(await call_next(request))

            # Playground listener: its static page plus model discovery, chat,
            # image and video endpoints.  It deliberately exposes no other API
            # surface.
            if is_playground:
                if path == "/":
                    return _finish(RedirectResponse("/playground/"))
                playground_media = _has_path_prefix(
                    path, ("/v1/images", "/v1/videos")
                )
                model_discovery = _has_path_prefix(path, ("/v1/models",))
                if (
                    playground_only
                    or model_discovery
                    or shared_v1
                    or playground_media
                ):
                    return _finish(await call_next(request))
                return _finish(JSONResponse({"detail": "not found"}, status_code=404))

            # Management paths can only be served on their own listeners.
            if admin_only or dashboard_only or playground_only:
                return _finish(JSONResponse({"detail": "not found"}, status_code=404))

            # Ollama interface: /, /api/*, shared /v1.
            if ollama_iface is not None:
                if api_only:
                    return _finish(
                        JSONResponse({"detail": "not found"}, status_code=404)
                    )
                if ollama_only or shared_v1 or path in ("/",):
                    return _finish(await call_next(request))
                return _finish(
                    JSONResponse({"detail": "not found"}, status_code=404)
                )

            # API interface: /v1/messages*, /v1/models, shared /v1.
            if api_iface is not None:
                if ollama_only:
                    return _finish(
                        JSONResponse({"detail": "not found"}, status_code=404)
                    )
                if api_only or shared_v1:
                    return _finish(await call_next(request))
                return _finish(
                    JSONResponse({"detail": "not found"}, status_code=404)
                )

            # Unknown listener — should not happen, but fail closed.
            return _finish(JSONResponse({"detail": "not found"}, status_code=404))
        except BaseException as exc:
            # ``CancelledError`` is a BaseException in 3.8+; catching only
            # ``Exception`` leaves the in-flight record orphaned when the
            # client disconnects or the request is cancelled upstream.
            if not finalized:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if isinstance(exc, (asyncio.CancelledError, ClientDisconnect)):
                    # 499 = "client closed request" (nginx convention) —
                    # distinguishes cancellation from a real 5xx in stats.
                    status_code = 499
                    _log_request_access(request, status_code, elapsed_ms)
                else:
                    status_code = 500
                    _log_request_exception(request, elapsed_ms)
                if rid is not None and metrics is not None:
                    metrics.end(rid, status=status_code)
                if isinstance(exc, ClientDisconnect):
                    return Response(status_code=status_code)
            raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:
    from .api.routes import register_routes

    register_routes(app)


def _read_error_text(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.text
        except Exception:  # pragma: no cover
            pass
    return str(exc) or exc.__class__.__name__


def _upstream_error_status(exc: httpx.HTTPError) -> int:
    if isinstance(exc, LocalTargetResourceError):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return 502


def _anthropic_error_response(exc: httpx.HTTPError) -> JSONResponse:
    error_type = (
        exc.error_type
        if isinstance(exc, LocalTargetResourceError)
        else "upstream_error"
    )
    return JSONResponse(
        status_code=_upstream_error_status(exc),
        content={
            "type": "error",
            "error": {"type": error_type, "message": _read_error_text(exc)},
        },
    )


def _log_upstream_error(
    request: Request, exc: httpx.HTTPError, upstream_payload: Dict[str, Any]
) -> None:
    """Log enough context to debug 4xx/5xx responses from the upstream."""
    body = _read_error_text(exc)
    ctx = _request_log_context(request)
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
        "upstream listener=%s port=%s surface=%s method=%s path=%s status=%s error=%s request=%s",
        ctx["listener"],
        ctx["port"],
        ctx["surface"],
        ctx["method"],
        ctx["path"],
        _upstream_error_status(exc),
        body,
        json.dumps(summary, ensure_ascii=False),
    )
    metrics: Optional[RequestMetrics] = getattr(
        request.app.state, "request_metrics", None
    )
    rid = getattr(request.state, "request_metric_id", None)
    if metrics is not None and rid is not None:
        metrics.set_error_type(rid, exc.__class__.__name__)


def _upstream_error(exc: httpx.HTTPError) -> JSONResponse:
    return JSONResponse(
        status_code=_upstream_error_status(exc),
        content={"error": _read_error_text(exc)},
    )


def _json_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def _authorise_interface(request: Request):
    iface = _interface_for(request)
    if iface is None:
        raise HTTPException(
            status_code=404,
            detail="this listener does not serve model traffic",
        )
    if iface.auth_required:
        token = _bearer_or_api_key(request)
        if not iface.is_valid_token(token):
            raise HTTPException(
                status_code=401,
                detail=(
                    "missing or invalid api token (send via x-api-key "
                    "or Authorization: Bearer header)"
                ),
            )
    return iface


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
            **params,
        )
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, payload)
        return _upstream_error(exc)
    return _image_response(images, payload)


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
    _enforce_reference_image_limit(target, "video_generation", image_inputs)
    params = _video_request_params(payload, target, app=app)
    revised_prompt: Optional[str] = None
    context_ir_profile_name = getattr(target, "context_ir_profile", None)
    video_mode = "auto"
    if context_ir_profile_name or getattr(target, "preset", "") == "minimax_h3":
        try:
            video_mode = resolve_base_mode(
                str(
                    payload.get("context_ir_mode")
                    or payload.get("video_mode")
                    or "auto"
                ),
                len(image_inputs),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if context_ir_profile_name:
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
            estimated_vram_gb=profile.estimated_vram_gb,
            estimated_memory_gb=profile.estimated_memory_gb,
            **params,
        )
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, payload)
        return _upstream_error(exc)
    return _image_response(videos, payload, revised_prompt=revised_prompt)


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
        except httpx.HTTPError as exc:
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
        except httpx.HTTPError as exc:
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
                openai_model, show_thinking=profile.show_thinking
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
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, ollama_payload)
                err_frame = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": openai_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"[upstream error: {_read_error_text(exc)}]"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
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
                async for line in lc.stream_chat(
                    llama_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                ):
                    if line.startswith("data:"):
                        yield (line + "\n\n").encode("utf-8")
                    else:
                        yield ("data: " + line + "\n\n").encode("utf-8")
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, llama_payload)
                err_frame = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": openai_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"[upstream error: {_read_error_text(exc)}]"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                yield b"data: [DONE]\n\n"

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
                async for line in loc.stream_chat(
                    forward_payload,
                    estimated_vram_gb=estimated_vram_gb,
                    estimated_memory_gb=estimated_memory_gb,
                ):
                    if line.startswith("data:"):
                        yield (line + "\n\n").encode("utf-8")
                    else:
                        yield ("data: " + line + "\n\n").encode("utf-8")
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, forward_payload)
                err_frame = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": openai_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"[upstream error: {_read_error_text(exc)}]"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                yield b"data: [DONE]\n\n"

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
                async for line in oc_remote.stream_chat(forward_payload):
                    if line.startswith("data:"):
                        yield (line + "\n\n").encode("utf-8")
                    else:
                        yield ("data: " + line + "\n\n").encode("utf-8")
            except httpx.HTTPError as exc:
                _log_upstream_error(request, exc, forward_payload)
                err_frame = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": openai_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"[upstream error: {_read_error_text(exc)}]"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield (
                    "data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n"
                ).encode("utf-8")
                yield b"data: [DONE]\n\n"

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
            openai_model, show_thinking=profile.show_thinking
        )
        try:
            async for event_type, data in client.stream_messages(upstream_payload):
                for frame in translator.feed_event(event_type, data):
                    yield (
                        "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
                    ).encode("utf-8")
        except httpx.HTTPError as exc:
            _log_upstream_error(request, exc, upstream_payload)
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


# ---------------------------------------------------------------------------
# Reverse proxy: Anthropic Messages API in -> local Ollama out (or pass-through)
# ---------------------------------------------------------------------------


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
    return JSONResponse({"input_tokens": int(data.get("input_tokens") or 0)})


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
            except httpx.HTTPError as exc:
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
            except httpx.HTTPError as exc:
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
            except httpx.HTTPError as exc:
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
            except httpx.HTTPError as exc:
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
        try:
            async for event_type, data in client.stream_messages(upstream_payload):
                yield (
                    f"event: {event_type}\ndata: "
                    + json.dumps(data, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")
        except httpx.HTTPError as exc:
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
