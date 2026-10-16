"""Listener admission, credentials, model dispatch and request context."""

from __future__ import annotations

import json
import asyncio
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.requests import ClientDisconnect
from .config import FORWARDED_BY_HEADER, INSTANCE_ID, GenericOpenAITarget, Settings, parse_forwarded_chain, reset_inbound_forwarded_chain, set_inbound_forwarded_chain
from .openai_client import OpenAIClient
from .thinking_cache import bind_scope as bind_thinking_scope
from .security import is_loopback_host, require_management_access, playground_anonymous_allowed
from .dashboard import RequestMetrics

import logging

logger = logging.getLogger("fake_ollama")

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
    # An opt-in Python-only test hook; never infer test mode from a port/header.
    if not getattr(request.app.state, "test_listener_fallback", False):
        return None
    if _has_path_prefix(request.scope["path"], _API_ONLY_PATH_PREFIXES) and settings.api_interfaces:
        return settings.api_interfaces[0]
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
    bind_thinking_scope(
        principal=_bearer_or_api_key(request) if iface.auth_required else "anonymous:" + iface.name,
        backend=backend.source.model_dump(mode="json"),
        model=backend.source.resolve_model(model),
        configuration=[request.app.state.thinking_cache_namespace, settings.model_dump(mode="json")],
        allow_unsigned=getattr(backend.source, "allow_unsigned_thinking", backend.protocol == "openai"),
        missing_policy=getattr(backend.source, "missing_thinking_policy", "omit"),
    )
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
            raise HTTPException(
                status_code=503,
                detail=f"anthropic upstream '{backend.name}' is not initialised",
            )
        return app.state.runtime_resources.check(client)
    if backend.protocol == "openai" and backend.kind == "remote":
        return _openai_client_for(app, backend.name)
    if backend.protocol == "ollama":
        client = app.state.ollama_clients.get(backend.name)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"ollama_target '{backend.name}' is not initialised",
            )
        return app.state.runtime_resources.check(client)
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
        return app.state.runtime_resources.check(client)
    if backend.protocol == "comfyui":
        client = app.state.comfyui_clients.get(backend.name)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=f"comfyui_target '{backend.name}' is not initialised",
            )
        return app.state.runtime_resources.check(client)
    raise HTTPException(
        status_code=500,
        detail=f"no client wiring for backend {backend.name!r} ({backend.protocol}/{backend.kind})",
    )


def _profile_for_public_id(settings: Settings, iface: Any, public_id: str) -> Any:
    entry = iface.exposure_for_public_id(public_id) if iface is not None else None
    if entry is not None:
        return settings.profile_for(f"{entry.model}@{entry.target}")
    return settings.profile_for(public_id)


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
    """Select explicit credentials; anonymous use stays on a local listener."""
    candidates = [*settings.api_interfaces, *settings.ollama_interfaces]
    token = _bearer_or_api_key(request)
    if token:
        for iface in candidates:
            if iface.auth_required and iface.is_valid_token(token):
                return iface
        raise HTTPException(401, "invalid api token")
    if playground_anonymous_allowed(request):
        for iface in candidates:
            if not iface.auth_required and is_loopback_host(iface.host):
                return iface
    raise HTTPException(401, "Playground requires an interface API token")


_ADMIN_ONLY_PATH_PREFIXES = ("/admin",)


_DASHBOARD_ONLY_PATH_PREFIXES = ("/dashboard",)


_PLAYGROUND_ONLY_PATH_PREFIXES = ("/playground",)


_API_ONLY_PATH_PREFIXES = (
    "/v1/messages",
    "/v1/models",
    "/v1/images",
    "/v1/videos",
)


_SHARED_V1_PATH_PREFIXES = ("/v1/chat/completions",)


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
        "surface": _request_surface(request.scope["path"]),
        "client": client,
        "method": request.method,
        "path": request.scope["path"],
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
        path = request.scope["path"]
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

        async def _serve():
            # Header checks run before any request parsing or data logging.
            try:
                if path.startswith(("/admin", "/dashboard")):
                    require_management_access(request)
                elif path.startswith(("/api/", "/v1/", "/playground/api/")):
                    _authorise_interface(request)
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
            return await call_next(request)

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

            # Synthetic ASGI callers must explicitly opt into the test hook.
            if (
                getattr(request.app.state, "test_listener_fallback", False)
                and not is_admin
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
                return _finish(await _serve())

            # Dashboard listener: only /dashboard/* and a / redirect.
            if is_dashboard:
                if path == "/":
                    return _finish(RedirectResponse("/dashboard/"))
                if not dashboard_only:
                    return _finish(JSONResponse({"detail": "not found"}, status_code=404))
                return _finish(await _serve())

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
                    return _finish(await _serve())
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
                    return _finish(await _serve())
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
                    return _finish(await _serve())
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
