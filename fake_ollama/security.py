"""Shared admission checks for model traffic and the management pages."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import time
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse


MANAGEMENT_HEADER = "x-management-token"


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def token_matches(token: str, candidates) -> bool:
    encoded = token.encode("utf-8")
    return bool(token) and any(
        secrets.compare_digest(encoded, candidate.encode("utf-8"))
        for candidate in candidates
    )


def management_session_token(app) -> str:
    # Separate from model credentials, never persisted, and invalidated by a
    # management credential change as well as by a process restart.
    seed = getattr(app.state, "management_token_seed", None)
    if seed is None:
        seed = secrets.token_bytes(32)
        app.state.management_token_seed = seed
    credentials = json.dumps(app.state.settings.management_access_tokens).encode()
    return hmac.new(seed, credentials, hashlib.sha256).hexdigest()


def require_management_access(request: Request) -> None:
    settings = request.app.state.settings
    path = request.scope.get("path", "")
    surface = "admin" if path == "/admin" or path.startswith("/admin/") else "dashboard"
    expected_port = getattr(settings, f"{surface}_port")
    server = request.scope.get("server")
    if not getattr(settings, f"{surface}_listener_enabled") or not server or server[1] != expected_port:
        raise HTTPException(404, "not found")

    # Reject browser cross-site access even when the browser has cached HTTP
    # Basic credentials. Host is used only for origin comparison, never routing.
    host = request.headers.get("host", "")
    try:
        authority = urlsplit("http://" + host)
        valid_host = bool(authority.hostname) and not (
            authority.username or authority.password or authority.path
            or authority.query or authority.fragment
        )
        port = authority.port or (443 if request.scope.get("scheme") == "https" else 80)
    except ValueError:
        valid_host, port = False, None
    if not valid_host:
        raise HTTPException(400, "invalid Host header")
    origin = request.headers.get("origin")
    expected_origin = f"{request.scope.get('scheme', 'http')}://{host}"
    if (origin is not None and origin != expected_origin) or request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "cross-site management requests are not allowed")

    configured = settings.management_access_tokens
    page = request.method in ("GET", "HEAD") and path in (
        "/admin", "/admin/", "/dashboard", "/dashboard/",
    )
    supplied = request.headers.get(MANAGEMENT_HEADER, "")
    authorization = request.headers.get("authorization", "")
    bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    basic = ""
    if authorization.lower().startswith("basic "):
        try:
            credentials = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
            basic = credentials.split(":", 1)[1]
        except (ValueError, IndexError, UnicodeError):
            pass

    if not configured:
        # Local bootstrap is permitted only on a loopback listener and through
        # a loopback Host. A rebinding domain cannot obtain the page's token.
        peer = request.scope.get("client")
        if not (
            is_loopback_host(getattr(settings, f"{surface}_host"))
            and peer and is_loopback_host(peer[0])
            and is_loopback_host(authority.hostname or "")
            and port == expected_port
        ):
            raise HTTPException(403, "remote management requires management_access_tokens")
        if page:
            return
    elif token_matches(supplied or bearer, configured):
        return
    elif page and token_matches(basic, configured):
        return

    # Browser pages send this header on every API call. It is inaccessible to
    # other origins and never placed in URLs, cookies, or persistent storage.
    if not page and token_matches(supplied, [management_session_token(request.app)]):
        return
    raise HTTPException(
        401, "management authentication required",
        headers={"WWW-Authenticate": 'Basic realm="fake-ollama management", charset="UTF-8"'},
    )


def management_page(app, html: str) -> HTMLResponse:
    token = management_session_token(app)
    return HTMLResponse(
        html.replace("__MANAGEMENT_TOKEN__", token),
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            ),
        },
    )


class RequestBodyLimitMiddleware:
    """Bound bodies while they are consumed, without buffering a second copy."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        settings = scope["app"].state.settings
        limit = settings.max_request_body_bytes
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0:
                raise ValueError
        except ValueError:
            await JSONResponse({"detail": "invalid Content-Length"}, status_code=400)(scope, receive, send)
            return
        if length > limit:
            await JSONResponse({"detail": "request body too large"}, status_code=413)(scope, receive, send)
            return
        size = 0
        deadline = None
        complete = False

        async def bounded_receive():
            nonlocal size, deadline, complete
            if complete:
                return await receive()
            if deadline is None:
                deadline = time.monotonic() + settings.request_body_timeout_seconds
            try:
                message = await asyncio.wait_for(receive(), max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError as exc:
                raise HTTPException(408, "request body timed out") from exc
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > limit:
                    raise HTTPException(413, "request body too large")
                complete = not message.get("more_body", False)
            elif message["type"] == "http.disconnect":
                complete = True
            return message

        await self.app(scope, bounded_receive, send)
