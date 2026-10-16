"""Dependency-free, capability-aware model playground."""

from __future__ import annotations

from pathlib import Path

import json
import httpx
from typing import List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from . import routing, playground_support

from .config import Settings


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_PLAYGROUND_HTML = (_STATIC_DIR / "playground.html").read_text(encoding="utf-8")
_PLAYGROUND_CSS = (_STATIC_DIR / "playground.css").read_text(encoding="utf-8")
_PLAYGROUND_JS = (_STATIC_DIR / "playground.js").read_text(encoding="utf-8")

_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def register_playground_routes(app: FastAPI) -> None:
    """Register static playground assets on the dedicated listener."""
    settings: Settings = app.state.settings
    if not settings.playground_enabled:
        return

    @app.get("/playground", include_in_schema=False)
    @app.get("/playground/", include_in_schema=False)
    async def playground_index() -> HTMLResponse:
        return HTMLResponse(
            _PLAYGROUND_HTML,
            headers={
                **_NO_STORE_HEADERS,
                "Content-Security-Policy": (
                    "default-src 'self'; style-src 'self'; script-src 'self'; "
                    "connect-src 'self'; img-src 'self' data: blob:; "
                    "media-src 'self' data: blob:; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'none'"
                ),
            },
        )

    @app.get("/playground/playground.css", include_in_schema=False)
    async def playground_css() -> Response:
        return Response(
            _PLAYGROUND_CSS,
            media_type="text/css; charset=utf-8",
            headers=_NO_STORE_HEADERS,
        )

    @app.get("/playground/playground.js", include_in_schema=False)
    async def playground_js() -> Response:
        return Response(
            _PLAYGROUND_JS,
            media_type="text/javascript; charset=utf-8",
            headers=_NO_STORE_HEADERS,
        )

    @app.get("/playground/api/models", include_in_schema=False)
    async def playground_models(request: Request) -> JSONResponse:
        settings, iface = routing._model_interface_for_request(request)
        models = [
            playground_support._playground_model_entry(settings, iface, name)
            for name in routing._public_model_ids(iface)
        ]
        models.extend(
            playground_support._playground_context_ir_entry(settings, iface, profile)
            for profile in settings.h3_context_ir_profiles
            if profile.enabled and profile.playground_visible
        )
        return JSONResponse(
            {
                "schema_version": 1,
                "models": models,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/playground/api/external-models", include_in_schema=False)
    async def playground_external_models(request: Request) -> JSONResponse:
        settings, _iface = routing._model_interface_for_request(request)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        requested_profile = str(payload.get("profile") or "").strip()
        profile = settings.h3_context_ir_profile_by_name(requested_profile)
        if profile is None or not profile.enabled:
            raise HTTPException(
                status_code=404,
                detail=f"H3 Context-IR profile {requested_profile!r} not found",
            )
        if not profile.allow_external_api:
            raise HTTPException(
                status_code=403,
                detail="this H3 Context-IR profile does not allow external APIs",
            )
        protocol = playground_support._external_planner_protocol(payload.get("protocol"))
        base_url = playground_support._authorise_external_planner_url(profile, payload.get("base_url"))
        token = playground_support._external_planner_token(request)
        try:
            async with playground_support._external_planner_http_client(request.app) as client:
                response = await client.get(
                    f"{base_url}/v1/models",
                    headers=playground_support._external_planner_headers(protocol, token),
                    timeout=min(30.0, settings.timeout_seconds),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "external model detection failed: upstream returned "
                    f"HTTP {exc.response.status_code}"
                ),
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"external model detection failed: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="external model endpoint returned a non-object response",
            )
        raw_models = data.get("data") or data.get("models") or []
        names: List[str] = []
        seen: set[str] = set()
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
            if len(names) >= 1000:
                break
        return JSONResponse(
            {
                "protocol": protocol,
                "base_url": base_url,
                "models": names,
            },
            headers={"Cache-Control": "no-store"},
        )
