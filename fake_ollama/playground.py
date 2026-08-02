"""Dependency-free, capability-aware model playground."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

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
