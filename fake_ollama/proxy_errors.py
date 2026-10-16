"""Shared upstream error formatting and diagnostics."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from .vram import LocalTargetResourceError
from .dashboard import RequestMetrics
from .routing import (
    _request_log_context,
)

import logging

logger = logging.getLogger("fake_ollama")

def _read_error_text(exc: Exception) -> str:
    if not isinstance(exc, httpx.HTTPError):
        return "Upstream stream failed (" + type(exc).__name__ + ")"
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
