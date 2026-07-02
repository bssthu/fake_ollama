"""Separate JSONL request/response data logging.

This logger is intentionally isolated from the normal application logger so
large request bodies and streaming chunks do not get mixed into operational
logs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

LOGGER_NAME = "fake_ollama.request_data"
DEFAULT_REQUEST_DATA_LOG_FILE = Path("logs") / "fake_ollama.requests.jsonl"
SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}

_logger = logging.getLogger(LOGGER_NAME)
_logger.propagate = False
_logger.setLevel(logging.INFO)
_configured_path: Optional[str] = None
_current_request_id: ContextVar[Optional[str]] = ContextVar(
    "fake_ollama_request_data_id", default=None
)
_CLIENT_CLOSED_EXCEPTION_NAMES = {"CancelledError", "ClientDisconnect"}


def configure_request_data_logging(log_file: Optional[str]) -> None:
    """Enable or disable isolated request-data logging."""
    global _configured_path

    for handler in list(_logger.handlers):
        _logger.removeHandler(handler)
        handler.close()

    _configured_path = None
    if not log_file:
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _configured_path = str(path)


def request_data_log_path() -> Optional[str]:
    return _configured_path


def request_data_logging_enabled() -> bool:
    return bool(_logger.handlers)


def current_request_id() -> Optional[str]:
    return _current_request_id.get()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> str:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return str(value)


def log_data_event(event: str, *, request_id: Optional[str] = None, **fields: Any) -> None:
    """Write one JSONL event if request-data logging is enabled."""
    if not request_data_logging_enabled():
        return

    rid = request_id if request_id is not None else current_request_id()
    record: Dict[str, Any] = {
        "ts": _now_iso(),
        "event": event,
    }
    if rid:
        record["request_id"] = rid
    record.update(fields)
    _logger.info(json.dumps(record, ensure_ascii=False, default=_json_default))
    for handler in _logger.handlers:
        handler.flush()


def body_from_bytes(body: bytes) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"bytes": len(body)}
    try:
        payload["encoding"] = "utf-8"
        payload["text"] = body.decode("utf-8")
    except UnicodeDecodeError:
        payload["encoding"] = "base64"
        payload["base64"] = base64.b64encode(body).decode("ascii")
    return payload


def body_from_text(text: str) -> Dict[str, Any]:
    return {
        "bytes": len(text.encode("utf-8")),
        "encoding": "utf-8",
        "text": text,
    }


def body_from_json(value: Any) -> Dict[str, Any]:
    return {"json": value}


def _redact_value(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"<redacted sha256:{digest}>"


def headers_from_raw(raw_headers: Iterable[tuple[bytes, bytes]]) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}
    for raw_name, raw_value in raw_headers:
        name = raw_name.decode("latin-1").lower()
        value = raw_value.decode("latin-1")
        if name in SENSITIVE_HEADERS:
            value = _redact_value(value)
        existing = headers.get(name)
        if existing is None:
            headers[name] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            headers[name] = [existing, value]
    return headers


def headers_from_mapping(headers: Dict[str, str]) -> Dict[str, Any]:
    return headers_from_raw(
        (
            str(k).encode("latin-1", errors="replace"),
            str(v).encode("latin-1", errors="replace"),
        )
        for k, v in headers.items()
    )


def _client_addr(scope: Scope) -> Optional[str]:
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        host = client[0]
        port = client[1] if len(client) > 1 else None
        return f"{host}:{port}" if port is not None else str(host)
    return None


def _server_addr(scope: Scope) -> Optional[str]:
    server = scope.get("server")
    if isinstance(server, (tuple, list)) and server:
        host = server[0]
        port = server[1] if len(server) > 1 else None
        return f"{host}:{port}" if port is not None else str(host)
    return None


def _scope_path(scope: Scope) -> str:
    return str(scope.get("path") or "")


def _should_log_scope(scope: Scope) -> bool:
    if scope.get("type") != "http":
        return False
    path = _scope_path(scope)
    if path == "/" or path.startswith("/admin"):
        return False
    return path.startswith("/api/") or path.startswith("/v1/")


class RequestDataLogMiddleware:
    """ASGI middleware that records inbound requests and outbound responses."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not request_data_logging_enabled() or not _should_log_scope(scope):
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        token = _current_request_id.set(request_id)
        started_at = time.perf_counter()
        method = str(scope.get("method") or "")
        path = _scope_path(scope)
        query_string = scope.get("query_string") or b""
        query = query_string.decode("latin-1") if query_string else ""
        response_status: Optional[int] = None
        response_bytes = 0
        outcome = "complete"
        error: Optional[str] = None

        log_data_event(
            "http_request_start",
            request_id=request_id,
            method=method,
            path=path,
            query=query,
            client=_client_addr(scope),
            server=_server_addr(scope),
            headers=headers_from_raw(scope.get("headers") or []),
        )

        replay_messages: list[Message] = []
        request_chunks: list[bytes] = []
        disconnected = False
        while True:
            message = await receive()
            replay_messages.append(message)
            message_type = message.get("type")
            if message_type == "http.request":
                chunk = message.get("body") or b""
                if chunk:
                    request_chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            elif message_type == "http.disconnect":
                disconnected = True
                break
            else:
                break

        request_body = b"".join(request_chunks)
        log_data_event(
            "http_request_body",
            request_id=request_id,
            disconnected=disconnected,
            body=body_from_bytes(request_body),
        )

        replay_index = 0

        async def replay_receive() -> Message:
            nonlocal replay_index
            if replay_index < len(replay_messages):
                msg = replay_messages[replay_index]
                replay_index += 1
                return msg
            return await receive()

        async def logging_send(message: Message) -> None:
            nonlocal response_status, response_bytes
            message_type = message.get("type")
            if message_type == "http.response.start":
                response_status = int(message.get("status") or 0)
                log_data_event(
                    "http_response_start",
                    request_id=request_id,
                    status=response_status,
                    headers=headers_from_raw(message.get("headers") or []),
                )
            elif message_type == "http.response.body":
                chunk = message.get("body") or b""
                response_bytes += len(chunk)
                if chunk:
                    log_data_event(
                        "http_response_body",
                        request_id=request_id,
                        more_body=bool(message.get("more_body", False)),
                        body=body_from_bytes(chunk),
                    )
            await send(message)

        try:
            await self.app(scope, replay_receive, logging_send)
        except BaseException as exc:
            outcome = (
                "cancelled"
                if exc.__class__.__name__ in _CLIENT_CLOSED_EXCEPTION_NAMES
                else "exception"
            )
            if outcome == "cancelled" and response_status is None:
                response_status = 499
            error = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}"
            log_data_event(
                "http_request_error",
                request_id=request_id,
                outcome=outcome,
                error=error,
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            end_outcome = (
                "cancelled"
                if outcome == "complete" and disconnected and response_status == 499
                else outcome
            )
            log_data_event(
                "http_request_end",
                request_id=request_id,
                outcome=end_outcome,
                status=response_status,
                duration_ms=round(duration_ms, 2),
                request_bytes=len(request_body),
                response_bytes=response_bytes,
                error=error,
            )
            _current_request_id.reset(token)
