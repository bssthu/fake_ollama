"""Thin async client for the upstream Anthropic-compatible API."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from .request_data_log import (
    body_from_bytes,
    body_from_json,
    body_from_text,
    headers_from_mapping,
    log_data_event,
    request_data_logging_enabled,
)


class AnthropicClient:
    def __init__(
        self,
        base_url: str,
        auth_token: str,
        *,
        timeout: float = 300.0,
        trust_env: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=trust_env)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AnthropicClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # headers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        token = self.auth_token
        # Support both Anthropic-native (x-api-key) and OpenAI-style proxies
        # (Authorization: Bearer ...) since "Anthropic-compatible" gateways
        # vary in what they accept.
        return {
            "x-api-key": token,
            "authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "application/json",
        }

    # ------------------------------------------------------------------
    # requests
    # ------------------------------------------------------------------

    async def messages(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/messages"
        headers = self._headers()
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="anthropic",
                operation="messages",
                method="POST",
                url=url,
                headers=headers_from_mapping(headers),
                body=body_from_json(payload),
            )
        try:
            resp = await self._client.post(url, json=payload, headers=headers)
        except BaseException as exc:
            log_data_event(
                "backend_error",
                backend="anthropic",
                operation="messages",
                method="POST",
                url=url,
                error=f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
            )
            raise
        if resp.status_code >= 400:
            # Make sure the body is read so callers can include it in error
            # responses / logs.
            try:
                await resp.aread()
            except Exception:
                pass
        if request_data_logging_enabled():
            log_data_event(
                "backend_response_start",
                backend="anthropic",
                operation="messages",
                method="POST",
                url=url,
                status=resp.status_code,
                headers=headers_from_mapping(dict(resp.headers)),
            )
            log_data_event(
                "backend_response_body",
                backend="anthropic",
                operation="messages",
                method="POST",
                url=url,
                body=body_from_bytes(resp.content),
            )
            log_data_event(
                "backend_response_end",
                backend="anthropic",
                operation="messages",
                method="POST",
                url=url,
                status=resp.status_code,
                response_bytes=len(resp.content),
            )
        if resp.status_code >= 400:
            log_data_event(
                "backend_error",
                backend="anthropic",
                operation="messages",
                method="POST",
                url=url,
                error=f"http status {resp.status_code}",
            )
            resp.raise_for_status()
        return resp.json()

    async def count_tokens(
        self,
        payload: Dict[str, Any],
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/messages/count_tokens"
        headers = self._headers()
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="anthropic",
                operation="count_tokens",
                method="POST",
                url=url,
                params=params or {},
                headers=headers_from_mapping(headers),
                body=body_from_json(payload),
            )
        try:
            resp = await self._client.post(
                url,
                json=payload,
                headers=headers,
                params=params,
            )
        except BaseException as exc:
            log_data_event(
                "backend_error",
                backend="anthropic",
                operation="count_tokens",
                method="POST",
                url=url,
                params=params or {},
                error=f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
            )
            raise
        if resp.status_code >= 400:
            try:
                await resp.aread()
            except Exception:
                pass
        if request_data_logging_enabled():
            log_data_event(
                "backend_response_start",
                backend="anthropic",
                operation="count_tokens",
                method="POST",
                url=url,
                status=resp.status_code,
                headers=headers_from_mapping(dict(resp.headers)),
            )
            log_data_event(
                "backend_response_body",
                backend="anthropic",
                operation="count_tokens",
                method="POST",
                url=url,
                body=body_from_bytes(resp.content),
            )
            log_data_event(
                "backend_response_end",
                backend="anthropic",
                operation="count_tokens",
                method="POST",
                url=url,
                status=resp.status_code,
                response_bytes=len(resp.content),
            )
        if resp.status_code >= 400:
            log_data_event(
                "backend_error",
                backend="anthropic",
                operation="count_tokens",
                method="POST",
                url=url,
                params=params or {},
                error=f"http status {resp.status_code}",
            )
            resp.raise_for_status()
        return resp.json()

    async def stream_messages(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Yield (event_type, json_data) tuples from the upstream SSE stream."""
        url = f"{self.base_url}/v1/messages"
        headers = {**self._headers(), "accept": "text/event-stream"}
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="anthropic",
                operation="stream_messages",
                method="POST",
                url=url,
                headers=headers_from_mapping(headers),
                body=body_from_json(payload),
            )
        response_started = False
        status_code: Optional[int] = None
        response_bytes = 0
        outcome = "complete"
        error: Optional[str] = None
        stream_done = False
        try:
            async with self._client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                response_started = True
                status_code = resp.status_code
                log_data_event(
                    "backend_response_start",
                    backend="anthropic",
                    operation="stream_messages",
                    method="POST",
                    url=url,
                    status=resp.status_code,
                    headers=headers_from_mapping(dict(resp.headers)),
                )
                if resp.status_code >= 400:
                    # In streaming mode the body has not been consumed yet; we
                    # MUST aread() before raise_for_status() or `exc.response.text`
                    # will be empty (httpx ResponseNotRead).
                    try:
                        error_body = await resp.aread()
                    except Exception:
                        error_body = b""
                    response_bytes += len(error_body)
                    log_data_event(
                        "backend_response_body",
                        backend="anthropic",
                        operation="stream_messages",
                        method="POST",
                        url=url,
                        body=body_from_bytes(error_body),
                    )
                    resp.raise_for_status()
                event_name: Optional[str] = None
                async for raw_line in resp.aiter_lines():
                    if raw_line is None:
                        continue
                    response_bytes += len(raw_line.encode("utf-8"))
                    log_data_event(
                        "backend_response_body",
                        backend="anthropic",
                        operation="stream_messages",
                        method="POST",
                        url=url,
                        body=body_from_text(raw_line),
                    )
                    line = raw_line.rstrip("\r")
                    if line == "":
                        event_name = None
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                        continue
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        parsed_event = event_name or data.get("type", "message")
                        if parsed_event == "message_stop" or data.get("type") == "message_stop":
                            stream_done = True
                        yield (parsed_event, data)
        except BaseException as exc:
            if exc.__class__.__name__ == "GeneratorExit":
                outcome = "complete" if stream_done else "closed"
                error = None if stream_done else "consumer closed stream"
            else:
                outcome = (
                    "cancelled"
                    if exc.__class__.__name__ == "CancelledError"
                    else "exception"
                )
                error = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}"
                log_data_event(
                    "backend_error",
                    backend="anthropic",
                    operation="stream_messages",
                    method="POST",
                    url=url,
                    error=error,
                )
            raise
        finally:
            if response_started:
                log_data_event(
                    "backend_response_end",
                    backend="anthropic",
                    operation="stream_messages",
                    method="POST",
                    url=url,
                    status=status_code,
                    outcome=outcome,
                    response_bytes=response_bytes,
                    error=error,
                )
