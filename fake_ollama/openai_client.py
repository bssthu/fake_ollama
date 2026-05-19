"""Async client for a remote OpenAI-compatible Chat Completions upstream.

Designed for backends that speak the OpenAI wire format but live on the
other side of the network (OpenAI, DeepSeek, Together, Groq, …). This is
deliberately a stripped-down sibling of :class:`fake_ollama.llama_cpp_client.LlamaCppClient`:
remote upstreams have no process to start, no VRAM to reserve and no
``auto_start`` semantics, so all that lifecycle code is gone.

Two surfaces are exposed:

* :meth:`chat` — non-streaming POST to ``/v1/chat/completions`` returning
  the decoded JSON body.
* :meth:`stream_chat` — streaming POST yielding the raw ``data: ...``
  SSE lines, which the reverse converters in
  :mod:`fake_ollama.reverse_converters` can translate into Anthropic
  events when the frontend speaks Anthropic / Ollama.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .config import outbound_cycle_headers
from .request_data_log import (
    body_from_bytes,
    body_from_json,
    body_from_text,
    headers_from_mapping,
    log_data_event,
    request_data_logging_enabled,
)


class OpenAIClient:
    """Thin async client for an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str = "",
        timeout: float = 300.0,
        trust_env: bool = False,
        upstream_name: str = "openai",
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._timeout = timeout
        self.target_id = f"openai:{upstream_name}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=trust_env)

    # ------------------------------------------------------------------
    # lifecycle (remote — there is nothing to manage)
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAIClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # headers
    # ------------------------------------------------------------------

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        h: Dict[str, str] = {
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
        }
        if self.auth_token:
            # Some gateways accept ``x-api-key`` instead of ``Authorization``;
            # set both so we work against either flavour.
            h["authorization"] = f"Bearer {self.auth_token}"
            h["x-api-key"] = self.auth_token
        h.update(outbound_cycle_headers())
        return h

    # ------------------------------------------------------------------
    # requests
    # ------------------------------------------------------------------

    async def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(payload)
        body["stream"] = False
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="openai",
                target_id=self.target_id,
                operation="chat",
                method="POST",
                url=url,
                headers=headers_from_mapping(headers),
                body=body_from_json(body),
            )
        try:
            resp = await self._client.post(url, json=body, headers=headers)
        except BaseException as exc:
            log_data_event(
                "backend_error",
                backend="openai",
                target_id=self.target_id,
                operation="chat",
                method="POST",
                url=url,
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
                backend="openai",
                target_id=self.target_id,
                operation="chat",
                method="POST",
                url=url,
                status=resp.status_code,
                headers=headers_from_mapping(dict(resp.headers)),
            )
            log_data_event(
                "backend_response_body",
                backend="openai",
                target_id=self.target_id,
                operation="chat",
                method="POST",
                url=url,
                body=body_from_bytes(resp.content),
            )
            log_data_event(
                "backend_response_end",
                backend="openai",
                target_id=self.target_id,
                operation="chat",
                method="POST",
                url=url,
                status=resp.status_code,
                response_bytes=len(resp.content),
            )
        if resp.status_code >= 400:
            log_data_event(
                "backend_error",
                backend="openai",
                target_id=self.target_id,
                operation="chat",
                method="POST",
                url=url,
                error=f"http status {resp.status_code}",
            )
            resp.raise_for_status()
        return resp.json()

    async def stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        body = dict(payload)
        body["stream"] = True
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers(stream=True)
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="openai",
                target_id=self.target_id,
                operation="stream_chat",
                method="POST",
                url=url,
                headers=headers_from_mapping(headers),
                body=body_from_json(body),
            )
        response_started = False
        status_code: Optional[int] = None
        response_bytes = 0
        outcome = "complete"
        error: Optional[str] = None
        stream_done = False
        try:
            async with self._client.stream(
                "POST", url, json=body, headers=headers
            ) as resp:
                response_started = True
                status_code = resp.status_code
                log_data_event(
                    "backend_response_start",
                    backend="openai",
                    target_id=self.target_id,
                    operation="stream_chat",
                    method="POST",
                    url=url,
                    status=resp.status_code,
                    headers=headers_from_mapping(dict(resp.headers)),
                )
                if resp.status_code >= 400:
                    try:
                        error_body = await resp.aread()
                    except Exception:
                        error_body = b""
                    response_bytes += len(error_body)
                    log_data_event(
                        "backend_response_body",
                        backend="openai",
                        target_id=self.target_id,
                        operation="stream_chat",
                        method="POST",
                        url=url,
                        body=body_from_bytes(error_body),
                    )
                    resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if raw_line is None:
                        continue
                    if raw_line:
                        response_bytes += len(raw_line.encode("utf-8"))
                        if raw_line.strip() == "data: [DONE]":
                            stream_done = True
                        log_data_event(
                            "backend_response_body",
                            backend="openai",
                            target_id=self.target_id,
                            operation="stream_chat",
                            method="POST",
                            url=url,
                            body=body_from_text(raw_line),
                        )
                        yield raw_line
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
                    backend="openai",
                    target_id=self.target_id,
                    operation="stream_chat",
                    method="POST",
                    url=url,
                    error=error,
                )
            raise
        finally:
            if response_started:
                log_data_event(
                    "backend_response_end",
                    backend="openai",
                    target_id=self.target_id,
                    operation="stream_chat",
                    method="POST",
                    url=url,
                    status=status_code,
                    outcome=outcome,
                    response_bytes=response_bytes,
                    error=error,
                )
