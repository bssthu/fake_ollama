"""Async client for a local Ollama-compatible server.

Used by the reverse proxy: convert an Anthropic Messages request to Ollama
format, POST it to the local Ollama daemon, and stream the response back.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

import httpx


class OllamaClient:
    """Tiny async wrapper around an Ollama-compatible HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        trust_env: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=timeout, trust_env=trust_env)
            self._owns_client = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Non-stream chat: POST /api/chat with stream=false."""
        body = dict(payload)
        body["stream"] = False
        resp = await self._client.post(f"{self._base}/api/chat", json=body)
        await resp.aread()
        resp.raise_for_status()
        return resp.json()

    async def stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream chat: POST /api/chat with stream=true; yield raw NDJSON lines."""
        body = dict(payload)
        body["stream"] = True
        async with self._client.stream(
            "POST", f"{self._base}/api/chat", json=body
        ) as resp:
            if resp.status_code >= 400:
                err_body = await resp.aread()
                resp.raise_for_status()  # raises HTTPStatusError with body
                # Should be unreachable, but keep the error_body in scope so
                # type-checkers see it as used.
                _ = err_body
            async for raw_line in resp.aiter_lines():
                if raw_line:
                    yield raw_line.encode("utf-8")
