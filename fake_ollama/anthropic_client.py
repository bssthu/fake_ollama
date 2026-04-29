"""Thin async client for the upstream Anthropic-compatible API."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx


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
        resp = await self._client.post(url, json=payload, headers=self._headers())
        if resp.status_code >= 400:
            # Make sure the body is read so callers can include it in error
            # responses / logs.
            try:
                await resp.aread()
            except Exception:
                pass
            resp.raise_for_status()
        return resp.json()

    async def stream_messages(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Yield (event_type, json_data) tuples from the upstream SSE stream."""
        url = f"{self.base_url}/v1/messages"
        headers = {**self._headers(), "accept": "text/event-stream"}
        async with self._client.stream(
            "POST", url, json=payload, headers=headers
        ) as resp:
            if resp.status_code >= 400:
                # In streaming mode the body has not been consumed yet; we
                # MUST aread() before raise_for_status() or `exc.response.text`
                # will be empty (httpx ResponseNotRead).
                try:
                    await resp.aread()
                except Exception:
                    pass
                resp.raise_for_status()
            event_name: Optional[str] = None
            async for raw_line in resp.aiter_lines():
                if raw_line is None:
                    continue
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
                    yield (event_name or data.get("type", "message"), data)
