"""Async client for llama.cpp's OpenAI-compatible server.

Used by the reverse proxy to expose llama.cpp models on the same external
surface as Ollama targets. The client also owns optional lifecycle management:
health-check, auto-start, and idle stop for processes it starts itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx

logger = logging.getLogger("fake_ollama")


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str = "",
        timeout: float = 300.0,
        trust_env: bool = False,
        auto_start: bool = False,
        start_command: Optional[str] = None,
        stop_command: Optional[str] = None,
        idle_timeout_seconds: Optional[float] = None,
        startup_timeout_seconds: float = 120.0,
        health_path: str = "/health",
        cwd: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._auto_start = auto_start
        self._start_command = start_command
        self._stop_command = stop_command
        self._idle_timeout = idle_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._health_path = health_path if health_path.startswith("/") else "/" + health_path
        self._cwd = cwd
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=trust_env)
        self._owns_client = client is None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._started_by_us = False
        self._start_lock = asyncio.Lock()
        self._active = 0
        self._last_used = time.monotonic()

    @property
    def idle_timeout_seconds(self) -> Optional[float]:
        return self._idle_timeout

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if stream:
            headers["accept"] = "text/event-stream"
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
            headers["x-api-key"] = self._auth_token
        return headers

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        await self.stop_if_owned()

    async def _healthy(self) -> bool:
        try:
            resp = await self._client.get(
                self._base + self._health_path,
                headers=self._headers(),
                timeout=min(5.0, self._timeout),
            )
            return 200 <= resp.status_code < 300
        except httpx.HTTPError:
            return False

    async def _ensure_ready(self) -> None:
        if await self._healthy():
            return
        if not (self._auto_start and self._start_command):
            return

        async with self._start_lock:
            if await self._healthy():
                return
            if self._process is None or self._process.returncode is not None:
                logger.info("starting llama.cpp target: %s", self._start_command)
                self._process = await asyncio.create_subprocess_shell(
                    self._start_command,
                    cwd=self._cwd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                self._started_by_us = True

            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if await self._healthy():
                    logger.info("llama.cpp target ready at %s", self._base)
                    return
                await asyncio.sleep(1.0)
            raise httpx.ConnectError(
                f"llama.cpp target did not become healthy within {self._startup_timeout}s"
            )

    async def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_ready()
        self._active += 1
        try:
            body = dict(payload)
            body["stream"] = False
            resp = await self._client.post(
                f"{self._base}/v1/chat/completions",
                json=body,
                headers=self._headers(),
            )
            await resp.aread()
            resp.raise_for_status()
            return resp.json()
        finally:
            self._active -= 1
            self._last_used = time.monotonic()

    async def stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        await self._ensure_ready()
        self._active += 1
        body = dict(payload)
        body["stream"] = True
        try:
            async with self._client.stream(
                "POST",
                f"{self._base}/v1/chat/completions",
                json=body,
                headers=self._headers(stream=True),
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if raw_line:
                        yield raw_line
        finally:
            self._active -= 1
            self._last_used = time.monotonic()

    async def stop_if_idle(self) -> None:
        if not self._idle_timeout or self._active:
            return
        if time.monotonic() - self._last_used < self._idle_timeout:
            return
        if not (self._started_by_us or self._stop_command):
            return
        await self.stop_if_owned()

    async def stop_if_owned(self) -> None:
        if self._stop_command:
            logger.info("stopping llama.cpp target with stop_command")
            proc = await asyncio.create_subprocess_shell(
                self._stop_command,
                cwd=self._cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._started_by_us = False
            return

        if self._process is not None and self._started_by_us:
            if self._process.returncode is None:
                logger.info("terminating owned llama.cpp target process")
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._started_by_us = False