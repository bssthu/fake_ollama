"""Async client for a local Ollama-compatible server.

Used by the reverse proxy: convert an Anthropic Messages request to Ollama
format, POST it to the local Ollama daemon, and stream the response back.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx


logger = logging.getLogger("fake_ollama")


class OllamaClient:
    """Tiny async wrapper around an Ollama-compatible HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        trust_env: bool = False,
        auto_start: bool = False,
        start_command: Optional[str] = None,
        stop_command: Optional[str] = None,
        idle_timeout_seconds: Optional[float] = None,
        startup_timeout_seconds: float = 60.0,
        health_path: str = "/api/version",
        cwd: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._auto_start = auto_start
        self._start_command = start_command
        self._stop_command = stop_command
        self._idle_timeout = idle_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._health_path = health_path if health_path.startswith("/") else "/" + health_path
        self._cwd = cwd
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=timeout, trust_env=trust_env)
            self._owns_client = True
        self._process: Optional[asyncio.subprocess.Process] = None
        self._started_by_us = False
        self._start_lock = asyncio.Lock()
        self._active = 0
        self._last_used = time.monotonic()
        self._shutdown_requested = False

    @property
    def idle_timeout_seconds(self) -> Optional[float]:
        return self._idle_timeout

    def begin_shutdown(self) -> None:
        self._shutdown_requested = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        await self.stop_if_owned()

    async def _healthy(self) -> bool:
        try:
            resp = await self._client.get(
                self._base + self._health_path,
                timeout=min(5.0, self._timeout),
            )
            return 200 <= resp.status_code < 300
        except httpx.HTTPError:
            return False

    async def _ensure_ready(self) -> None:
        if await self._healthy():
            return
        if self._shutdown_requested:
            raise httpx.ConnectError(
                "Ollama target is unavailable and fake-ollama is shutting down; "
                "refusing auto-start"
            )
        if not (self._auto_start and self._start_command):
            return

        async with self._start_lock:
            if await self._healthy():
                return
            if self._shutdown_requested:
                raise httpx.ConnectError(
                    "Ollama target is unavailable and fake-ollama is shutting down; "
                    "refusing auto-start"
                )
            if self._process is None or self._process.returncode is not None:
                logger.info("starting Ollama target: %s", self._start_command)
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
                    logger.info("Ollama target ready at %s", self._base)
                    return
                await asyncio.sleep(1.0)
            raise httpx.ConnectError(
                f"Ollama target did not become healthy within {self._startup_timeout}s"
            )

    async def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Non-stream chat: POST /api/chat with stream=false."""
        await self._ensure_ready()
        self._active += 1
        body = dict(payload)
        body["stream"] = False
        try:
            resp = await self._client.post(f"{self._base}/api/chat", json=body)
            await resp.aread()
            resp.raise_for_status()
            return resp.json()
        finally:
            self._active -= 1
            self._last_used = time.monotonic()

    async def stream_chat(self, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream chat: POST /api/chat with stream=true; yield raw NDJSON lines."""
        await self._ensure_ready()
        self._active += 1
        body = dict(payload)
        body["stream"] = True
        try:
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
            logger.info("stopping Ollama target with stop_command")
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
                logger.info("terminating owned Ollama target process")
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._started_by_us = False
