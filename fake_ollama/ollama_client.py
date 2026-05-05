"""Async client for a local Ollama-compatible server.

Used by the reverse proxy: convert an Anthropic Messages request to Ollama
format, POST it to the local Ollama daemon, and stream the response back.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .process_utils import create_managed_subprocess_shell, terminate_process_tree
from .vram import VramCoordinator, VramReleaseCandidate


logger = logging.getLogger("fake_ollama")


@dataclass
class _LoadedModel:
    estimated_vram_gb: float
    last_used_monotonic: float


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
        target_name: str = "ollama",
        vram_coordinator: Optional[VramCoordinator] = None,
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
        self.target_id = f"ollama:{target_name}"
        self._vram_coordinator = vram_coordinator
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
        self._loaded_models: dict[str, _LoadedModel] = {}
        self._shutdown_requested = False
        if self._vram_coordinator is not None:
            self._vram_coordinator.register(self)

    @property
    def idle_timeout_seconds(self) -> Optional[float]:
        return self._idle_timeout

    @property
    def active_requests(self) -> int:
        return self._active

    @property
    def last_used_monotonic(self) -> float:
        return self._last_used

    def has_vram_reservation(self, model: str) -> bool:
        return bool(model and model in self._loaded_models)

    def vram_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[VramReleaseCandidate]:
        if self._active:
            return []
        candidates: list[VramReleaseCandidate] = []
        for model, loaded in self._loaded_models.items():
            if now - loaded.last_used_monotonic < idle_seconds:
                continue
            candidates.append(
                VramReleaseCandidate(
                    owner_id=self.target_id,
                    model=model,
                    estimated_vram_gb=loaded.estimated_vram_gb,
                    last_used_monotonic=loaded.last_used_monotonic,
                    release=lambda model=model: self._release_model_for_vram(model),
                )
            )
        return candidates

    def begin_shutdown(self) -> None:
        self._shutdown_requested = True

    async def aclose(self) -> None:
        if self._vram_coordinator is not None:
            self._vram_coordinator.unregister(self)
        if self._owns_client:
            await self._client.aclose()
        await self.stop_if_owned()

    async def _ensure_vram(
        self, model: Optional[str], estimated_vram_gb: Optional[float]
    ) -> None:
        if self._vram_coordinator is None or estimated_vram_gb is None:
            return
        await self._vram_coordinator.ensure_available(
            self,
            model=model or "",
            estimated_vram_gb=estimated_vram_gb,
        )

    def _mark_vram_reserved(
        self, model: Optional[str], estimated_vram_gb: Optional[float]
    ) -> None:
        if not model or estimated_vram_gb is None:
            return
        self._loaded_models[model] = _LoadedModel(
            estimated_vram_gb=estimated_vram_gb,
            last_used_monotonic=time.monotonic(),
        )
        if self._vram_coordinator is not None:
            self._vram_coordinator.confirm_loaded(self.target_id, model)

    def _discard_vram_pending(self, model: Optional[str]) -> None:
        if not model or self._vram_coordinator is None:
            return
        self._vram_coordinator.discard_pending(self.target_id, model)

    def _clear_all_vram_state(self) -> None:
        models = list(self._loaded_models.keys())
        self._loaded_models.clear()
        if self._vram_coordinator is not None:
            for m in models:
                self._vram_coordinator.discard_pending(self.target_id, m)

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
        self._clear_all_vram_state()
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
                self._process = await create_managed_subprocess_shell(
                    self._start_command,
                    cwd=self._cwd,
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

    async def chat(
        self,
        payload: Dict[str, Any],
        *,
        estimated_vram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Non-stream chat: POST /api/chat with stream=false."""
        body = dict(payload)
        body["stream"] = False
        model = str(body.get("model") or "")
        await self._ensure_vram(model, estimated_vram_gb)
        try:
            await self._ensure_ready()
        except BaseException:
            self._discard_vram_pending(model)
            raise
        self._active += 1
        try:
            try:
                resp = await self._client.post(f"{self._base}/api/chat", json=body)
                await resp.aread()
                resp.raise_for_status()
            except BaseException:
                self._discard_vram_pending(model)
                raise
            self._mark_vram_reserved(model, estimated_vram_gb)
            return resp.json()
        finally:
            self._active -= 1
            self._last_used = time.monotonic()

    async def stream_chat(
        self,
        payload: Dict[str, Any],
        *,
        estimated_vram_gb: Optional[float] = None,
    ) -> AsyncIterator[bytes]:
        """Stream chat: POST /api/chat with stream=true; yield raw NDJSON lines."""
        body = dict(payload)
        body["stream"] = True
        model = str(body.get("model") or "")
        await self._ensure_vram(model, estimated_vram_gb)
        try:
            await self._ensure_ready()
        except BaseException:
            self._discard_vram_pending(model)
            raise
        self._active += 1
        marked = False
        try:
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
                    self._mark_vram_reserved(model, estimated_vram_gb)
                    marked = True
                    async for raw_line in resp.aiter_lines():
                        if raw_line:
                            yield raw_line.encode("utf-8")
            except BaseException:
                if not marked:
                    self._discard_vram_pending(model)
                raise
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

    async def release_for_vram(self) -> bool:
        candidates = self.vram_release_candidates(
            now=time.monotonic(), idle_seconds=0.0
        )
        for candidate in candidates:
            if await candidate.release():
                return True
        return False

    async def _release_model_for_vram(self, model: str) -> bool:
        if self._active:
            return False
        if await self._unload_model(model):
            self._loaded_models.pop(model, None)
            self._discard_vram_pending(model)
            return True
        if self._started_by_us or self._stop_command:
            return await self.stop_if_owned()
        return False

    async def _unload_model(self, model: str) -> bool:
        try:
            resp = await self._client.post(
                f"{self._base}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=min(30.0, self._timeout),
            )
            await resp.aread()
            if 200 <= resp.status_code < 300:
                logger.info("unloaded Ollama model %s from %s", model, self._base)
                return True
            logger.warning(
                "failed to unload Ollama model %s from %s: status=%s body=%s",
                model,
                self._base,
                resp.status_code,
                resp.text,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "failed to unload Ollama model %s from %s: %s",
                model,
                self._base,
                exc,
            )
        return False

    async def stop_if_owned(self) -> bool:
        if self._stop_command:
            logger.info("stopping Ollama target with stop_command")
            proc = await asyncio.create_subprocess_shell(
                self._stop_command,
                cwd=self._cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            returncode = await proc.wait()
            if returncode == 0:
                self._started_by_us = False
                self._clear_all_vram_state()
                return True
            logger.warning("Ollama target stop_command exited with status %s", returncode)
            return False

        if self._process is not None and self._started_by_us:
            if self._process.returncode is None:
                logger.info("terminating owned Ollama target process tree")
                if not await terminate_process_tree(self._process, timeout=10.0):
                    logger.warning("failed to terminate owned Ollama target process tree")
                    return False
                self._started_by_us = False
                self._clear_all_vram_state()
                return True
            logger.warning(
                "owned Ollama target launcher process already exited; cannot stop "
                "remaining detached server process without stop_command"
            )
            return False
        return False
