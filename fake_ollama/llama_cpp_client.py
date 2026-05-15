"""Async client for llama.cpp's OpenAI-compatible server.

Used by the reverse proxy to expose llama.cpp models on the same external
surface as Ollama targets. The client also owns optional lifecycle management:
health-check, auto-start, and idle stop for processes it starts itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .process_utils import (
    create_managed_subprocess_exec,
    create_managed_subprocess_shell,
    terminate_process_tree,
)
from .request_data_log import (
    body_from_bytes,
    body_from_json,
    body_from_text,
    headers_from_mapping,
    log_data_event,
    request_data_logging_enabled,
)
from .vram import VRAM_IDLE_RECLAIM_SECONDS, VramCoordinator, VramReleaseCandidate

logger = logging.getLogger("fake_ollama")


@dataclass
class _LoadedModel:
    model: str
    estimated_vram_gb: float
    last_used_monotonic: float


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
        start_argv: Optional[list] = None,
        stop_command: Optional[str] = None,
        idle_timeout_seconds: Optional[float] = None,
        startup_timeout_seconds: float = 120.0,
        health_path: str = "/health",
        cwd: Optional[str] = None,
        launch_env: Optional[Dict[str, str]] = None,
        target_name: str = "llama.cpp",
        vram_coordinator: Optional[VramCoordinator] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._trust_env = trust_env
        self._auto_start = auto_start
        # Prefer argv (exec) over a shell string when both are provided:
        # exec captures the actual server PID instead of a cmd.exe wrapper,
        # which is essential for reliable termination on Windows.
        self._start_argv = list(start_argv) if start_argv else None
        self._start_command = start_command
        self._stop_command = stop_command
        self._idle_timeout = idle_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._health_path = health_path if health_path.startswith("/") else "/" + health_path
        self._cwd = cwd
        self._launch_env = launch_env
        self.target_id = f"llama.cpp:{target_name}"
        self._vram_coordinator = vram_coordinator
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=trust_env)
        self._owns_client = client is None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._started_by_us = False
        self._start_lock = asyncio.Lock()
        self._active = 0
        self._request_refs = 0
        self._last_used = time.monotonic()
        self._loaded_model: Optional[_LoadedModel] = None
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
        return self._loaded_model is not None

    def _begin_request_lifecycle(self) -> None:
        self._request_refs += 1

    def _end_request_lifecycle(self) -> None:
        self._request_refs -= 1
        self._last_used = time.monotonic()

    def vram_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[VramReleaseCandidate]:
        if self._active or self._request_refs or self._loaded_model is None:
            return []
        if now - self._loaded_model.last_used_monotonic < idle_seconds:
            return []
        if not (self._started_by_us or self._stop_command):
            return []
        loaded = self._loaded_model
        return [
            VramReleaseCandidate(
                owner_id=self.target_id,
                model=loaded.model,
                estimated_vram_gb=loaded.estimated_vram_gb,
                last_used_monotonic=loaded.last_used_monotonic,
                release=self._release_server_for_vram,
            )
        ]

    def loaded_model_snapshots(
        self,
        *,
        now: Optional[float] = None,
        idle_reclaim_seconds: float = VRAM_IDLE_RECLAIM_SECONDS,
    ) -> list[dict[str, object]]:
        if self._loaded_model is None:
            return []
        now = time.monotonic() if now is None else now
        loaded = self._loaded_model
        idle_seconds = max(0.0, now - loaded.last_used_monotonic)
        in_flight = bool(self._active or self._request_refs)
        can_stop = bool(self._started_by_us or self._stop_command)
        return [
            {
                "backend": "llama.cpp",
                "target_id": self.target_id,
                "model": loaded.model,
                "estimated_vram_gb": loaded.estimated_vram_gb,
                "estimated_vram_mib": loaded.estimated_vram_gb * 1024.0,
                "active_requests": self._active,
                "request_refs": self._request_refs,
                "idle_seconds": idle_seconds,
                "reclaimable": (
                    can_stop and not in_flight and idle_seconds >= idle_reclaim_seconds
                ),
            }
        ]

    def begin_shutdown(self) -> None:
        self._shutdown_requested = True

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if stream:
            headers["accept"] = "text/event-stream"
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
            headers["x-api-key"] = self._auth_token
        return headers

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
        self._loaded_model = _LoadedModel(
            model=model,
            estimated_vram_gb=estimated_vram_gb,
            last_used_monotonic=time.monotonic(),
        )
        if self._vram_coordinator is not None:
            self._vram_coordinator.confirm_loaded(self.target_id, model)

    def _touch_vram_reservation(self, model: Optional[str]) -> None:
        if not model or self._loaded_model is None:
            return
        if self._loaded_model.model == model:
            self._loaded_model.last_used_monotonic = time.monotonic()

    def _discard_vram_pending(self, model: Optional[str]) -> None:
        if not model or self._vram_coordinator is None:
            return
        self._vram_coordinator.discard_pending(self.target_id, model)

    def _clear_all_vram_state(self) -> None:
        loaded = self._loaded_model
        self._loaded_model = None
        if loaded is not None and self._vram_coordinator is not None:
            self._vram_coordinator.discard_pending(self.target_id, loaded.model)

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
        self._clear_all_vram_state()
        if self._shutdown_requested:
            raise httpx.ConnectError(
                "llama.cpp target is unavailable and fake-ollama is shutting down; "
                "refusing auto-start"
            )
        if not (self._auto_start and (self._start_argv or self._start_command)):
            return

        async with self._start_lock:
            if await self._healthy():
                return
            if self._shutdown_requested:
                raise httpx.ConnectError(
                    "llama.cpp target is unavailable and fake-ollama is shutting down; "
                    "refusing auto-start"
                )
            if self._process is None or self._process.returncode is not None:
                if self._start_argv:
                    logger.info(
                        "starting llama.cpp target: %s",
                        " ".join(self._start_argv),
                    )
                    self._process = await create_managed_subprocess_exec(
                        self._start_argv,
                        cwd=self._cwd,
                        env=self._launch_env,
                    )
                else:
                    logger.info("starting llama.cpp target: %s", self._start_command)
                    self._process = await create_managed_subprocess_shell(
                        self._start_command,
                        cwd=self._cwd,
                        env=self._launch_env,
                    )
                self._started_by_us = True

            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if await self._healthy():
                    logger.info("llama.cpp target ready at %s", self._base)
                    return
                # Surface early subprocess death instead of silently waiting
                # the full startup_timeout. Common causes: binary_path points
                # to a directory, missing GGUF, port already bound, etc.
                proc = self._process
                if proc is not None and proc.returncode is not None:
                    rc = proc.returncode
                    self._process = None
                    self._started_by_us = False
                    cmd_repr = (
                        " ".join(self._start_argv)
                        if self._start_argv
                        else self._start_command
                    )
                    raise httpx.ConnectError(
                        f"llama.cpp target process exited with code {rc} before "
                        f"becoming healthy. Check the start_command / binary_path / "
                        f"model_path. Command was: {cmd_repr}"
                    )
                if self._shutdown_requested:
                    raise httpx.ConnectError(
                        "fake-ollama is shutting down; aborting llama.cpp startup wait"
                    )
                await asyncio.sleep(1.0)
            raise httpx.ConnectError(
                f"llama.cpp target did not become healthy within {self._startup_timeout}s"
            )

    async def chat(
        self,
        payload: Dict[str, Any],
        *,
        estimated_vram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        body = dict(payload)
        body["stream"] = False
        model = str(body.get("model") or "")
        self._begin_request_lifecycle()
        try:
            await self._ensure_vram(model, estimated_vram_gb)
            try:
                await self._ensure_ready()
            except BaseException:
                self._discard_vram_pending(model)
                raise
            self._active += 1
            try:
                try:
                    url = f"{self._base}/v1/chat/completions"
                    headers = self._headers()
                    if request_data_logging_enabled():
                        log_data_event(
                            "backend_request",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="chat",
                            method="POST",
                            url=url,
                            headers=headers_from_mapping(headers),
                            body=body_from_json(body),
                        )
                    resp = await self._client.post(
                        url,
                        json=body,
                        headers=headers,
                    )
                    await resp.aread()
                    if request_data_logging_enabled():
                        log_data_event(
                            "backend_response_start",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="chat",
                            method="POST",
                            url=url,
                            status=resp.status_code,
                            headers=headers_from_mapping(dict(resp.headers)),
                        )
                        log_data_event(
                            "backend_response_body",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="chat",
                            method="POST",
                            url=url,
                            body=body_from_bytes(resp.content),
                        )
                        log_data_event(
                            "backend_response_end",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="chat",
                            method="POST",
                            url=url,
                            status=resp.status_code,
                            response_bytes=len(resp.content),
                        )
                    resp.raise_for_status()
                except BaseException as exc:
                    log_data_event(
                        "backend_error",
                        backend="llama.cpp",
                        target_id=self.target_id,
                        operation="chat",
                        method="POST",
                        url=f"{self._base}/v1/chat/completions",
                        error=f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
                    )
                    self._discard_vram_pending(model)
                    raise
                self._mark_vram_reserved(model, estimated_vram_gb)
                return resp.json()
            finally:
                self._active -= 1
        finally:
            self._touch_vram_reservation(model)
            self._end_request_lifecycle()

    async def stream_chat(
        self,
        payload: Dict[str, Any],
        *,
        estimated_vram_gb: Optional[float] = None,
    ) -> AsyncIterator[str]:
        body = dict(payload)
        body["stream"] = True
        model = str(body.get("model") or "")
        self._begin_request_lifecycle()
        try:
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
                    url = f"{self._base}/v1/chat/completions"
                    headers = self._headers(stream=True)
                    if request_data_logging_enabled():
                        log_data_event(
                            "backend_request",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="stream_chat",
                            method="POST",
                            url=url,
                            headers=headers_from_mapping(headers),
                            body=body_from_json(body),
                        )
                    response_started = False
                    response_bytes = 0
                    outcome = "complete"
                    error: Optional[str] = None
                    stream_done = False
                    async with self._client.stream(
                        "POST",
                        url,
                        json=body,
                        headers=headers,
                    ) as resp:
                        response_started = True
                        log_data_event(
                            "backend_response_start",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="stream_chat",
                            method="POST",
                            url=url,
                            status=resp.status_code,
                            headers=headers_from_mapping(dict(resp.headers)),
                        )
                        if resp.status_code >= 400:
                            error_body = await resp.aread()
                            response_bytes += len(error_body)
                            log_data_event(
                                "backend_response_body",
                                backend="llama.cpp",
                                target_id=self.target_id,
                                operation="stream_chat",
                                method="POST",
                                url=url,
                                body=body_from_bytes(error_body),
                            )
                            log_data_event(
                                "backend_response_end",
                                backend="llama.cpp",
                                target_id=self.target_id,
                                operation="stream_chat",
                                method="POST",
                                url=url,
                                status=resp.status_code,
                                outcome="exception",
                                response_bytes=response_bytes,
                                error=f"http status {resp.status_code}",
                            )
                            resp.raise_for_status()
                        self._mark_vram_reserved(model, estimated_vram_gb)
                        marked = True
                        try:
                            async for raw_line in resp.aiter_lines():
                                if raw_line:
                                    response_bytes += len(raw_line.encode("utf-8"))
                                    if raw_line.strip() == "data: [DONE]":
                                        stream_done = True
                                    log_data_event(
                                        "backend_response_body",
                                        backend="llama.cpp",
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
                                error = (
                                    f"{exc.__class__.__module__}."
                                    f"{exc.__class__.__name__}: {exc}"
                                )
                            raise
                        finally:
                            if response_started:
                                log_data_event(
                                    "backend_response_end",
                                    backend="llama.cpp",
                                    target_id=self.target_id,
                                    operation="stream_chat",
                                    method="POST",
                                    url=url,
                                    status=resp.status_code,
                                    outcome=outcome,
                                    response_bytes=response_bytes,
                                    error=error,
                                )
                except BaseException as exc:
                    if exc.__class__.__name__ != "GeneratorExit":
                        log_data_event(
                            "backend_error",
                            backend="llama.cpp",
                            target_id=self.target_id,
                            operation="stream_chat",
                            method="POST",
                            url=f"{self._base}/v1/chat/completions",
                            error=f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
                        )
                    if not marked:
                        self._discard_vram_pending(model)
                    raise
            finally:
                self._active -= 1
        finally:
            self._touch_vram_reservation(model)
            self._end_request_lifecycle()

    async def stop_if_idle(self) -> None:
        if not self._idle_timeout or self._active or self._request_refs:
            return
        idle_for = time.monotonic() - self._last_used
        if idle_for < self._idle_timeout:
            return
        if not (self._started_by_us or self._stop_command):
            return
        logger.info(
            "stopping idle llama.cpp target %s after %.1fs idle",
            self.target_id,
            idle_for,
        )
        await self.stop_if_owned()

    async def release_for_vram(self) -> bool:
        return await self._release_server_for_vram()

    async def _release_server_for_vram(self) -> bool:
        if self._active or self._request_refs:
            return False
        if not (self._started_by_us or self._stop_command):
            return False
        return await self.stop_if_owned()

    async def stop_if_owned(self) -> bool:
        if self._stop_command:
            logger.info("stopping llama.cpp target with stop_command")
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
            logger.warning(
                "llama.cpp target stop_command exited with status %s", returncode
            )
            return False

        if self._process is not None and self._started_by_us:
            if self._process.returncode is None:
                logger.info("terminating owned llama.cpp target process tree")
                if not await terminate_process_tree(self._process, timeout=10.0):
                    logger.warning("failed to terminate owned llama.cpp target process tree")
                    return False
                self._started_by_us = False
                self._clear_all_vram_state()
                return True
            logger.warning(
                "owned llama.cpp target launcher process already exited; cannot stop "
                "remaining detached server process without stop_command"
            )
            return False
        return False
