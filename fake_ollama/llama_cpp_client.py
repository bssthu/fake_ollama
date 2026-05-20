"""Async client for llama.cpp's OpenAI-compatible server.

Used by the reverse proxy to expose llama.cpp models on the same external
surface as Ollama targets. The client also owns optional lifecycle management:
health-check, auto-start, and idle stop for processes it starts itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .config import outbound_cycle_headers
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
        max_concurrent_requests: Optional[int] = None,
        request_read_timeout_seconds: Optional[float] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._trust_env = trust_env
        self._request_read_timeout_seconds = request_read_timeout_seconds
        self._max_concurrent_requests = max_concurrent_requests
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
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            # Allow caller to override only the upstream *read* timeout, so a
            # request that is queued behind other in-flight requests on a
            # single-slot llama.cpp server does not get killed by httpx's
            # default 5-tuple timeout. A negative or zero value disables the
            # read timeout entirely (httpx None == no timeout).
            if request_read_timeout_seconds is None:
                httpx_timeout: httpx.Timeout = httpx.Timeout(timeout)
            else:
                read_t = (
                    None
                    if request_read_timeout_seconds <= 0
                    else float(request_read_timeout_seconds)
                )
                httpx_timeout = httpx.Timeout(timeout, read=read_t)
            self._client = httpx.AsyncClient(
                timeout=httpx_timeout, trust_env=trust_env
            )
            self._owns_client = True
        # Local concurrency gate: when the upstream llama.cpp server only has
        # a small number of decoding slots (typically ``--parallel N``), we
        # serialise extra requests in fake_ollama instead of letting them all
        # pile up on the upstream TCP socket where they would otherwise time
        # out. ``None`` keeps the original unbounded behaviour.
        if max_concurrent_requests is not None and max_concurrent_requests > 0:
            self._request_semaphore: Optional[asyncio.Semaphore] = asyncio.Semaphore(
                max_concurrent_requests
            )
        else:
            self._request_semaphore = None
        self._queued = 0
        self._process: Optional[asyncio.subprocess.Process] = None
        self._started_by_us = False
        self._start_lock = asyncio.Lock()
        self._active = 0
        self._request_refs = 0
        self._last_used = time.monotonic()
        self._loaded_model: Optional[_LoadedModel] = None
        self._shutdown_requested = False
        # Set when ``begin_shutdown`` is called so every request currently
        # blocked in ``_concurrency_slot`` wakes up at once and bails out,
        # instead of draining the queue one-by-one as the active request
        # finishes (or the upstream is killed and each next slot owner has
        # to discover the connection error). Lazily created on first use so
        # we don't bind to a loop at __init__ time.
        self._shutdown_event: Optional[asyncio.Event] = None
        if self._vram_coordinator is not None:
            self._vram_coordinator.register(self)

    @property
    def idle_timeout_seconds(self) -> Optional[float]:
        return self._idle_timeout

    @property
    def active_requests(self) -> int:
        return self._active

    @property
    def queued_requests(self) -> int:
        return self._queued

    @property
    def last_used_monotonic(self) -> float:
        return self._last_used

    def _get_shutdown_event(self) -> asyncio.Event:
        ev = self._shutdown_event
        if ev is None:
            ev = asyncio.Event()
            if self._shutdown_requested:
                ev.set()
            self._shutdown_event = ev
        return ev

    @asynccontextmanager
    async def _concurrency_slot(self):
        sem = self._request_semaphore
        if sem is None:
            if self._shutdown_requested:
                raise asyncio.CancelledError(
                    f"{self.target_id} shutting down"
                )
            yield
            return
        if self._shutdown_requested:
            raise asyncio.CancelledError(
                f"{self.target_id} shutting down"
            )
        # Fast path: if a slot is immediately available, don't bother logging
        # — this is the common case under low load and we want zero noise.
        if sem.locked() or self._queued > 0:
            logger.info(
                "[%s] queueing request (active=%d queued=%d -> %d, cap=%d)",
                self.target_id,
                self._active,
                self._queued,
                self._queued + 1,
                self._max_concurrent_requests or 0,
            )
            queued_log = True
        else:
            queued_log = False
        self._queued += 1
        t0 = time.monotonic()
        shutdown_event = self._get_shutdown_event()
        acquire_task = asyncio.ensure_future(sem.acquire())
        shutdown_task = asyncio.ensure_future(shutdown_event.wait())
        try:
            try:
                done, _pending = await asyncio.wait(
                    {acquire_task, shutdown_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                acquire_task.cancel()
                shutdown_task.cancel()
                self._queued -= 1
                if queued_log:
                    logger.info(
                        "[%s] queue exit (cancelled before slot, waited=%.2fs, queued=%d)",
                        self.target_id,
                        time.monotonic() - t0,
                        self._queued,
                    )
                raise
            if acquire_task in done and not acquire_task.cancelled():
                # We won the race. If shutdown also fired we still hand the
                # slot back immediately — no point starting new upstream work.
                shutdown_task.cancel()
                if shutdown_event.is_set():
                    sem.release()
                    self._queued -= 1
                    if queued_log:
                        logger.info(
                            "[%s] queue exit (shutdown after slot, waited=%.2fs, queued=%d)",
                            self.target_id,
                            time.monotonic() - t0,
                            self._queued,
                        )
                    raise asyncio.CancelledError(
                        f"{self.target_id} shutting down"
                    )
            else:
                # shutdown won. Cancel the pending acquire; if it already
                # acquired between wait() returning and us cancelling, give
                # the slot back so the next waiter (also being cancelled)
                # doesn't get blocked.
                if acquire_task.cancel() is False and not acquire_task.cancelled():
                    try:
                        await acquire_task
                    except BaseException:
                        pass
                    else:
                        sem.release()
                self._queued -= 1
                if queued_log:
                    logger.info(
                        "[%s] queue exit (shutdown, waited=%.2fs, queued=%d)",
                        self.target_id,
                        time.monotonic() - t0,
                        self._queued,
                    )
                raise asyncio.CancelledError(
                    f"{self.target_id} shutting down"
                )
        finally:
            if not shutdown_task.done():
                shutdown_task.cancel()
        self._queued -= 1
        if queued_log:
            logger.info(
                "[%s] slot acquired (waited=%.2fs, active=%d queued=%d)",
                self.target_id,
                time.monotonic() - t0,
                self._active + 1,
                self._queued,
            )
        try:
            yield
        finally:
            sem.release()
            if queued_log:
                logger.info(
                    "[%s] slot released (active=%d queued=%d)",
                    self.target_id,
                    self._active,
                    self._queued,
                )

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
        # ``last_used_monotonic`` is only refreshed when a request *finishes*
        # (via ``_touch_vram_reservation``). For long-running streams or while
        # requests sit in the local queue the model is obviously not idle, so
        # we report 0 idle whenever there is any in-flight or queued work —
        # otherwise the dashboard misleadingly shows a busy model as idle.
        in_flight = bool(self._active or self._request_refs or self._queued)
        if in_flight:
            idle_seconds = 0.0
        else:
            idle_seconds = max(0.0, now - loaded.last_used_monotonic)
        can_stop = bool(self._started_by_us or self._stop_command)
        return [
            {
                "backend": "llama.cpp",
                "target_id": self.target_id,
                "model": loaded.model,
                "estimated_vram_gb": loaded.estimated_vram_gb,
                "estimated_vram_mib": loaded.estimated_vram_gb * 1024.0,
                "active_requests": self._active,
                "queued_requests": self._queued,
                "request_refs": self._request_refs,
                "idle_seconds": idle_seconds,
                "reclaimable": (
                    can_stop and not in_flight and idle_seconds >= idle_reclaim_seconds
                ),
            }
        ]

    def begin_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        queued_before = self._queued
        ev = self._shutdown_event
        if ev is not None:
            ev.set()
        if queued_before:
            logger.info(
                "[%s] shutdown requested; releasing %d queued request(s) at once",
                self.target_id,
                queued_before,
            )

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if stream:
            headers["accept"] = "text/event-stream"
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
            headers["x-api-key"] = self._auth_token
        headers.update(outbound_cycle_headers())
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
            async with self._concurrency_slot():
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
            async with self._concurrency_slot():
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
