"""Protocol-neutral HTTP process lifecycle and resource leases.

Shares runtime_ownership, cancellation-safe cleanup and VRAM/RAM admission with
other providers. This class deliberately has no chat, slots, token, prompt or
llama-specific request methods. Decision transports use its public json_request
lease, so admission, queue cancellation, reload drain and process cleanup follow
the same ownership contracts as chat and image providers.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx
from . import runtime_ownership
from .config import outbound_cycle_headers
from .lifecycle import BackendShuttingDownError, finish_cleanup
from .privacy import privacy_mode_enabled
from .process_utils import (create_managed_subprocess_exec, create_managed_subprocess_shell,
                            process_tree_running, run_stop_command, terminate_process_tree)
from .vram import (VRAM_IDLE_RECLAIM_SECONDS, MemoryCoordinator, MemoryReleaseCandidate,
                   VramCoordinator, VramReleaseCandidate)

logger = logging.getLogger("fake_ollama")


def _quantile_seconds(values, q):
    return values[max(0, min(len(values)-1, math.ceil(q*len(values))-1))] if values else None


@dataclass
class _LoadedModel:
    model: str
    estimated_vram_gb: float
    last_used_monotonic: float
    estimated_memory_gb: float = 0.0


class ManagedHTTPRuntime:
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
        manage_external_runtime: bool = False,
        stop_timeout_seconds: float = 30.0,
        ownership_active: bool = True,
        idle_timeout_seconds: Optional[float] = None,
        startup_timeout_seconds: float = 120.0,
        health_path: str = "/health",
        cwd: Optional[str] = None,
        launch_env: Optional[Dict[str, str]] = None,
        target_name: str = "http",
        target_prefix: str = "http",
        target_log_label: str = "HTTP runtime",
        vram_coordinator: Optional[VramCoordinator] = None,
        memory_coordinator: Optional[MemoryCoordinator] = None,
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
        self._manage_external_runtime = manage_external_runtime
        self._stop_timeout = stop_timeout_seconds
        self._used_since_last_stop = False
        self._idle_timeout = idle_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._health_path = health_path if health_path.startswith("/") else "/" + health_path
        self._cwd = cwd
        self._launch_env = launch_env
        self._target_log_label = target_log_label
        self.target_id = f"{target_prefix}:{target_name}"
        self._vram_coordinator = vram_coordinator
        self._memory_coordinator = memory_coordinator
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            # Allow caller to override only the upstream *read* timeout, so a
            # request that is queued behind other in-flight requests on a
            # single-slot HTTP runtime server does not get killed by httpx's
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
        # Local concurrency gate: when the upstream HTTP runtime server only has
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
        # Rolling history of completed-queue wait times (wall_ts, wait_seconds).
        # Used by the dashboard to surface p50/p95 wait latency per target.
        self._wait_history: Deque[Tuple[float, float]] = deque(maxlen=200)
        # Currently waiting in the gate: pending_id -> t0 (monotonic). Lets
        # the dashboard report the longest still-waiting request.
        self._wait_pending: Dict[int, float] = {}
        self._wait_next_id = 1
        self._process: Optional[asyncio.subprocess.Process] = None
        self._started_by_us = False
        self._start_lock = asyncio.Lock()
        # Process and stop ownership is shared across clients of this runtime.
        self._active = 0
        self._request_refs = 0
        self._model_request_refs: dict[str, int] = {}
        self._starting_models: set[str] = set()
        self._last_used = time.monotonic()
        self._loaded_model: Optional[_LoadedModel] = None
        self._loaded_models: Dict[str, _LoadedModel] = {}
        self._reserved_vram_models: set[str] = set()
        self._reserved_memory_models: set[str] = set()
        self._shutdown_requested = False
        # Set when ``begin_shutdown`` is called so every request currently
        # blocked in ``_concurrency_slot`` wakes up at once and bails out,
        # instead of draining the queue one-by-one as the active request
        # finishes (or the upstream is killed and each next slot owner has
        # to discover the connection error). Lazily created on first use so
        # we don't bind to a loop at __init__ time.
        self._shutdown_event: Optional[asyncio.Event] = None
        self._ownership = runtime_ownership.register(self, group=getattr(self, "vram_runtime_group", None), active=ownership_active)
        self._start_lock = self._ownership.lock
        if self._vram_coordinator is not None:
            self._vram_coordinator.register(self)
        if self._memory_coordinator is not None:
            self._memory_coordinator.register(self)


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
        if self._shutdown_requested:
            raise BackendShuttingDownError(self.target_id)
        if sem is None:
            yield
            return
        self._queued += 1
        queued = True
        t0 = time.monotonic()
        pending_id = self._wait_next_id
        self._wait_next_id += 1
        self._wait_pending[pending_id] = t0
        shutdown_event = self._get_shutdown_event()
        acquire_task = asyncio.create_task(sem.acquire())
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        try:
            await asyncio.wait({acquire_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)
            if shutdown_event.is_set():
                raise BackendShuttingDownError(self.target_id)
            await acquire_task
            self._queued -= 1
            queued = False
            self._wait_pending.pop(pending_id, None)
            self._wait_history.append((time.time(), time.monotonic() - t0))
            yield
        finally:
            # The acquire task owns the permit until this join. Cancellation
            # after acquisition cannot erase that ownership or release twice.
            async def cleanup():
                nonlocal queued
                for task in (acquire_task, shutdown_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(acquire_task, shutdown_task, return_exceptions=True)
                if not acquire_task.cancelled() and acquire_task.exception() is None and acquire_task.result():
                    sem.release()
                if queued:
                    self._queued -= 1
                    queued = False
                self._wait_pending.pop(pending_id, None)
            await finish_cleanup(cleanup())


    def has_vram_reservation(self, model: str) -> bool:
        return model in self._reserved_vram_models


    def has_memory_reservation(self, model: str) -> bool:
        return model in self._reserved_memory_models


    def _begin_request_lifecycle(self) -> None:
        runtime_ownership.begin_request(self)
        self._used_since_last_stop = True
        self._request_refs += 1


    def _end_request_lifecycle(self) -> None:
        self._request_refs -= 1
        self._last_used = time.monotonic()


    def vram_release_candidates(self, *, now: float, idle_seconds: float) -> list[VramReleaseCandidate]:
        if self._active or self._request_refs or not self._ownership.can_manage():
            return []
        return [VramReleaseCandidate(owner_id=self.target_id, model=loaded.model,
                    estimated_vram_gb=loaded.estimated_vram_gb,
                    last_used_monotonic=loaded.last_used_monotonic, release=self._release_server_for_vram)
                for loaded in self._loaded_models.values()
                if loaded.estimated_vram_gb > 0 and now - loaded.last_used_monotonic >= idle_seconds]


    def vram_force_release_candidates(self, *, now: float) -> list[VramReleaseCandidate]:
        if not self._ownership.can_manage():
            return []
        return [VramReleaseCandidate(owner_id=self.target_id, model=loaded.model,
                    estimated_vram_gb=loaded.estimated_vram_gb,
                    last_used_monotonic=loaded.last_used_monotonic,
                    release=lambda: self._release_server_for_vram(force=True))
                for loaded in self._loaded_models.values() if loaded.estimated_vram_gb > 0]


    def memory_release_candidates(self, *, now: float, idle_seconds: float) -> list[MemoryReleaseCandidate]:
        if self._active or self._request_refs or not self._ownership.can_manage():
            return []
        return [MemoryReleaseCandidate(owner_id=self.target_id, model=loaded.model,
                    estimated_memory_gb=loaded.estimated_memory_gb,
                    last_used_monotonic=loaded.last_used_monotonic, release=self._release_server_for_vram)
                for loaded in self._loaded_models.values()
                if loaded.estimated_memory_gb > 0 and now - loaded.last_used_monotonic >= idle_seconds]


    def memory_force_release_candidates(self, *, now: float) -> list[MemoryReleaseCandidate]:
        if not self._ownership.can_manage():
            return []
        return [MemoryReleaseCandidate(owner_id=self.target_id, model=loaded.model,
                    estimated_memory_gb=loaded.estimated_memory_gb,
                    last_used_monotonic=loaded.last_used_monotonic,
                    release=lambda: self._release_server_for_vram(force=True))
                for loaded in self._loaded_models.values() if loaded.estimated_memory_gb > 0]


    def loaded_model_snapshots(self, *, now: Optional[float] = None,
                               idle_reclaim_seconds: float = VRAM_IDLE_RECLAIM_SECONDS) -> list[dict[str, object]]:
        now = time.monotonic() if now is None else now
        in_flight = bool(self._active or self._request_refs or self._queued)
        can_stop = bool(self._ownership.can_manage())
        snapshots = []
        for loaded in self._loaded_models.values():
            idle_seconds = 0.0 if in_flight else max(0.0, now - loaded.last_used_monotonic)
            snapshots.append({"backend": self.target_id.split(":", 1)[0], "target_id": self.target_id,
                "model": loaded.model, "estimated_vram_gb": loaded.estimated_vram_gb,
                "estimated_vram_mib": loaded.estimated_vram_gb * 1024.0,
                "estimated_memory_gb": loaded.estimated_memory_gb,
                "estimated_memory_mib": loaded.estimated_memory_gb * 1024.0,
                "active_requests": self._active, "queued_requests": self._queued,
                "request_refs": self._request_refs, "idle_seconds": idle_seconds,
                "loading": loaded.model in self._starting_models,
                "execution_unconfirmed": self._ownership.uncertain_decisions,
                "reclaimable": can_stop and not in_flight and idle_seconds >= idle_reclaim_seconds})
        return snapshots


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
        await runtime_ownership.close_client(self, terminate=terminate_process_tree, run_command=run_stop_command)


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


    async def _ensure_memory(
        self, model: Optional[str], estimated_memory_gb: Optional[float]
    ) -> None:
        if self._memory_coordinator is None or estimated_memory_gb is None:
            return
        await self._memory_coordinator.ensure_available(
            self,
            model=model or "",
            estimated_memory_gb=estimated_memory_gb,
        )


    def _mark_vram_reserved(self, model: Optional[str], estimated_vram_gb: Optional[float]) -> None:
        if not model or estimated_vram_gb is None:
            return
        self._used_since_last_stop = True
        loaded = self._loaded_models.setdefault(model, _LoadedModel(model, 0.0, time.monotonic()))
        loaded.estimated_vram_gb = max(loaded.estimated_vram_gb, estimated_vram_gb)
        loaded.last_used_monotonic = time.monotonic()
        self._loaded_model = loaded
        self._reserved_vram_models.add(model)
        if self._vram_coordinator is not None:
            self._vram_coordinator.confirm_loaded(self.target_id, model)


    def _mark_memory_reserved(self, model: Optional[str], estimated_memory_gb: Optional[float]) -> None:
        if not model or estimated_memory_gb is None:
            return
        loaded = self._loaded_models.setdefault(model, _LoadedModel(model, 0.0, time.monotonic()))
        loaded.estimated_memory_gb = max(loaded.estimated_memory_gb, estimated_memory_gb)
        self._loaded_model = loaded
        self._reserved_memory_models.add(model)
        if self._memory_coordinator is not None:
            self._memory_coordinator.confirm_loaded(self.target_id, model)


    def _touch_vram_reservation(self, model: Optional[str]) -> None:
        loaded = self._loaded_models.get(model)
        if loaded is not None:
            loaded.last_used_monotonic = time.monotonic()

    def _retain_startup_reservation(self, model, estimated_vram_gb, estimated_memory_gb):
        """An owned loader can still allocate after its triggering request ends.

        Keep the admission reservation unconfirmed until health succeeds or the
        owned process tree is stopped. A pending load must not age out as if the
        live memory measurement already included it.
        """
        loaded = self._loaded_models.setdefault(model, _LoadedModel(model, 0.0, time.monotonic()))
        self._loaded_model = loaded
        self._starting_models.add(model)
        if estimated_vram_gb is not None:
            loaded.estimated_vram_gb = max(loaded.estimated_vram_gb, estimated_vram_gb)
            self._reserved_vram_models.add(model)
        if estimated_memory_gb is not None:
            loaded.estimated_memory_gb = max(loaded.estimated_memory_gb, estimated_memory_gb)
            self._reserved_memory_models.add(model)


    def _discard_vram_pending(self, model: Optional[str]) -> None:
        if not model or self._vram_coordinator is None:
            return
        self._vram_coordinator.discard_pending(self.target_id, model)


    def _discard_memory_pending(self, model: Optional[str]) -> None:
        if not model or self._memory_coordinator is None:
            return
        self._memory_coordinator.discard_pending(self.target_id, model)


    def _clear_all_vram_state(self) -> None:
        for model in self._loaded_models:
            if self._vram_coordinator is not None:
                self._vram_coordinator.discard_pending(self.target_id, model)
            if self._memory_coordinator is not None:
                self._memory_coordinator.discard_pending(self.target_id, model)
        self._loaded_models.clear()
        self._starting_models.clear()
        self._reserved_vram_models.clear()
        self._reserved_memory_models.clear()
        self._loaded_model = None


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
        if self._shutdown_requested:
            raise httpx.ConnectError(
                "HTTP runtime target is unavailable and fake-ollama is shutting down; "
                "refusing auto-start"
            )
        if not (self._auto_start and (self._start_argv or self._start_command)):
            # External providers may not implement a health route. A missing
            # health URL is not evidence that their resident models vanished.
            return

        async with self._start_lock:
            if await self._healthy():
                return
            if self._shutdown_requested:
                raise httpx.ConnectError(
                    "HTTP runtime target is unavailable and fake-ollama is shutting down; "
                    "refusing auto-start"
                )
            await runtime_ownership.discard_finished_process(self, inspect=process_tree_running, terminate=terminate_process_tree)
            if self._ownership.process is None and not self._started_by_us:
                # Verified process death clears old residency before admission.
                # Do not erase this request's new startup reservation here.
                await self.prepare_start()
                stderr_log = self._stderr_log_path()
                self._stderr_start_offset = stderr_log.stat().st_size if stderr_log is not None and stderr_log.exists() else 0
                if stderr_log is not None:
                    logger.info(
                        "[%s] capturing HTTP runtime stderr to %s",
                        self.target_id,
                        stderr_log,
                    )
                try:
                    if self._start_argv:
                        logger.info("starting HTTP runtime target %s with configured argv", self.target_id)
                        self._process = await create_managed_subprocess_exec(
                            self._start_argv,
                            cwd=self._cwd,
                            env=self._launch_env,
                            stderr=stderr_log,
                        )
                    else:
                        logger.info("starting HTTP runtime target %s with configured command", self.target_id)
                        self._process = await create_managed_subprocess_shell(
                            self._start_command,
                            cwd=self._cwd,
                            env=self._launch_env,
                            stderr=stderr_log,
                        )
                except (OSError, ValueError) as exc:
                    raise httpx.ConnectError(
                        f"Could not start {self.target_id}: {exc}. Install the runtime explicitly "
                        "and check binary_path/python_path and local model_path; no weights are downloaded."
                    ) from exc
                self._started_by_us = True
                self._ownership.capture(self)

            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if await self._healthy():
                    logger.info("HTTP runtime target ready at %s", self._base)
                    return
                # Surface early subprocess death instead of silently waiting
                # the full startup_timeout. Common causes: binary_path points
                # to a directory, missing GGUF, port already bound, etc.
                proc = self._process
                if proc is not None and proc.returncode is not None and proc.returncode != 0:
                    rc = proc.returncode
                    # Retain process-tree ownership even after launcher failure.
                    raise httpx.ConnectError(
                        f"HTTP runtime target process exited with code {rc} before "
                        f"becoming healthy. {self._startup_error_detail()}"
                    )
                if self._shutdown_requested:
                    raise httpx.ConnectError(
                        "fake-ollama is shutting down; aborting HTTP runtime startup wait"
                    )
                await asyncio.sleep(1.0)
            raise httpx.ConnectError(
                f"HTTP runtime target did not become healthy within {self._startup_timeout}s"
            )


    async def stop_if_idle(self) -> None:
        if not self._idle_timeout or self._ownership.busy():
            return
        # A successful stop_command clears _loaded_model and sets
        # _started_by_us=False. Do not keep re-running stop_command forever
        # just because it is configured on the target.
        if self._loaded_model is None and not self._started_by_us:
            return
        last_used = max([self._last_used, *[loaded.last_used_monotonic for loaded in self._loaded_models.values()]])
        idle_for = time.monotonic() - max(last_used, *(m._last_used for m in self._ownership.members))
        if idle_for < self._idle_timeout:
            return
        if not (self._ownership.can_manage()):
            return
        logger.info(
            "stopping idle %s target %s after %.1fs idle",
            self._target_log_label,
            self.target_id,
            idle_for,
        )
        await self.stop_if_owned()


    async def release_for_vram(self) -> bool:
        return await self._release_server_for_vram()


    async def _release_server_for_vram(self, *, force: bool = False) -> bool:
        if not force and (self._active or self._request_refs):
            return False
        if not (self._ownership.can_manage()):
            return False
        return await self.stop_if_owned()


    async def stop_if_owned(self) -> bool:
        return await runtime_ownership.stop_owned(self, terminate=terminate_process_tree, run_command=run_stop_command)


    def queue_wait_metrics(
        self,
        *,
        now_monotonic: Optional[float] = None,
        now_wall: Optional[float] = None,
        history_window_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """Snapshot of queue depth and recent wait-time percentiles.

        The dashboard pairs this with the access log to distinguish "upstream
        is slow" (active high, queue empty) from "we are admission-gated"
        (queue grows, p95 wait climbs into the read-timeout window).
        """
        now_m = time.monotonic() if now_monotonic is None else now_monotonic
        now_w = time.time() if now_wall is None else now_wall
        cutoff = now_w - max(1.0, float(history_window_seconds))
        recent = sorted(s for ts, s in self._wait_history if ts >= cutoff)
        if self._wait_pending:
            pending_max: Optional[float] = max(
                now_m - t0 for t0 in self._wait_pending.values()
            )
        else:
            pending_max = None
        return {
            "target_id": self.target_id,
            "active": self._active,
            "queued": self._queued,
            "capacity": self._max_concurrent_requests or 0,
            "history_window_seconds": float(history_window_seconds),
            "history_size": len(recent),
            "p50_seconds": _quantile_seconds(recent, 0.5),
            "p95_seconds": _quantile_seconds(recent, 0.95),
            "max_seconds": recent[-1] if recent else None,
            "pending_max_seconds": pending_max,
        }


    async def prepare_start(self) -> None:
        """Validate the configured launcher immediately before process start."""

    def _stderr_log_path(self) -> Optional[Path]:
        if privacy_mode_enabled():
            return None
        import re
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.target_id)
        return Path("logs") / f"{name}.err.log"

    def _startup_error_detail(self) -> str:
        path = self._stderr_log_path()
        if privacy_mode_enabled():
            return (
                "Check the configured executable and local model_path; "
                "stderr capture is disabled in privacy mode."
            )
        if path is not None:
            try:
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    stream.seek(max(getattr(self, "_stderr_start_offset", 0), stream.tell() - 4096))
                    lines = stream.read(4096).decode("utf-8", errors="replace").splitlines()
                for line in reversed(lines):
                    # The optional companion's own diagnostic is safe and
                    # actionable. Do not echo arbitrary argv or backend logs
                    # (they may contain operator-supplied secrets).
                    if line.startswith("StartLux setup error:"):
                        return line[:1500]
            except OSError:
                pass
        return f"Check the configured executable, local model_path, and runtime stderr log {path}."

    @asynccontextmanager
    async def request_lease(self, model: str, *, estimated_vram_gb=None, estimated_memory_gb=None):
        """Pin runtime ownership and resource admission until response cleanup."""
        self._begin_request_lifecycle()
        self._model_request_refs[model] = self._model_request_refs.get(model, 0) + 1
        starting = ready = False
        try:
            # A confirmed dead owned process cannot supply sibling residency
            # for admission. Launcher exit alone is not proof: descendants may
            # still be alive, so use the shared process-tree verifier first.
            self._ownership.can_manage()
            process = self._ownership.process
            if process is not None and process.returncode is not None:
                async with self._ownership.lock:
                    await runtime_ownership.discard_finished_process(
                        self, inspect=process_tree_running, terminate=terminate_process_tree)
            await self._ensure_vram(model, estimated_vram_gb)
            await self._ensure_memory(model, estimated_memory_gb)
            starting = True
            await self._ensure_ready()
            self._mark_vram_reserved(model, estimated_vram_gb)
            self._mark_memory_reserved(model, estimated_memory_gb)
            self._starting_models.discard(model)
            ready = True
            async with self._concurrency_slot():
                # An earlier request can lose its transport while this one waits.
                runtime_ownership.begin_request(self)
                self._active += 1
                try:
                    yield
                finally:
                    self._active -= 1
        finally:
            remaining = self._model_request_refs[model] - 1
            if remaining:
                self._model_request_refs[model] = remaining
            else:
                del self._model_request_refs[model]
            if not ready:
                if starting and self._ownership.process is not None:
                    self._retain_startup_reservation(model, estimated_vram_gb, estimated_memory_gb)
                elif not remaining:
                    self._discard_vram_pending(model)
                    self._discard_memory_pending(model)
            self._touch_vram_reservation(model)
            self._end_request_lifecycle()

    async def post_until_complete(self, path: str, payload: dict, *, headers=None):
        """Keep the caller's lease while a submitted, non-cancellable POST drains.

        Closing a socket is not an inference cancellation acknowledgement. Drain
        the response for the configured cleanup budget despite repeated caller
        cancellations. Transport loss or expiry leaves a shared quarantine until
        verified runtime cleanup, not a free slot for another inference.
        """
        async def send():
            try:
                return await self._client.post(self._base + path, json=payload,
                    headers={**self._headers(), **(headers or {})})
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                # No upstream request was sent.
                raise
            except (httpx.RequestError, asyncio.CancelledError):
                self._ownership.uncertain_decisions = True
                raise

        task = asyncio.create_task(send())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            async def drain():
                done, _ = await asyncio.wait({task}, timeout=self._stop_timeout)
                if not done:
                    self._ownership.uncertain_decisions = True
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    logger.warning("[%s] decision drain expired; execution unconfirmed", self.target_id)
                else:
                    task.result()
            try:
                await finish_cleanup(drain())
            except Exception:
                # The original caller is gone; send() already retained any
                # uncertain execution on the shared runtime owner.
                logger.warning("[%s] cancelled request drain failed", self.target_id)
            raise

    async def json_request(self, path: str, payload: dict, *, model: str,
                           estimated_vram_gb=None, estimated_memory_gb=None, headers=None):
        async with self.request_lease(model, estimated_vram_gb=estimated_vram_gb,
                                      estimated_memory_gb=estimated_memory_gb):
            response = await self.post_until_complete(path, payload, headers=headers)
            response.raise_for_status()
            return response.json()
