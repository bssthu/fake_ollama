"""Shared admission control for local model targets.

Two resources are gated the same way:

* **GPU VRAM** — observed via ``nvidia-smi`` (:class:`VramCoordinator`).
* **System RAM** — observed via the OS (:class:`MemoryCoordinator`). Some
  local models (e.g. ComfyUI workflows that offload part of the graph to the
  CPU) need a large chunk of host memory in addition to VRAM, so host memory
  has to be admission-controlled too.

Both coordinators share the same algorithm (:class:`_ResourceCoordinator`),
which tracks two things:

1. **Reservations** — recorded by participants once the resource use has
   actually been observed. These are persistent and used purely to decide
   which models are reclaimable when a new request would otherwise exceed the
   budget.
2. **Pending acquisitions** — recorded by ``ensure_available`` *before* the
   model starts loading. Pending entries are subtracted from the live reading
   so two concurrent requests for two different models cannot both pass
   admission against the same headroom. After ``confirm_loaded`` the entry is
   kept for a short grace window (until the live reading catches up) and then
   dropped.

The coordinator's lock also serialises start-up admission: while the very
first request for ``(target, model)`` is between ``ensure_available`` and
``confirm_loaded``, subsequent ``ensure_available`` calls for the same key
short-circuit (they observe the existing pending entry) and proceed straight
into the per-client startup lock, which queues them behind the in-flight
start.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

import httpx

logger = logging.getLogger("fake_ollama")

VRAM_IDLE_RECLAIM_SECONDS = 60.0
# Pending bookkeeping is meant to absorb the brief gap between
# ``confirm_loaded`` and the live reading reflecting the new allocation
# (typically well under a second on Windows). Anything longer
# double-counts the model — once the live reading has caught up, the
# pending entry would subtract resource the live reading already accounts
# for, falsely reporting "0 GiB available" to the next admission.
PENDING_LOADED_GRACE_SECONDS = 2.0
_NVIDIA_SMI_TIMEOUT_SECONDS = 3.0
_POST_RELEASE_REFRESH_DELAYS_SECONDS = (0.0, 0.5, 1.0, 2.0)


class LocalTargetResourceError(httpx.ConnectError):
    """HTTP-ish error raised before forwarding to a local model target."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 503,
        error_type: str = "overloaded_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


ReleaseCallback = Callable[[], Awaitable[bool]]
InterruptCallback = Callable[[], Awaitable[bool]]


@dataclass
class VramReleaseCandidate:
    owner_id: str
    model: str
    estimated_vram_gb: float
    last_used_monotonic: float
    release: ReleaseCallback

    @property
    def estimated_gb(self) -> float:
        return self.estimated_vram_gb


@dataclass
class MemoryReleaseCandidate:
    owner_id: str
    model: str
    estimated_memory_gb: float
    last_used_monotonic: float
    release: ReleaseCallback

    @property
    def estimated_gb(self) -> float:
        return self.estimated_memory_gb


class VramParticipant(Protocol):
    target_id: str

    @property
    def active_requests(self) -> int: ...

    def has_vram_reservation(self, model: str) -> bool: ...

    def vram_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[VramReleaseCandidate]: ...


class MemoryParticipant(Protocol):
    target_id: str

    @property
    def active_requests(self) -> int: ...

    def has_memory_reservation(self, model: str) -> bool: ...

    def memory_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[MemoryReleaseCandidate]: ...


ResourceProvider = Callable[[], Awaitable[Optional[float]]]
# Backwards-compatible alias (the original name predates host-memory support).
VramProvider = ResourceProvider


def _gb_to_mib(value: float) -> float:
    return float(value) * 1024.0


def _fmt_gib(mib: float) -> str:
    if not math.isfinite(mib):
        return "unknown"
    return f"{mib / 1024.0:.2f} GiB"


@dataclass
class _PendingEntry:
    estimated_gb: float
    acquired_at: float
    confirmed_loaded_at: Optional[float] = None


@dataclass
class _ExecutionLeaseState:
    lease_id: int
    owner_id: str
    runtime_group: str
    model: str
    workload_key: str
    reserved_headroom_mib: float
    min_free_mib: float
    exclusive: bool
    resident_at_start: bool
    baseline_free_mib: float
    minimum_free_mib: float
    interrupt: Optional[InterruptCallback]
    cleanup: Optional[ReleaseCallback]
    monitor_task: Optional[asyncio.Task[None]] = None
    interrupted: bool = False
    breached: bool = False


class VramExecutionLease:
    """A request-duration GPU lease returned by :class:`VramCoordinator`.

    Admission and model residency are deliberately separate.  A resident model
    may have zero *load* cost while its next inference still needs several GiB
    of transient workspace.  This lease keeps that request headroom accounted
    for until the workflow has fully completed.
    """

    def __init__(
        self,
        coordinator: "VramCoordinator",
        state: _ExecutionLeaseState,
        *,
        cleaned_for_headroom: bool,
    ) -> None:
        self._coordinator = coordinator
        self._state = state
        self.cleaned_for_headroom = cleaned_for_headroom
        self._released = False

    @property
    def minimum_free_mib(self) -> float:
        return self._state.minimum_free_mib

    @property
    def breached(self) -> bool:
        return self._state.breached

    async def __aenter__(self) -> "VramExecutionLease":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self._released:
            self._released = True
            await self._coordinator.release_execution(self._state.lease_id)


# ---------------------------------------------------------------------------
# Live readings
# ---------------------------------------------------------------------------


async def nvidia_smi_free_vram_mib() -> Optional[float]:
    """Return free MiB on the most constrained NVIDIA GPU, if available.

    Model profiles describe one device's capacity. Summing multiple GPUs would
    let a job pass admission even when no individual card can hold it, so the
    shared default governor deliberately uses the minimum device reading.
    """
    return await _nvidia_smi_query("memory.free")


async def nvidia_smi_total_vram_mib() -> Optional[float]:
    """Return capacity of the smallest NVIDIA GPU in MiB, if available."""
    return await _nvidia_smi_query("memory.total")


async def _nvidia_smi_query(field_name: str) -> Optional[float]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--query-gpu={field_name}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        return None

    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_NVIDIA_SMI_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    if proc.returncode != 0:
        return None

    values: list[float] = []
    for raw_line in stdout.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            values.append(float(line.split()[0]))
        except (ValueError, IndexError):
            continue
    return min(values) if values else None


def system_memory_status_mib() -> tuple[Optional[float], Optional[float]]:
    """Return ``(available, total)`` host RAM in MiB, or ``(None, None)``."""
    if os.name == "nt":
        return _windows_memory_status_mib()
    proc = _proc_meminfo_status_mib()
    if proc != (None, None):
        return proc
    return _posix_memory_status_mib()


def _windows_memory_status_mib() -> tuple[Optional[float], Optional[float]]:
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        if not ok:
            return None, None
        return stat.ullAvailPhys / 1048576.0, stat.ullTotalPhys / 1048576.0
    except Exception:
        return None, None


def _proc_meminfo_status_mib() -> tuple[Optional[float], Optional[float]]:
    try:
        values: dict[str, float] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key not in ("MemAvailable", "MemFree", "MemTotal"):
                    continue
                raw = rest.strip().split()
                if raw:
                    values[key] = float(raw[0]) / 1024.0
        available = values.get("MemAvailable", values.get("MemFree"))
        total = values.get("MemTotal")
        return available, total
    except (OSError, ValueError):
        return None, None


def _posix_memory_status_mib() -> tuple[Optional[float], Optional[float]]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = (pages * page_size) / 1048576.0
        return None, total
    except (OSError, ValueError, AttributeError):
        return None, None


async def system_free_memory_mib() -> Optional[float]:
    """Return available host RAM in MiB, or None if it cannot be read."""
    return system_memory_status_mib()[0]


async def system_total_memory_mib() -> Optional[float]:
    """Return total host RAM (capacity) in MiB, or None if it cannot be read."""
    return system_memory_status_mib()[1]


# ---------------------------------------------------------------------------
# Resource specs: the per-resource bits the generic algorithm needs
# ---------------------------------------------------------------------------


def _vram_force_candidates(participant: Any, now: float) -> list:
    getter = getattr(participant, "vram_force_release_candidates", None)
    if callable(getter):
        return getter(now=now)
    return participant.vram_release_candidates(now=now, idle_seconds=0.0)


def _memory_force_candidates(participant: Any, now: float) -> list:
    getter = getattr(participant, "memory_force_release_candidates", None)
    if callable(getter):
        return getter(now=now)
    return participant.memory_release_candidates(now=now, idle_seconds=0.0)


@dataclass(frozen=True)
class _ResourceSpec:
    label: str  # human label for log/error messages, e.g. "GPU VRAM"
    amount_key: str  # result-dict key, e.g. "estimated_vram_gb"
    unavailable_hint: str  # remediation hint when the live reading is missing
    reservation: Callable[[Any, str], bool]
    release_candidates: Callable[[Any, float, float], list]
    force_candidates: Callable[[Any, float], list]


_VRAM_SPEC = _ResourceSpec(
    label="GPU VRAM",
    amount_key="estimated_vram_gb",
    unavailable_hint=(
        "Install nvidia-smi or unset model_profiles.<model>.estimated_vram_gb."
    ),
    reservation=lambda p, model: p.has_vram_reservation(model),
    release_candidates=lambda p, now, idle: p.vram_release_candidates(
        now=now, idle_seconds=idle
    ),
    force_candidates=_vram_force_candidates,
)


_MEMORY_SPEC = _ResourceSpec(
    label="system RAM",
    amount_key="estimated_memory_gb",
    unavailable_hint="Unset model_profiles.<model>.estimated_memory_gb.",
    reservation=lambda p, model: p.has_memory_reservation(model),
    release_candidates=lambda p, now, idle: p.memory_release_candidates(
        now=now, idle_seconds=idle
    ),
    force_candidates=_memory_force_candidates,
)


class _ResourceCoordinator:
    """Coordinate resource checks, startup queueing, and idle eviction."""

    def __init__(
        self,
        provider: ResourceProvider,
        *,
        total_provider: ResourceProvider,
        spec: _ResourceSpec,
    ) -> None:
        self._provider = provider
        self._total_provider = total_provider
        self._spec = spec
        self._cached_total_mib: Optional[float] = None
        self._participants: dict[str, Any] = {}
        self._pending: dict[tuple[str, str], _PendingEntry] = {}
        self._lock = asyncio.Lock()

    def register(self, participant: Any) -> None:
        self._participants[participant.target_id] = participant

    def unregister(self, participant: Any) -> None:
        self._participants.pop(participant.target_id, None)

    async def _free_mib(self) -> Optional[float]:
        return await self._provider()

    async def _total_mib(self) -> Optional[float]:
        if self._cached_total_mib is not None:
            return self._cached_total_mib
        total = await self._total_provider()
        if total is not None:
            self._cached_total_mib = total
        return total

    # -- Pending reservation API ---------------------------------------

    def has_pending(self, target_id: str, model: str) -> bool:
        return (target_id, model) in self._pending

    def discard_pending(self, target_id: str, model: str) -> None:
        self._pending.pop((target_id, model), None)

    def confirm_loaded(self, target_id: str, model: str) -> None:
        """Mark a pending reservation as actually loaded.

        The entry is kept for a short grace window so concurrent admissions
        keep subtracting it from the live reading until it catches up to the
        new allocation.
        """
        entry = self._pending.get((target_id, model))
        if entry is None:
            return
        if entry.confirmed_loaded_at is None:
            entry.confirmed_loaded_at = time.monotonic()

    async def _ensure_available(
        self,
        requester: Any,
        *,
        model: str = "",
        estimated_gb: Optional[float] = None,
    ) -> None:
        if estimated_gb is None:
            return
        key = (requester.target_id, model)
        required_mib = _gb_to_mib(estimated_gb)
        async with self._lock:
            # Idempotent: same target+model already pending or already loaded.
            if key in self._pending:
                return
            if self._spec.reservation(requester, model):
                return

            available_mib = await self._provider()
            if available_mib is None:
                raise LocalTargetResourceError(
                    f"Unable to determine available {self._spec.label} for local "
                    f"model '{model or requester.target_id}'. "
                    f"{self._spec.unavailable_hint}"
                )
            effective_free = self._effective_free_mib(available_mib, exclude_key=key)
            if effective_free >= required_mib:
                self._record_pending(key, estimated_gb)
                return

            candidates = self._eligible_candidates(model)
            total_reclaimable_mib = sum(
                _gb_to_mib(candidate.estimated_gb) for candidate in candidates
            )
            if effective_free + total_reclaimable_mib < required_mib:
                total_mib = await self._total_mib()
                if (
                    total_mib is None
                    or total_mib < required_mib
                    or not candidates
                ):
                    raise self._insufficient_error(
                        requester,
                        model=model,
                        required_mib=required_mib,
                        available_mib=effective_free,
                        reclaimable_mib=total_reclaimable_mib,
                    )
                logger.info(
                    "local model %s needs %s; current effective free plus estimated "
                    "idle reclaim is only %s, but total %s is %s, so trying "
                    "best-effort idle model release before failing admission",
                    model or requester.target_id,
                    _fmt_gib(required_mib),
                    _fmt_gib(effective_free + total_reclaimable_mib),
                    self._spec.label,
                    _fmt_gib(total_mib),
                )

            current_effective_free = effective_free
            attempted_release_mib = 0.0
            remaining_reclaimable_mib = total_reclaimable_mib
            for candidate in candidates:
                released = await candidate.release()
                candidate_mib = _gb_to_mib(candidate.estimated_gb)
                remaining_reclaimable_mib = max(
                    0.0, remaining_reclaimable_mib - candidate_mib
                )
                if released:
                    # Drop any pending entry for the released model so it
                    # stops counting against the budget.
                    self._pending.pop(
                        (candidate.owner_id, candidate.model), None
                    )
                    attempted_release_mib += candidate_mib
                    logger.info(
                        "requested release of idle local model %s on %s; estimated %s %.2f GiB",
                        candidate.model,
                        candidate.owner_id,
                        self._spec.label,
                        candidate.estimated_gb,
                    )
                    refreshed_mib = await self._wait_for_available_after_release(
                        required_mib, exclude_key=key
                    )
                    if refreshed_mib is not None:
                        current_effective_free = self._effective_free_mib(
                            refreshed_mib, exclude_key=key
                        )
                    if current_effective_free >= required_mib:
                        logger.info(
                            "current effective free %s after release is %s; proceeding with local model %s",
                            self._spec.label,
                            _fmt_gib(current_effective_free),
                            model or requester.target_id,
                        )
                        self._record_pending(key, estimated_gb)
                        return
                else:
                    logger.warning(
                        "failed to release idle local model %s on %s",
                        candidate.model,
                        candidate.owner_id,
                    )

                refreshed_mib = await self._provider()
                if refreshed_mib is not None:
                    current_effective_free = self._effective_free_mib(
                        refreshed_mib, exclude_key=key
                    )
                if current_effective_free >= required_mib:
                    self._record_pending(key, estimated_gb)
                    return

            raise self._insufficient_error(
                requester,
                model=model,
                required_mib=required_mib,
                available_mib=current_effective_free,
                reclaimable_mib=remaining_reclaimable_mib,
                attempted_release_mib=attempted_release_mib,
            )

    async def reclaim_if_below(
        self,
        *,
        threshold_mib: float,
        idle_seconds: float = VRAM_IDLE_RECLAIM_SECONDS,
    ) -> dict[str, object]:
        """Release eligible idle local models when free resource is critically low."""
        threshold_mib = max(0.0, float(threshold_mib))
        async with self._lock:
            available_mib = await self._provider()
            if available_mib is None:
                return {
                    "available_mib": None,
                    "threshold_mib": threshold_mib,
                    "released": [],
                    "checked": False,
                }
            if available_mib >= threshold_mib:
                return {
                    "available_mib": available_mib,
                    "threshold_mib": threshold_mib,
                    "released": [],
                    "checked": True,
                }

            candidates = self._eligible_candidates(
                "", idle_seconds=idle_seconds, include_same_model=True
            )
            released_models: list[dict[str, object]] = []
            current_mib = available_mib
            for candidate in candidates:
                released = await candidate.release()
                if released:
                    self._pending.pop((candidate.owner_id, candidate.model), None)
                    released_models.append(
                        {
                            "owner_id": candidate.owner_id,
                            "model": candidate.model,
                            self._spec.amount_key: candidate.estimated_gb,
                        }
                    )
                    logger.info(
                        "low free %s (%s below threshold %s); requested release of idle local model %s on %s",
                        self._spec.label,
                        _fmt_gib(current_mib),
                        _fmt_gib(threshold_mib),
                        candidate.model,
                        candidate.owner_id,
                    )
                    refreshed = await self._wait_for_raw_available_after_release(
                        threshold_mib
                    )
                    if refreshed is not None:
                        current_mib = refreshed
                    if current_mib >= threshold_mib:
                        break
                else:
                    logger.warning(
                        "failed to release idle local model %s on %s during low-resource check",
                        candidate.model,
                        candidate.owner_id,
                    )
                    refreshed = await self._provider()
                    if refreshed is not None:
                        current_mib = refreshed
                    if current_mib >= threshold_mib:
                        break

            return {
                "available_mib": current_mib,
                "threshold_mib": threshold_mib,
                "released": released_models,
                "checked": True,
            }

    async def reclaim_model(
        self,
        *,
        target_id: str,
        model: str,
        idle_seconds: float = VRAM_IDLE_RECLAIM_SECONDS,
        force: bool = False,
    ) -> dict[str, object]:
        """Release one local model by target/model identity.

        Normal dashboard reclaim uses the same idle eligibility as automatic
        reclaim. Force mode is user-confirmed and asks participants for
        a stronger release path that may interrupt in-flight work.
        """
        async with self._lock:
            candidates = (
                self._force_candidates()
                if force
                else self._eligible_candidates(
                    "", idle_seconds=idle_seconds, include_same_model=True
                )
            )
            for candidate in candidates:
                if candidate.owner_id != target_id or candidate.model != model:
                    continue

                released = await candidate.release()
                if released:
                    self._pending.pop((candidate.owner_id, candidate.model), None)
                    if force:
                        logger.info(
                            "dashboard force-closed local model %s on %s; estimated %s %.2f GiB",
                            candidate.model,
                            candidate.owner_id,
                            self._spec.label,
                            candidate.estimated_gb,
                        )
                    else:
                        logger.info(
                            "dashboard requested release of idle local model %s on %s; estimated %s %.2f GiB",
                            candidate.model,
                            candidate.owner_id,
                            self._spec.label,
                            candidate.estimated_gb,
                        )
                    reason = None
                else:
                    logger.warning(
                        "dashboard failed to release local model %s on %s",
                        candidate.model,
                        candidate.owner_id,
                    )
                    reason = "release_failed"

                result: dict[str, object] = {
                    "target_id": candidate.owner_id,
                    "model": candidate.model,
                    self._spec.amount_key: candidate.estimated_gb,
                    "released": released,
                    "reason": reason,
                }
                if force:
                    result["forced"] = True
                return result

            result = {
                "target_id": target_id,
                "model": model,
                "released": False,
                "reason": "not_eligible",
            }
            if force:
                result["forced"] = True
            return result

    # -- Internals -----------------------------------------------------

    def _record_pending(
        self, key: tuple[str, str], estimated_gb: float
    ) -> None:
        self._pending[key] = _PendingEntry(
            estimated_gb=estimated_gb,
            acquired_at=time.monotonic(),
        )

    def _pending_used_mib(self, exclude_key: Optional[tuple[str, str]] = None) -> float:
        now = time.monotonic()
        total = 0.0
        stale: list[tuple[str, str]] = []
        for k, entry in self._pending.items():
            if k == exclude_key:
                continue
            if (
                entry.confirmed_loaded_at is not None
                and now - entry.confirmed_loaded_at > PENDING_LOADED_GRACE_SECONDS
            ):
                # The live reading should have caught up; drop the bookkeeping.
                stale.append(k)
                continue
            total += _gb_to_mib(entry.estimated_gb)
        for k in stale:
            self._pending.pop(k, None)
        return total

    def _effective_free_mib(
        self, free_mib: float, *, exclude_key: Optional[tuple[str, str]] = None
    ) -> float:
        return max(0.0, free_mib - self._pending_used_mib(exclude_key=exclude_key))

    async def _wait_for_available_after_release(
        self, required_mib: float, *, exclude_key: Optional[tuple[str, str]] = None
    ) -> Optional[float]:
        latest_mib: Optional[float] = None
        for delay_seconds in _POST_RELEASE_REFRESH_DELAYS_SECONDS:
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            current_mib = await self._provider()
            if current_mib is None:
                continue
            latest_mib = current_mib
            effective = self._effective_free_mib(current_mib, exclude_key=exclude_key)
            if effective >= required_mib:
                return current_mib
        return latest_mib

    async def _wait_for_raw_available_after_release(
        self, threshold_mib: float
    ) -> Optional[float]:
        latest_mib: Optional[float] = None
        for delay_seconds in _POST_RELEASE_REFRESH_DELAYS_SECONDS:
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            current_mib = await self._provider()
            if current_mib is None:
                continue
            latest_mib = current_mib
            if current_mib >= threshold_mib:
                return current_mib
        return latest_mib

    def _eligible_candidates(
        self,
        requested_model: str,
        *,
        idle_seconds: float = VRAM_IDLE_RECLAIM_SECONDS,
        include_same_model: bool = False,
    ) -> list:
        now = time.monotonic()
        candidates: list = []
        for participant in self._participants.values():
            if participant.active_requests:
                continue
            for candidate in self._spec.release_candidates(
                participant, now, idle_seconds
            ):
                if not include_same_model and candidate.model == requested_model:
                    continue
                candidates.append(candidate)
        candidates.sort(
            key=lambda item: (item.last_used_monotonic, item.owner_id, item.model)
        )
        return candidates

    def _force_candidates(self) -> list:
        now = time.monotonic()
        candidates: list = []
        for participant in self._participants.values():
            candidates.extend(self._spec.force_candidates(participant, now))
        candidates.sort(
            key=lambda item: (item.last_used_monotonic, item.owner_id, item.model)
        )
        return candidates

    def _insufficient_error(
        self,
        requester: Any,
        *,
        model: str,
        required_mib: float,
        available_mib: float,
        reclaimable_mib: float,
        attempted_release_mib: float = 0.0,
    ) -> LocalTargetResourceError:
        message = (
            f"Insufficient {self._spec.label} to start local model "
            f"'{model or requester.target_id}': requires about {_fmt_gib(required_mib)}, "
            f"but only {_fmt_gib(available_mib)} is currently available. "
            f"Remaining eligible idle local models can free about {_fmt_gib(reclaimable_mib)}."
        )
        if attempted_release_mib > 0:
            message += (
                f" Already requested release for about {_fmt_gib(attempted_release_mib)} "
                "and rechecked current free resource."
            )
        return LocalTargetResourceError(message)


class VramCoordinator(_ResourceCoordinator):
    """Coordinate GPU VRAM checks, startup queueing, and idle eviction."""

    def __init__(
        self,
        provider: ResourceProvider = nvidia_smi_free_vram_mib,
        *,
        total_provider: Optional[ResourceProvider] = None,
    ) -> None:
        super().__init__(
            provider,
            total_provider=(
                total_provider
                if total_provider is not None
                else nvidia_smi_total_vram_mib
            ),
            spec=_VRAM_SPEC,
        )
        self._execution_condition = asyncio.Condition()
        self._execution_leases: dict[int, _ExecutionLeaseState] = {}
        self._next_execution_lease_id = 1
        self._observed_headroom_mib: dict[str, float] = {}

    @staticmethod
    def runtime_group_for(participant: Any) -> str:
        value = getattr(participant, "vram_runtime_group", None)
        return str(value or participant.target_id)

    def invalidate_runtime_group(self, runtime_group: str) -> list[str]:
        """Invalidate every participant affected by one runtime-wide unload.

        ComfyUI's ``/free`` endpoint operates on the whole server process.  A
        target-local reservation therefore becomes stale when a sibling target
        sharing the same base URL unloads models.  Participants opt into the
        group by exposing ``vram_runtime_group`` and an invalidation callback.
        """

        invalidated: list[str] = []
        for participant in list(self._participants.values()):
            if self.runtime_group_for(participant) != runtime_group:
                continue
            callback = getattr(participant, "invalidate_vram_reservation", None)
            if callable(callback):
                callback()
            for key in list(self._pending):
                if key[0] == participant.target_id:
                    self._pending.pop(key, None)
            invalidated.append(participant.target_id)
        return invalidated

    def runtime_group_active_requests(self, runtime_group: str) -> int:
        return sum(
            int(getattr(participant, "active_requests", 0) or 0)
            for participant in self._participants.values()
            if self.runtime_group_for(participant) == runtime_group
        )

    def observed_headroom_mib(self, workload_key: str) -> float:
        return max(0.0, self._observed_headroom_mib.get(workload_key, 0.0))

    def observed_headroom_snapshots(self) -> list[dict[str, object]]:
        return [
            {"workload_key": key, "observed_headroom_mib": value}
            for key, value in sorted(self._observed_headroom_mib.items())
        ]

    def execution_snapshots(self) -> list[dict[str, object]]:
        return [
            {
                "lease_id": state.lease_id,
                "owner_id": state.owner_id,
                "runtime_group": state.runtime_group,
                "model": state.model,
                "workload_key": state.workload_key,
                "reserved_headroom_mib": state.reserved_headroom_mib,
                "min_free_mib": state.min_free_mib,
                "minimum_free_mib": state.minimum_free_mib,
                "exclusive": state.exclusive,
                "breached": state.breached,
            }
            for state in self._execution_leases.values()
        ]

    def _effective_free_mib(
        self, free_mib: float, *, exclude_key: Optional[tuple[str, str]] = None
    ) -> float:
        effective = super()._effective_free_mib(
            free_mib, exclude_key=exclude_key
        )
        excluded_owner = exclude_key[0] if exclude_key is not None else None
        other_leases = [
            state
            for state in self._execution_leases.values()
            if state.owner_id != excluded_owner
        ]
        if any(state.exclusive for state in other_leases):
            return 0.0
        return max(
            0.0,
            effective
            - sum(state.reserved_headroom_mib for state in other_leases),
        )

    async def acquire_execution(
        self,
        requester: VramParticipant,
        *,
        model: str,
        workload_key: str,
        reload_vram_gb: float = 0.0,
        request_headroom_gb: float = 0.0,
        min_free_vram_gb: float = 0.0,
        cleanup_policy: str = "keep",
        exclusive: bool = False,
        cleanup: Optional[ReleaseCallback] = None,
        interrupt: Optional[InterruptCallback] = None,
    ) -> VramExecutionLease:
        """Queue and admit one full inference execution.

        Existing residency is intentionally *not* treated as zero-cost here.
        For a resident model, live free VRAM must still cover its transient
        request headroom plus the configured safety floor.  ``adaptive`` may
        unload GPU weights while retaining the engine's CPU/object cache, then
        normal model-load admission runs again inside the returned lease.
        """

        policy = str(cleanup_policy or "keep").strip().lower()
        if policy not in {"keep", "adaptive", "unload"}:
            policy = "keep"
        runtime_group = self.runtime_group_for(requester)
        configured_headroom_mib = _gb_to_mib(max(0.0, request_headroom_gb))
        learned_headroom_mib = self.observed_headroom_mib(workload_key)
        headroom_mib = max(configured_headroom_mib, learned_headroom_mib)
        floor_mib = _gb_to_mib(max(0.0, min_free_vram_gb))
        reload_mib = _gb_to_mib(max(0.0, reload_vram_gb))
        cleaned = False

        if headroom_mib > 0 and self._cached_total_mib is None:
            await self._total_mib()

        async with self._execution_condition:
            while True:
                if exclusive and any(
                    int(getattr(participant, "active_requests", 0) or 0) > 0
                    and participant.target_id != requester.target_id
                    for participant in self._participants.values()
                ):
                    # Legacy local clients use load admission rather than a
                    # request-duration lease. Poll their active counter so a
                    # high-variance video job queues behind in-flight text work
                    # instead of overlapping it.
                    try:
                        await asyncio.wait_for(
                            self._execution_condition.wait(), timeout=0.5
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue
                if not self._execution_compatible(
                    runtime_group=runtime_group,
                    exclusive=exclusive,
                    headroom_mib=headroom_mib,
                    floor_mib=floor_mib,
                ):
                    await self._execution_condition.wait()
                    continue

                available_mib = await self._provider()
                if available_mib is None:
                    raise LocalTargetResourceError(
                        f"Unable to determine available GPU VRAM for local model "
                        f"'{model or requester.target_id}'. "
                        f"{_VRAM_SPEC.unavailable_hint}"
                    )
                reserved_by_others = sum(
                    state.reserved_headroom_mib
                    for state in self._execution_leases.values()
                )
                effective_free_mib = max(0.0, available_mib - reserved_by_others)
                resident = bool(requester.has_vram_reservation(model))
                required_mib = floor_mib + (headroom_mib if resident else 0.0)

                should_clean = resident and (
                    policy == "unload"
                    or (policy == "adaptive" and effective_free_mib < required_mib)
                )
                if should_clean and cleanup is not None:
                    logger.info(
                        "local model %s on %s has %s effective free VRAM; "
                        "request needs %s transient headroom plus %s safety floor; "
                        "requesting %s runtime cleanup",
                        model or requester.target_id,
                        runtime_group,
                        _fmt_gib(effective_free_mib),
                        _fmt_gib(headroom_mib),
                        _fmt_gib(floor_mib),
                        policy,
                    )
                    cleaned = bool(await cleanup())
                    if cleaned:
                        available_mib = await self._wait_for_raw_available_after_release(
                            max(floor_mib, reload_mib)
                        ) or 0.0
                        effective_free_mib = max(
                            0.0, available_mib - reserved_by_others
                        )
                        resident = bool(requester.has_vram_reservation(model))
                        required_mib = floor_mib + (
                            headroom_mib if resident else 0.0
                        )
                        if reload_mib > 0 and effective_free_mib < reload_mib:
                            raise LocalTargetResourceError(
                                f"GPU VRAM cleanup for local model "
                                f"'{model or requester.target_id}' completed, but only "
                                f"{_fmt_gib(effective_free_mib)} became available; "
                                f"about {_fmt_gib(reload_mib)} is required to reload it. "
                                "No prompt was submitted."
                            )

                if effective_free_mib < required_mib:
                    # An already-running lease may be consuming the live free
                    # reading even when weighted admission allowed both. Queue
                    # until it exits; with no active lease this is a real
                    # admission failure and must not reach ComfyUI /prompt.
                    if self._execution_leases:
                        await self._execution_condition.wait()
                        continue
                    raise LocalTargetResourceError(
                        f"Insufficient GPU VRAM headroom to run local model "
                        f"'{model or requester.target_id}': current effective free "
                        f"is {_fmt_gib(effective_free_mib)}, but this request needs "
                        f"{_fmt_gib(headroom_mib)} transient headroom plus "
                        f"{_fmt_gib(floor_mib)} safety reserve. No prompt was submitted."
                    )

                lease_id = self._next_execution_lease_id
                self._next_execution_lease_id += 1
                state = _ExecutionLeaseState(
                    lease_id=lease_id,
                    owner_id=requester.target_id,
                    runtime_group=runtime_group,
                    model=model,
                    workload_key=workload_key,
                    reserved_headroom_mib=headroom_mib,
                    min_free_mib=floor_mib,
                    exclusive=exclusive,
                    resident_at_start=resident,
                    baseline_free_mib=available_mib,
                    minimum_free_mib=available_mib,
                    interrupt=interrupt,
                    cleanup=cleanup,
                )
                self._execution_leases[lease_id] = state
                state.monitor_task = asyncio.create_task(
                    self._monitor_execution(state),
                    name=f"vram-execution-monitor-{lease_id}",
                )
                return VramExecutionLease(
                    self, state, cleaned_for_headroom=cleaned
                )

    def _execution_compatible(
        self,
        *,
        runtime_group: str,
        exclusive: bool,
        headroom_mib: float,
        floor_mib: float,
    ) -> bool:
        leases = list(self._execution_leases.values())
        if any(state.runtime_group == runtime_group for state in leases):
            return False
        if exclusive:
            return not leases
        if any(state.exclusive for state in leases):
            return False
        if self._cached_total_mib is None:
            return True
        usable_mib = max(0.0, self._cached_total_mib - floor_mib)
        reserved_mib = sum(state.reserved_headroom_mib for state in leases)
        return reserved_mib + headroom_mib <= usable_mib

    async def _monitor_execution(self, state: _ExecutionLeaseState) -> None:
        try:
            while state.lease_id in self._execution_leases:
                await asyncio.sleep(0.5)
                available_mib = await self._provider()
                if available_mib is None:
                    continue
                state.minimum_free_mib = min(state.minimum_free_mib, available_mib)
                if (
                    state.min_free_mib > 0
                    and available_mib < state.min_free_mib
                    and not state.interrupted
                ):
                    state.breached = True
                    logger.error(
                        "GPU VRAM safety floor breached for %s on %s: free %s, floor %s; interrupting active workflow",
                        state.model,
                        state.runtime_group,
                        _fmt_gib(available_mib),
                        _fmt_gib(state.min_free_mib),
                    )
                    if state.interrupt is not None:
                        try:
                            state.interrupted = bool(await state.interrupt())
                        except Exception:
                            state.interrupted = False
                            logger.warning(
                                "failed to interrupt workflow after GPU VRAM safety breach",
                                exc_info=True,
                            )
                    else:
                        state.interrupted = True
        except asyncio.CancelledError:
            raise

    async def release_execution(self, lease_id: int) -> None:
        async with self._execution_condition:
            state = self._execution_leases.get(lease_id)
            if state is None:
                return
            monitor = state.monitor_task
            if monitor is not None:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass

            if state.resident_at_start:
                observed_mib = max(
                    0.0, state.baseline_free_mib - state.minimum_free_mib
                )
                if observed_mib > 0:
                    # Keep a small allocator/sampling margin. Learned values
                    # only move upward and never weaken an explicit profile.
                    learned_mib = observed_mib + 256.0
                    previous_mib = self._observed_headroom_mib.get(
                        state.workload_key, 0.0
                    )
                    if learned_mib > previous_mib:
                        self._observed_headroom_mib[state.workload_key] = learned_mib

            if state.breached and state.cleanup is not None:
                try:
                    await state.cleanup()
                except Exception:
                    logger.warning(
                        "failed to clean runtime after GPU VRAM safety breach",
                        exc_info=True,
                    )
            self._execution_leases.pop(lease_id, None)
            self._execution_condition.notify_all()

    async def free_vram_mib(self) -> Optional[float]:
        return await self._free_mib()

    async def total_vram_mib(self) -> Optional[float]:
        return await self._total_mib()

    async def ensure_available(
        self,
        requester: VramParticipant,
        *,
        model: str = "",
        estimated_vram_gb: Optional[float] = None,
    ) -> None:
        blocking = next(
            (
                state
                for state in self._execution_leases.values()
                if state.exclusive and state.owner_id != requester.target_id
            ),
            None,
        )
        if blocking is not None:
            raise LocalTargetResourceError(
                f"GPU is reserved by exclusive local workflow "
                f"'{blocking.model}' on {blocking.runtime_group}; local model "
                f"'{model or requester.target_id}' was not started."
            )
        await self._ensure_available(
            requester, model=model, estimated_gb=estimated_vram_gb
        )


class MemoryCoordinator(_ResourceCoordinator):
    """Coordinate host RAM checks, startup queueing, and idle eviction."""

    def __init__(
        self,
        provider: ResourceProvider = system_free_memory_mib,
        *,
        total_provider: Optional[ResourceProvider] = None,
    ) -> None:
        super().__init__(
            provider,
            total_provider=(
                total_provider
                if total_provider is not None
                else system_total_memory_mib
            ),
            spec=_MEMORY_SPEC,
        )

    def invalidate_runtime_group(self, runtime_group: str) -> list[str]:
        invalidated: list[str] = []
        for participant in list(self._participants.values()):
            participant_group = str(
                getattr(participant, "vram_runtime_group", None)
                or participant.target_id
            )
            if participant_group != runtime_group:
                continue
            callback = getattr(participant, "invalidate_memory_reservation", None)
            if callable(callback):
                callback()
            for key in list(self._pending):
                if key[0] == participant.target_id:
                    self._pending.pop(key, None)
            invalidated.append(participant.target_id)
        return invalidated

    async def free_memory_mib(self) -> Optional[float]:
        return await self._free_mib()

    async def total_memory_mib(self) -> Optional[float]:
        return await self._total_mib()

    async def ensure_available(
        self,
        requester: MemoryParticipant,
        *,
        model: str = "",
        estimated_memory_gb: Optional[float] = None,
    ) -> None:
        await self._ensure_available(
            requester, model=model, estimated_gb=estimated_memory_gb
        )
