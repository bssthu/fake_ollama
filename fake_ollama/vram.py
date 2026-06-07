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


# ---------------------------------------------------------------------------
# Live readings
# ---------------------------------------------------------------------------


async def nvidia_smi_free_vram_mib() -> Optional[float]:
    """Return total free NVIDIA GPU VRAM in MiB, or None if unavailable."""
    return await _nvidia_smi_query("memory.free")


async def nvidia_smi_total_vram_mib() -> Optional[float]:
    """Return total NVIDIA GPU VRAM (capacity) in MiB, or None if unavailable."""
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

    total = 0.0
    found = False
    for raw_line in stdout.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            total += float(line.split()[0])
            found = True
        except (ValueError, IndexError):
            continue
    return total if found else None


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
