"""Unit tests for the host-RAM admission coordinator.

The algorithm is shared with :class:`VramCoordinator` (covered by
``test_vram_queue``); these tests lock in the memory-specific wiring: the
``estimated_memory_gb`` kwarg, ``has_memory_reservation`` /
``memory_release_candidates`` participant hooks, and the
``estimated_memory_gb`` result-dict key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from fake_ollama.vram import (
    LocalTargetResourceError,
    MemoryCoordinator,
    MemoryReleaseCandidate,
)


@dataclass
class _FakeParticipant:
    target_id: str
    _reservations: set[str] = field(default_factory=set)
    active_requests: int = 0  # type: ignore[assignment]

    def has_memory_reservation(self, model: str) -> bool:
        return model in self._reservations

    def memory_release_candidates(self, *, now: float, idle_seconds: float):
        return []


def _free_provider(mib: float):
    async def _f() -> Optional[float]:
        return mib

    return _f


@pytest.mark.asyncio
async def test_concurrent_different_models_respect_capacity():
    coord = MemoryCoordinator(provider=_free_provider(40 * 1024.0))
    p_a = _FakeParticipant("target-a")
    p_b = _FakeParticipant("target-b")
    coord.register(p_a)
    coord.register(p_b)

    await coord.ensure_available(p_a, model="a", estimated_memory_gb=30)
    assert coord.has_pending("target-a", "a")

    with pytest.raises(LocalTargetResourceError, match="Insufficient system RAM"):
        await coord.ensure_available(p_b, model="b", estimated_memory_gb=30)
    assert not coord.has_pending("target-b", "b")


@pytest.mark.asyncio
async def test_same_model_pending_is_idempotent():
    coord = MemoryCoordinator(provider=_free_provider(40 * 1024.0))
    p = _FakeParticipant("t")
    coord.register(p)

    await coord.ensure_available(p, model="m", estimated_memory_gb=30)
    await coord.ensure_available(p, model="m", estimated_memory_gb=30)
    assert coord.has_pending("t", "m")


@pytest.mark.asyncio
async def test_none_estimate_skips_admission():
    coord = MemoryCoordinator(provider=_free_provider(0.0))
    p = _FakeParticipant("t")
    coord.register(p)
    # No estimate => no gating, even with zero free RAM reported.
    await coord.ensure_available(p, model="m", estimated_memory_gb=None)
    assert not coord.has_pending("t", "m")


@pytest.mark.asyncio
async def test_reclaim_model_returns_memory_key():
    coord = MemoryCoordinator(provider=_free_provider(100.0))

    class _IdleParticipant:
        target_id = "target"
        active_requests = 0

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic() - 120.0

        def has_memory_reservation(self, model: str) -> bool:
            return model == "old"

        def memory_release_candidates(self, *, now: float, idle_seconds: float):
            if now - self.last_used < idle_seconds:
                return []
            return [
                MemoryReleaseCandidate(
                    owner_id=self.target_id,
                    model="old",
                    estimated_memory_gb=30.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            return True

    participant = _IdleParticipant()
    coord.register(participant)

    result = await coord.reclaim_model(
        target_id="target", model="old", idle_seconds=60.0
    )

    assert participant.released is True
    assert result == {
        "target_id": "target",
        "model": "old",
        "estimated_memory_gb": 30.0,
        "released": True,
        "reason": None,
    }


@pytest.mark.asyncio
async def test_reclaim_if_below_releases_idle_candidates():
    free = {"mib": 100.0}

    async def provider() -> Optional[float]:
        return free["mib"]

    coord = MemoryCoordinator(provider=provider)

    class _IdleParticipant:
        target_id = "target"
        active_requests = 0

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic() - 120.0

        def has_memory_reservation(self, model: str) -> bool:
            return model == "old"

        def memory_release_candidates(self, *, now: float, idle_seconds: float):
            if now - self.last_used < idle_seconds:
                return []
            return [
                MemoryReleaseCandidate(
                    owner_id=self.target_id,
                    model="old",
                    estimated_memory_gb=30.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            free["mib"] = 40 * 1024.0
            return True

    participant = _IdleParticipant()
    coord.register(participant)

    result = await coord.reclaim_if_below(threshold_mib=2048.0, idle_seconds=60.0)

    assert participant.released is True
    assert result["released"] == [
        {"owner_id": "target", "model": "old", "estimated_memory_gb": 30.0}
    ]
