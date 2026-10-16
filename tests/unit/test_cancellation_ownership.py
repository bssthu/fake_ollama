import asyncio

import httpx
import pytest

from fake_ollama.llama_cpp_client import LlamaCppClient
from fake_ollama.vram import VramCoordinator


async def test_cancel_after_semaphore_acquire_restores_capacity(monkeypatch):
    client = LlamaCppClient("http://local.test", max_concurrent_requests=1,
                            client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    original_wait = asyncio.wait
    acquired = asyncio.Event()
    async def barrier(*args, **kwargs):
        result = await original_wait(*args, **kwargs)
        acquired.set()
        await asyncio.Event().wait()
        return result
    monkeypatch.setattr(asyncio, "wait", barrier)
    async def enter():
        async with client._concurrency_slot():
            pytest.fail("Must cancel before execution")
    try:
        for _ in range(3):
            acquired.clear()
            task = asyncio.create_task(enter())
            await asyncio.wait_for(acquired.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client._request_semaphore._value == 1
            assert client.queued_requests == 0
            assert not client._wait_pending
            assert client.active_requests == 0
    finally:
        await client.aclose()


class Participant:
    target_id = "test"
    vram_runtime_group = "test-runtime"
    active_requests = 0
    def has_vram_reservation(self, model):
        return False


async def test_execution_cleanup_does_not_hold_condition_or_leak_on_cancel():
    async def provider():
        return 24 * 1024
    coordinator = VramCoordinator(provider=provider)
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def cleanup():
        calls.append(True)
        started.set()
        await finish.wait()
        return True
    lease = await coordinator.acquire_execution(Participant(), model="m", workload_key="m", cleanup=cleanup)
    lease._state.breached = True
    task = asyncio.create_task(coordinator.release_execution(lease._state.lease_id))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Cancellation must transfer work to an observable owner, not strand a lease.
    finish.set()
    await coordinator.release_execution(lease._state.lease_id)
    assert not coordinator.execution_snapshots()
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["exception", "false", "timeout"])
async def test_failed_cleanup_quarantines_without_permanent_lease(failure):
    from fake_ollama.vram import LocalTargetResourceError
    async def provider():
        return 24 * 1024
    coordinator = VramCoordinator(provider=provider)
    recovered = False
    async def cleanup():
        if recovered:
            return True
        if failure == "exception":
            raise RuntimeError("cleanup failure")
        if failure == "timeout":
            await asyncio.Event().wait()
        return False
    lease = await coordinator.acquire_execution(Participant(), model="m", workload_key="m", cleanup=cleanup)
    lease._state.breached = True
    coordinator._cleanup_timeout_seconds = 0 if failure == "timeout" else 1
    await coordinator.release_execution(lease._state.lease_id)
    assert not coordinator._execution_leases
    assert coordinator.execution_snapshots()[0]["status"] == "quarantined"
    with pytest.raises(LocalTargetResourceError):
        await coordinator.acquire_execution(Participant(), model="m", workload_key="m")
    with pytest.raises(LocalTargetResourceError):
        await coordinator.ensure_available(Participant(), model="m", estimated_vram_gb=1)
    recovered = True
    coordinator._cleanup_timeout_seconds = 1
    assert await coordinator.recover_runtime("test-runtime")
    assert not coordinator.execution_snapshots()


async def test_cleanup_io_releases_condition_lock():
    async def provider():
        return 24 * 1024
    coordinator = VramCoordinator(provider=provider)
    started, finish = asyncio.Event(), asyncio.Event()
    async def cleanup():
        started.set()
        await finish.wait()
        return True
    lease = await coordinator.acquire_execution(Participant(), model="m", workload_key="m", cleanup=cleanup)
    lease._state.breached = True
    release = asyncio.create_task(coordinator.release_execution(lease._state.lease_id))
    await started.wait()
    try:
        assert not coordinator._execution_condition.locked()
    finally:
        finish.set()
        await release


async def test_cleanup_ignoring_cancel_is_bounded_and_tracked():
    async def provider():
        return 24 * 1024
    coordinator = VramCoordinator(provider=provider)
    ignored, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def cleanup():
        calls.append(True)
        if len(calls) == 1:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                ignored.set()
                await finish.wait()
        return True
    lease = await coordinator.acquire_execution(Participant(), model="m", workload_key="m", cleanup=cleanup)
    lease._state.breached = True
    coordinator._cleanup_timeout_seconds = 0
    await coordinator.release_execution(lease._state.lease_id)
    await ignored.wait()
    assert not coordinator._execution_leases
    assert not await coordinator.recover_runtime("test-runtime")
    assert len(calls) == 1
    finish.set()
    await coordinator._cleanup_tasks["test-runtime"]
    coordinator._cleanup_timeout_seconds = 1
    assert await coordinator.recover_runtime("test-runtime")


async def test_repeated_cancel_during_reclaim_poll_keeps_one_owner(monkeypatch):
    async def provider():
        return 24 * 1024
    coordinator = VramCoordinator(provider=provider)
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def cleanup():
        calls.append(True)
        return True
    async def reclaim(threshold):
        started.set()
        await finish.wait()
        return 24 * 1024
    monkeypatch.setattr(coordinator, "_wait_for_raw_available_after_release", reclaim)
    lease = await coordinator.acquire_execution(Participant(), model="m", workload_key="m", cleanup=cleanup)
    lease._state.breached = True
    first = asyncio.create_task(coordinator.release_execution(lease._state.lease_id))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(coordinator.release_execution(lease._state.lease_id))
    # Run the second caller until it waits on the already-owned cleanup task.
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert coordinator.execution_snapshots()[0]["status"] == "releasing"
    finish.set()
    await coordinator.release_execution(lease._state.lease_id)
    assert len(calls) == 1
    assert not coordinator.execution_snapshots()
