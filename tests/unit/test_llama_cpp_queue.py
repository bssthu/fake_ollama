"""Unit tests for the local concurrency gate in ``LlamaCppClient``.

These exercise the ``max_concurrent_requests`` semaphore + the
``request_read_timeout_seconds`` knob added to keep llama.cpp's
``--parallel N`` slot count from being overwhelmed by external clients.

Each test pins the upstream behaviour with ``httpx.MockTransport`` and uses
``asyncio.Event`` to deterministically observe queue state without sleeping.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import httpx
import pytest

from fake_ollama.config import LlamaCppDefaults, LlamaCppTarget
from fake_ollama.llama_cpp_client import LlamaCppClient


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# core gate semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_serialises_to_max_concurrent_requests():
    """With cap=1, only one chat() runs at a time; others queue FIFO."""
    in_flight = asyncio.Event()
    release = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(request.url.path)

    # Wrap the transport so the first POST blocks until we say so. Other POSTs
    # return immediately — this lets us observe queue state with cap=1.
    real_handler = handler
    first_done = asyncio.Event()

    async def gated(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and not first_done.is_set():
            in_flight.set()
            first_done.set()
            await release.wait()
        return real_handler(request)

    transport = httpx.MockTransport(gated)
    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=transport),
        max_concurrent_requests=1,
    )
    try:
        t1 = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        # wait until first request grabbed the slot and is blocked upstream
        await asyncio.wait_for(in_flight.wait(), timeout=2.0)
        assert client.active_requests == 1
        assert client.queued_requests == 0

        t2 = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        t3 = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        # let the event loop run so t2/t3 enter the gate
        for _ in range(20):
            await asyncio.sleep(0)
            if client.queued_requests == 2:
                break
        assert client.active_requests == 1
        assert client.queued_requests == 2

        release.set()
        await asyncio.wait_for(asyncio.gather(t1, t2, t3), timeout=2.0)
        assert client.active_requests == 0
        assert client.queued_requests == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_queue_allows_n_concurrent_when_cap_is_n():
    """With cap=2, two requests run in parallel and the 3rd queues."""
    holds: List[asyncio.Event] = []
    entered = asyncio.Event()

    async def gated(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        ev = asyncio.Event()
        holds.append(ev)
        if len(holds) == 2:
            entered.set()
        await ev.wait()
        return httpx.Response(200, json={"i": len(holds)})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gated)),
        max_concurrent_requests=2,
    )
    try:
        tasks = [
            asyncio.create_task(client.chat({"model": "m", "messages": []}))
            for _ in range(3)
        ]
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        # Two upstream calls in flight, third is queued in fake_ollama.
        for _ in range(20):
            await asyncio.sleep(0)
            if client.queued_requests == 1:
                break
        assert client.active_requests == 2
        assert client.queued_requests == 1
        assert len(holds) == 2  # third hasn't reached the transport yet

        # Release first slot → third request takes the slot, reaches transport.
        holds[0].set()
        for _ in range(50):
            await asyncio.sleep(0)
            if len(holds) == 3:
                break
        assert len(holds) == 3
        assert client.queued_requests == 0

        for ev in holds[1:]:
            ev.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert client.active_requests == 0
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# cancellation / exception paths must not leak slots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_while_waiting_in_queue_releases_slot_counter():
    """If a queued request is cancelled before acquiring the slot, queued
    counter must decrement and the slot must remain usable by others."""
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def gated(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        in_flight.set()
        await release.wait()
        return httpx.Response(200, json={"ok": True})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gated)),
        max_concurrent_requests=1,
    )
    try:
        holding = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        await asyncio.wait_for(in_flight.wait(), timeout=2.0)

        waiter = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        for _ in range(20):
            await asyncio.sleep(0)
            if client.queued_requests == 1:
                break
        assert client.queued_requests == 1

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        # queued must reflect cancellation immediately
        assert client.queued_requests == 0

        # The slot is still held by the first request; once we release it the
        # first task completes and a *new* task can acquire fresh.
        release.set()
        await asyncio.wait_for(holding, timeout=2.0)
        assert client.active_requests == 0

        # New request should run end-to-end on a fresh slot.
        in_flight.clear()
        release.clear()
        release.set()  # let the second call through immediately
        new_task = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        await asyncio.wait_for(new_task, timeout=2.0)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upstream_exception_releases_slot():
    """A non-2xx / transport error must release the slot via the asynccontextmanager finally."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=_mock_client(handler),
        max_concurrent_requests=1,
    )
    try:
        for _ in range(3):
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat({"model": "m", "messages": []})
        assert calls["n"] == 3
        assert client.active_requests == 0
        assert client.queued_requests == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_consumer_cancellation_releases_slot():
    """If the API consumer stops iterating the stream early (GeneratorExit /
    CancelledError), the slot must be released so queued requests proceed."""
    sent_first_chunk = asyncio.Event()
    keep_streaming = asyncio.Event()

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            sent_first_chunk.set()
            await keep_streaming.wait()
            yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, stream=_Stream())

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=_mock_client(handler),
        max_concurrent_requests=1,
    )
    try:
        async def consume_one_then_drop():
            agen = client.stream_chat({"model": "m", "messages": []})
            async for _line in agen:
                # bail after first chunk → triggers GeneratorExit on next step
                break
            await agen.aclose()

        await asyncio.wait_for(consume_one_then_drop(), timeout=2.0)
        # slot must be free again
        assert client.active_requests == 0
        assert client.queued_requests == 0

        # And a follow-up stream should work without hanging on a leaked slot.
        keep_streaming.set()
        sent_first_chunk.clear()
        chunks = [
            line
            async for line in client.stream_chat({"model": "m", "messages": []})
        ]
        assert chunks
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# config plumbing & disabled mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrent_zero_or_none_disables_gate():
    """Both 0 and None must mean "no local gate"."""
    for cap in (0, None):
        client = LlamaCppClient(
            "http://127.0.0.1:21441",
            client=_mock_client(
                lambda r: httpx.Response(200, json={"ok": True})
                if r.url.path != "/health"
                else httpx.Response(200)
            ),
            max_concurrent_requests=cap,
        )
        try:
            assert client._request_semaphore is None
            # queued counter should never increment when gate is off
            await client.chat({"model": "m", "messages": []})
            assert client.queued_requests == 0
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# passthrough mode (default): cap=None / cap=0
#
# Pre-queue behaviour: every concurrent request is forwarded to the upstream
# llama.cpp server immediately. ``_active`` reflects in-flight requests at
# the upstream socket (which is the dashboard's "Active" column). ``_queued``
# must stay at 0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [None, 0])
@pytest.mark.asyncio
async def test_passthrough_forwards_all_concurrent_requests_immediately(cap):
    """N parallel chats with cap disabled: all N reach upstream simultaneously,
    Active==N, Queued==0 (matches pre-queue-feature behaviour)."""
    entered_count = 0
    all_entered = asyncio.Event()
    release = asyncio.Event()
    N = 5

    async def gated(request: httpx.Request) -> httpx.Response:
        nonlocal entered_count
        if request.url.path == "/health":
            return httpx.Response(200)
        entered_count += 1
        if entered_count == N:
            all_entered.set()
        await release.wait()
        return httpx.Response(200, json={"ok": True})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gated)),
        max_concurrent_requests=cap,
    )
    try:
        assert client._request_semaphore is None
        tasks = [
            asyncio.create_task(client.chat({"model": "m", "messages": []}))
            for _ in range(N)
        ]
        # All N should land on the upstream concurrently — no proxy gate.
        await asyncio.wait_for(all_entered.wait(), timeout=2.0)
        assert client.active_requests == N
        assert client.queued_requests == 0
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert client.active_requests == 0
        assert client.queued_requests == 0
    finally:
        await client.aclose()


@pytest.mark.parametrize("cap", [None, 0])
@pytest.mark.asyncio
async def test_passthrough_stream_chat_also_forwards_concurrently(cap):
    """Same passthrough guarantee for ``stream_chat``."""
    entered_count = 0
    all_entered = asyncio.Event()
    release = asyncio.Event()
    N = 4

    async def gated(request: httpx.Request) -> httpx.Response:
        nonlocal entered_count
        if request.url.path == "/health":
            return httpx.Response(200)
        entered_count += 1
        if entered_count == N:
            all_entered.set()
        await release.wait()
        # one trivial SSE chunk then end
        body = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gated)),
        max_concurrent_requests=cap,
    )

    async def consume(stream) -> int:
        n = 0
        async for _ in stream:
            n += 1
        return n

    try:
        streams = [
            client.stream_chat({"model": "m", "messages": []}) for _ in range(N)
        ]
        consumers = [asyncio.create_task(consume(s)) for s in streams]
        await asyncio.wait_for(all_entered.wait(), timeout=2.0)
        assert client.active_requests == N
        assert client.queued_requests == 0
        release.set()
        await asyncio.wait_for(asyncio.gather(*consumers), timeout=2.0)
        assert client.active_requests == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_passthrough_active_decrements_per_completion():
    """Active counter must decrement as each upstream response finishes."""
    holds: list[asyncio.Event] = []
    entered = asyncio.Event()
    N = 3

    async def gated(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        ev = asyncio.Event()
        holds.append(ev)
        if len(holds) == N:
            entered.set()
        await ev.wait()
        return httpx.Response(200, json={"ok": True})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gated)),
        max_concurrent_requests=None,
    )
    try:
        tasks = [
            asyncio.create_task(client.chat({"model": "m", "messages": []}))
            for _ in range(N)
        ]
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        assert client.active_requests == N

        # Release one at a time; Active should walk N -> N-1 -> ... -> 0.
        for i in range(N):
            holds[i].set()
            await asyncio.wait_for(tasks[i], timeout=2.0)
            # Other still-blocked requests keep Active at the right value.
            assert client.active_requests == N - i - 1
        assert client.queued_requests == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_parallel_setting_alone_does_not_create_a_gate():
    """Configuring only ``parallel: 4`` (no explicit
    ``max_concurrent_requests``) must NOT install a proxy queue.

    A strict proxy-side cap is markedly slower than the upstream's native
    HTTP queue because it forces a full fake_ollama ↔ llama.cpp round-trip
    between requests — the next prompt cannot be pre-tokenised while the
    current one is still decoding. ``parallel`` is for the upstream slot
    manager only; proxy-side queueing requires an explicit opt-in.
    """
    t = LlamaCppTarget(
        name="t",
        base_url="http://x",
        model="m",
        model_path="/tmp/m.gguf",
    ).with_defaults(LlamaCppDefaults(parallel=4))
    assert t.effective_max_concurrent_requests is None

    # And the client built from that setting must have no semaphore.
    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=_mock_client(lambda r: httpx.Response(200, json={})),
        max_concurrent_requests=t.effective_max_concurrent_requests,
    )
    try:
        assert client._request_semaphore is None
    finally:
        await client.aclose()


def test_effective_max_concurrent_defaults_to_none_when_unconfigured():
    """When neither ``max_concurrent_requests`` nor ``parallel`` is set, no
    fake_ollama-level gate is installed — requests pass straight through to
    the upstream llama.cpp server, which manages its own ``--parallel`` slots.
    """
    t = LlamaCppTarget(
        name="t",
        base_url="http://x",
        model="m",
        model_path="/tmp/m.gguf",
    ).with_defaults(LlamaCppDefaults())
    assert t.max_concurrent_requests is None
    assert t.parallel is None
    assert t.effective_max_concurrent_requests is None


def test_effective_max_concurrent_does_not_inherit_from_parallel():
    """``parallel`` alone must NOT install a proxy queue: the upstream
    llama.cpp server already manages its own slots, and silently mirroring
    the value here is meaningfully slower than letting llama.cpp queue
    natively (no round-trip stall between requests)."""
    defaults = LlamaCppDefaults(parallel=4)
    t = LlamaCppTarget(
        name="t",
        base_url="http://x",
        model="m",
        model_path="/tmp/m.gguf",
    ).with_defaults(defaults)
    assert t.parallel == 4
    assert t.max_concurrent_requests is None
    assert t.effective_max_concurrent_requests is None


def test_effective_max_concurrent_explicit_overrides_parallel():
    defaults = LlamaCppDefaults(parallel=4)
    t = LlamaCppTarget(
        name="t",
        base_url="http://x",
        model="m",
        model_path="/tmp/m.gguf",
        max_concurrent_requests=2,
    ).with_defaults(defaults)
    assert t.effective_max_concurrent_requests == 2


def test_effective_max_concurrent_zero_disables_gate():
    """An explicit ``0`` is preserved (and is semantically identical to the
    default ``None``: no proxy-side gate)."""
    t = LlamaCppTarget(
        name="t",
        base_url="http://x",
        model="m",
        model_path="/tmp/m.gguf",
        max_concurrent_requests=0,
    ).with_defaults(LlamaCppDefaults(parallel=4))
    assert t.effective_max_concurrent_requests == 0


# ---------------------------------------------------------------------------
# dashboard semantics — idle must be 0 while something is in flight or queued
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loaded_snapshot_reports_zero_idle_while_busy():
    """Whether a request is actively decoding or just sitting in the local
    queue, the dashboard snapshot must NOT report the model as idle —
    otherwise users see a busy model showing 'idle=50s' just because
    ``_touch_vram_reservation`` only fires on request completion."""

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=_mock_client(lambda r: httpx.Response(200)),
        max_concurrent_requests=1,
    )
    try:
        # Simulate a model that was already loaded "a long time ago" then a
        # new burst of work arrived: 1 in-flight + queued waiters.
        client._mark_vram_reserved("m", 1.0)
        assert client._loaded_model is not None
        client._loaded_model.last_used_monotonic -= 120.0  # 2 minutes ago

        # 1. idle when truly idle
        snap = client.loaded_model_snapshots()[0]
        assert snap["idle_seconds"] >= 120.0

        # 2. active=1 → idle reported as 0
        client._active = 1
        snap = client.loaded_model_snapshots()[0]
        assert snap["active_requests"] == 1
        assert snap["idle_seconds"] == 0.0
        assert snap["reclaimable"] is False

        # 3. queued waiters alone also count as busy
        client._active = 0
        client._queued = 3
        snap = client.loaded_model_snapshots()[0]
        assert snap["queued_requests"] == 3
        assert snap["idle_seconds"] == 0.0
        assert snap["reclaimable"] is False

        # 4. clean state restores real idle
        client._queued = 0
        snap = client.loaded_model_snapshots()[0]
        assert snap["idle_seconds"] >= 120.0
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# shutdown bulk cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_begin_shutdown_releases_all_queued_requests_at_once():
    """On Ctrl+C the queue must drain in one go instead of one-per-completion.

    Reproduces the slow-exit symptom: with cap=1 and ~20 queued requests,
    a single ``begin_shutdown()`` call should fail every waiter immediately
    with ``CancelledError`` and bring ``queued_requests`` to 0 without
    needing the in-flight request to finish or the upstream to die.
    """
    held = asyncio.Event()
    release = asyncio.Event()

    async def gated(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        held.set()
        await release.wait()
        return httpx.Response(200, json={"ok": True})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gated)),
        max_concurrent_requests=1,
    )
    try:
        # 1 in-flight + 20 queued
        head = asyncio.create_task(client.chat({"model": "m", "messages": []}))
        await asyncio.wait_for(held.wait(), timeout=2.0)
        waiters = [
            asyncio.create_task(client.chat({"model": "m", "messages": []}))
            for _ in range(20)
        ]
        for _ in range(50):
            await asyncio.sleep(0)
            if client.queued_requests == 20:
                break
        assert client.queued_requests == 20

        # Bulk-cancel all queued waiters; the in-flight one is still upstream.
        client.begin_shutdown()
        results = await asyncio.wait_for(
            asyncio.gather(*waiters, return_exceptions=True), timeout=2.0
        )
        assert all(isinstance(r, asyncio.CancelledError) for r in results), results
        assert client.queued_requests == 0
        # Head request should still be alive — shutdown does not abort the
        # request currently holding the slot (the process-kill path does).
        assert not head.done()

        # New requests after shutdown should also bail immediately.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                client.chat({"model": "m", "messages": []}), timeout=2.0
            )

        release.set()
        await asyncio.wait_for(head, timeout=2.0)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_begin_shutdown_is_idempotent_without_queue():
    """Calling begin_shutdown() with no queued requests must not crash and
    must still block subsequent enqueues."""
    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=_mock_client(lambda r: httpx.Response(200)),
        max_concurrent_requests=1,
    )
    try:
        client.begin_shutdown()
        client.begin_shutdown()  # second call is a no-op
        with pytest.raises(asyncio.CancelledError):
            await client.chat({"model": "m", "messages": []})
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# shutdown bulk cancellation
# ---------------------------------------------------------------------------

