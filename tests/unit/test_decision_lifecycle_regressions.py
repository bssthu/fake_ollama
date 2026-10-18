"""Cancelled callers must not erase a runtime's real work or resource ownership."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from fake_ollama.config import DecisionTarget
from fake_ollama.decision_client import DecisionClient
from fake_ollama.decisions import DecisionValidationError, NativeSystemOneCodec
from fake_ollama.managed_http_runtime import ManagedHTTPRuntime
from fake_ollama.vram import LocalTargetResourceError, MemoryCoordinator, VramCoordinator


def request(**options):
    return NativeSystemOneCodec.parse_request({"model": "m", "state": "test",
        "questions": {"q": {"type": "noul", "instructions": "True?"}}, **options})


def response():
    return {"answers": {"q": {"type": "noul", "noul": .8}}}


async def test_real_http_cancel_holds_slot_until_upstream_finishes():
    # A TCP server deliberately finishes submitted work after disconnection,
    # matching the official StartLux aiohttp server's default handler policy.
    entered, release = asyncio.Event(), asyncio.Event()
    handlers = set()
    counts = {"active": 0, "peak": 0, "posts": 0}

    async def serve(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        active = False
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = next((int(line.split(":", 1)[1]) for line in header.split("\r\n")
                           if line.lower().startswith("content-length:")), 0)
            await reader.readexactly(length)
            status, payload = "404 Not Found", {}
            if header.startswith("POST "):
                counts["active"] += 1
                counts["posts"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
                active = True
                entered.set()
                await release.wait()
                status, payload = "200 OK", response()
            body = json.dumps(payload).encode()
            writer.write(f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            if active:
                counts["active"] -= 1
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = DecisionClient(DecisionTarget(name="tcp", models=["m"],
        base_url=f"http://127.0.0.1:{port}", max_concurrent_requests=1))
    first = asyncio.create_task(client.decide(request()))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()  # repeated cancellation cannot cancel the draining POST
        second = asyncio.create_task(client.decide(request()))
        for _ in range(100):
            if client.queued_requests:
                break
            await asyncio.sleep(.005)
        assert client.queued_requests == 1
        assert not first.done()
        assert client.active_requests == 1
        assert client._ownership.busy()
        assert counts["posts"] == counts["active"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 2)
        assert (await asyncio.wait_for(second, 2)).answers[0].probability == .8
        assert counts["peak"] == 1
        assert client._active == client._request_refs == client._queued == 0
    finally:
        release.set()
        await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)
        await client.aclose()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.parametrize("resource", ["vram", "memory"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_startup_retains_pending_until_confirmed_stop(resource, cancel, monkeypatch):
    async def free():
        return 6 * 1024.0
    coordinator = (VramCoordinator if resource == "vram" else MemoryCoordinator)(provider=free, total_provider=free)
    runtime = ManagedHTTPRuntime("http://startup.test:8010", auto_start=True, start_argv=["fake-server"],
        target_name="startup", **{resource + "_coordinator": coordinator})
    other = ManagedHTTPRuntime("http://other.test:8010", target_name="other", **{resource + "_coordinator": coordinator})
    process = SimpleNamespace(returncode=None)
    starting = asyncio.Event()

    async def healthy():
        if runtime._started_by_us:
            starting.set()
            if cancel:
                await asyncio.Event().wait()
            raise httpx.ConnectError("startup failed after spawning")
        return False

    monkeypatch.setattr(runtime, "_healthy", healthy)
    monkeypatch.setattr("fake_ollama.managed_http_runtime.create_managed_subprocess_exec", AsyncMock(return_value=process))
    stop = AsyncMock(return_value=False)
    monkeypatch.setattr("fake_ollama.managed_http_runtime.terminate_process_tree", stop)

    async def call():
        async with runtime.request_lease("m", **{"estimated_" + resource + "_gb": 4}):
            pytest.fail("startup must not admit inference")

    task = asyncio.create_task(call())
    try:
        await asyncio.wait_for(starting.wait(), 2)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else httpx.ConnectError):
            await task
        assert runtime._request_refs == 0
        assert coordinator._pending_used_mib() == 4 * 1024
        assert coordinator._pending[(runtime.target_id, "m")].confirmed_loaded_at is None
        assert runtime.loaded_model_snapshots()[0]["loading"] is True
        with pytest.raises(LocalTargetResourceError):
            await getattr(other, "_ensure_" + resource)("other-model", 4)
        assert not await runtime.stop_if_owned()
        assert coordinator._pending_used_mib() == 4 * 1024
        stop.return_value = True
        assert await runtime.stop_if_owned()
        assert not runtime.loaded_model_snapshots()
        assert coordinator._pending_used_mib() == 0
        await getattr(other, "_ensure_" + resource)("other-model", 4)
    finally:
        stop.return_value = True
        await runtime.aclose()
        await other.aclose()


async def test_one_cancelled_startup_waiter_does_not_drop_other_request_reservation(monkeypatch):
    async def free():
        return 6 * 1024.0
    coordinator = VramCoordinator(provider=free, total_provider=free)
    runtime = ManagedHTTPRuntime("http://waiters.test", vram_coordinator=coordinator)
    entered, release = asyncio.Event(), asyncio.Event()
    waiting = 0

    async def ready():
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            entered.set()
        await release.wait()

    monkeypatch.setattr(runtime, "_ensure_ready", ready)
    async def call():
        async with runtime.request_lease("m", estimated_vram_gb=4):
            pass
    first, second = [asyncio.create_task(call()) for _ in range(2)]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert coordinator._pending_used_mib() == 4 * 1024
        release.set()
        await second
        assert runtime.has_vram_reservation("m")
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await runtime.aclose()


@pytest.mark.parametrize("cancel", [False, True])
async def test_lost_post_transport_quarantines_runtime_until_verified_cleanup(cancel, monkeypatch):
    from fake_ollama.generic_openai_client import GenericOpenAIClient
    entered, release = asyncio.Event(), asyncio.Event()
    async def upstream(req):
        if req.method == "POST":
            entered.set()
            await release.wait()
            raise httpx.ReadTimeout("response lost", request=req)
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = DecisionClient(DecisionTarget(name="lost", models=["m"], base_url="http://lost.test"), client=http)
        task = asyncio.create_task(client.decide(request()))
        await asyncio.wait_for(entered.wait(), 2)
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel else httpx.ReadTimeout):
            await task
        assert client._ownership.uncertain_decisions
        with pytest.raises(LocalTargetResourceError, match="unconfirmed"):
            await client.decide(request())
        sibling = GenericOpenAIClient("http://lost.test", target_name="chat")
        with pytest.raises(LocalTargetResourceError, match="unconfirmed"):
            sibling._begin_request_lifecycle()
        with pytest.raises(LocalTargetResourceError, match="unconfirmed"):
            await client.aclose()
        assert not client._closed  # reload cannot discard the quarantine
        # Simulate verified ownership so this isolated test can recover.
        client._process, client._started_by_us = SimpleNamespace(returncode=None), True
        stop = AsyncMock(return_value=False)
        monkeypatch.setattr("fake_ollama.managed_http_runtime.terminate_process_tree", stop)
        assert not await client.stop_if_owned()
        assert client._ownership.uncertain_decisions
        stop.return_value = True
        assert await client.stop_if_owned()
        assert not client._ownership.uncertain_decisions
        await client.aclose()
        await sibling.aclose()


async def test_cancelled_post_drain_is_bounded_and_gates_already_queued_request(monkeypatch):
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def upstream(req):
        if req.method == "POST":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = DecisionClient(DecisionTarget(name="deadline", models=["m"],
            base_url="http://deadline.test", max_concurrent_requests=1,
            request_read_timeout_seconds=0, stop_timeout_seconds=.03), client=http)
        active = asyncio.create_task(client.decide(request()))
        await asyncio.wait_for(entered.wait(), 2)
        queued = asyncio.create_task(client.decide(request()))
        for _ in range(100):
            if client.queued_requests:
                break
            await asyncio.sleep(0)
        assert client.queued_requests == 1
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 2)
        with pytest.raises(LocalTargetResourceError, match="unconfirmed"):
            await asyncio.wait_for(queued, 2)
        assert stopped.is_set()
        assert client._ownership.uncertain_decisions
        assert client._active == client._request_refs == client._queued == 0
        client._process, client._started_by_us = SimpleNamespace(returncode=None), True
        monkeypatch.setattr("fake_ollama.managed_http_runtime.terminate_process_tree", AsyncMock(return_value=True))
        await client.aclose()


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
async def test_failure_before_sending_does_not_claim_unconfirmed_execution(error):
    async def upstream(req):
        if req.method == "POST":
            raise error("request was never sent", request=req)
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = DecisionClient(DecisionTarget(name="unsent", models=["m"], base_url="http://unsent.test"), client=http)
        with pytest.raises(error):
            await client.decide(request())
        assert not client._ownership.uncertain_decisions
        await client.aclose()


async def test_retiring_uncertain_owned_runtime_cleans_shared_process_after_sibling_drains(monkeypatch):
    from fake_ollama.generic_openai_client import GenericOpenAIClient
    decision = DecisionClient(DecisionTarget(name="retired", models=["m"], base_url="http://retired.test"))
    chat = GenericOpenAIClient("http://retired.test", target_name="chat")
    decision._process, decision._started_by_us = SimpleNamespace(returncode=None), True
    decision._ownership.uncertain_decisions = True
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("fake_ollama.managed_http_runtime.terminate_process_tree", stop)
    chat._active = 1
    with pytest.raises(LocalTargetResourceError, match="cleanup"):
        await decision.aclose()
    assert not stop.called
    assert not decision._closed
    chat._active = 0
    await decision.aclose()
    assert stop.await_count == 1
    assert decision._closed
    assert not chat._ownership.uncertain_decisions
    assert chat._ownership.process is None
    await chat.aclose()


@pytest.mark.parametrize("health", [
    {"status": "ok", "kind": "auto_think", "images": True},
    {"status": "ok", "prompt_attention": "causal", "images": True},
])
async def test_expired_health_refresh_restores_configured_thinking_and_images(health):
    observed = {"status": "ok", "kind": "instant", "images": False}
    def upstream(req):
        if req.url.path == "/health":
            return httpx.Response(200, json=observed.copy())
        return httpx.Response(200, json=response()) if req.method == "POST" else httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = DecisionClient(DecisionTarget(name="refresh", models=["m"], runtime="startlux",
            decision_capabilities={"modalities": ["text", "image"], "extensions": ["think_mode"],
                                   "think_modes": ["never", "auto"]}), client=http)
        await client.decide(request())
        assert client.capability_snapshot()["think_modes"] == ["never"]
        observed.clear()
        observed.update(health)
        client._probe_at = 0
        await client.decide(request(think_mode="auto", images=["data:image/png;base64,AAAA"]))
        assert client.capability_snapshot()["think_modes"] == ["never", "auto"]
        assert client.capability_snapshot()["modalities"] == ["text", "image"]
        with pytest.raises(DecisionValidationError):
            await client.decide(request(think_mode="always"))  # never expand configuration
        await client.aclose()


@pytest.mark.parametrize("health", [{}, {"status": "ok"}, {"status": "loading", "kind": "auto_think", "images": True}])
async def test_inconclusive_health_does_not_erase_known_restrictions(health):
    observed = {"status": "ok", "kind": "instant", "images": False}
    def upstream(req):
        return httpx.Response(200, json=observed.copy()) if req.url.path == "/health" else httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        client = DecisionClient(DecisionTarget(name="redacted", models=["m"], runtime="startlux",
            decision_capabilities={"modalities": ["text", "image"], "extensions": ["think_mode"],
                                   "think_modes": ["never", "auto"]}), client=http)
        await client.probe_capabilities()
        observed.clear()
        observed.update(health)
        await client.probe_capabilities(force=True)
        assert client.capability_snapshot()["think_modes"] == ["never"]
        assert client.capability_snapshot()["modalities"] == ["text"]
        await client.aclose()
