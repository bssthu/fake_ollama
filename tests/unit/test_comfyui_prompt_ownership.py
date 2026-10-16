import asyncio

import httpx
import pytest

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.vram import LocalTargetResourceError, VramCoordinator
import json
import uuid


async def test_cancelled_prompt_remains_tracked_until_backend_completion(monkeypatch):
    entered = asyncio.Event()
    complete = asyncio.Event()
    async def handler(request):
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "owned"})
        if request.url.path == "/history/owned":
            entered.set()
            await complete.wait()
            return httpx.Response(200, json={"owned": {"outputs": {"9": {"images": []}}, "status": {"completed": True}}})
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [[0, "owned", {}, {}]], "queue_pending": []})
        return httpx.Response(200, json={})
    client = ComfyUIClient("http://prompt-ownership.test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    task = asyncio.create_task(client._submit_and_collect({}, operation="test"))
    await entered.wait()
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "owned" in client._active_prompt_ids
    finally:
        complete.set()
        if hasattr(client, "recover_pending_prompts"):
            await client.recover_pending_prompts()
        await client.aclose()


@pytest.mark.parametrize("failure", ["timeout", "disconnect", "cancel", "lost_ack", "legacy_lost_ack"])
async def test_uncertain_prompt_blocks_shared_runtime_and_gpu_until_confirmed(failure):
    entered, reconciling, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    records = []
    history_calls = 0
    async def provider():
        return 24 * 1024
    coordinator = VramCoordinator(provider=provider)
    async def handler(request):
        nonlocal history_calls
        path = request.url.path
        if path == "/prompt":
            body = json.loads(request.content)
            identifier = "legacy" if failure == "legacy_lost_ack" else body["prompt_id"]
            records.append([0, identifier, {}, {"client_id": body["client_id"]}])
            if "lost_ack" in failure:
                raise httpx.ReadError("lost acknowledgement", request=request)
            return httpx.Response(200, json={"prompt_id": identifier})
        if path.startswith("/history/"):
            history_calls += 1
            if failure == "cancel" and history_calls == 1:
                entered.set()
                await asyncio.Event().wait()
            if failure in ("timeout", "disconnect") and history_calls == 1:
                error = httpx.ReadTimeout if failure == "timeout" else httpx.ReadError
                raise error("lost history", request=request)
            entry = {records[0][1]: {"status": {"completed": True}, "outputs": {}}} if finish.is_set() else {}
            return httpx.Response(200, json=entry)
        if path == "/queue":
            reconciling.set()
            await finish.wait()
            return httpx.Response(200, json={"queue_running": [] if finish.is_set() else records, "queue_pending": []})
        if path == "/history":
            return httpx.Response(200, json={records[0][1]: {"prompt": records[0], "status": {"completed": True}}})
        pytest.fail("Unexpected request " + path)
    base = "http://" + uuid.uuid4().hex + ".test"
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first = ComfyUIClient(base, target_name="first", client=transport, vram_coordinator=coordinator)
    second = ComfyUIClient(base, target_name="second", client=transport, vram_coordinator=coordinator)
    first._recovery_poll_seconds = 0
    task = asyncio.create_task(first._submit_and_collect({}, operation="test"))
    if failure == "cancel":
        await entered.wait()
        task.cancel()
    with pytest.raises((httpx.HTTPError, asyncio.CancelledError)):
        await task
    await reconciling.wait()
    assert first.prompt_snapshots()
    with pytest.raises(LocalTargetResourceError):
        await second._submit_and_collect({}, operation="test")
    with pytest.raises(LocalTargetResourceError):
        await coordinator.ensure_available(second, model="another", estimated_vram_gb=1)
    assert len(records) == 1
    assert not await first.stop_if_owned()
    assert not await first._release_for_vram(force=True)
    finish.set()
    assert await second.recover_pending_prompts()
    assert not first._active_prompt_ids
    assert not coordinator.execution_snapshots()
    await first.aclose()
    await second.aclose()


async def test_queued_cancel_deletes_only_own_prompt_and_checks_race():
    records, deletes = [], []
    async def handler(request):
        if request.url.path == "/prompt":
            body = json.loads(request.content)
            records.extend([[0, body["prompt_id"], {}, {"client_id": body["client_id"]}], [1, "external", {}, {"client_id": "external"}]])
            raise httpx.ReadError("lost acknowledgement", request=request)
        if request.url.path == "/queue":
            if request.method == "POST":
                identifiers = json.loads(request.content)["delete"]
                deletes.extend(identifiers)
                records[:] = [r for r in records if r[1] not in identifiers]
                return httpx.Response(200)
            return httpx.Response(200, json={"queue_running": [], "queue_pending": records})
        return httpx.Response(200, json={})
    client = ComfyUIClient("http://" + uuid.uuid4().hex + ".test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(httpx.ReadError):
        await client._submit_and_collect({}, operation="test")
    assert await client.recover_pending_prompts()
    assert len(deletes) == 1
    assert records == [[1, "external", {}, {"client_id": "external"}]]
    await client.aclose()


async def test_failed_reconciliation_stays_quarantined_and_can_retry():
    complete = False
    identifier = None
    async def handler(request):
        nonlocal identifier
        if request.url.path == "/prompt":
            identifier = json.loads(request.content)["prompt_id"]
            raise httpx.ReadError("lost acknowledgement", request=request)
        if complete:
            return httpx.Response(200, json={identifier: {"status": {"completed": True}}})
        return httpx.Response(503)
    client = ComfyUIClient("http://" + uuid.uuid4().hex + ".test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client._recovery_timeout_seconds = 0
    with pytest.raises(httpx.ReadError):
        await client._submit_and_collect({}, operation="test")
    assert not await client.recover_pending_prompts()
    assert client.prompt_snapshots()[0]["status"] == "quarantined"
    with pytest.raises(LocalTargetResourceError):
        await client._submit_and_collect({}, operation="test")
    complete = True
    client._recovery_timeout_seconds = 1
    assert await client.recover_pending_prompts()
    await client.aclose()
