"""Exercise ComfyUI admission through generation with real resource coordinators."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.vram import LocalTargetResourceError, MemoryCoordinator, VramCoordinator


class _Runtime:
    def __init__(self, *, shared: bool, free_vram: float = 22, free_ram: float = 32):
        self.shared = shared
        self.free_vram = free_vram
        self.free_ram = free_ram
        self.released_vram = 23
        self.released_ram = 80
        self.release_status = 200
        self.calls = []
        self.on_free = None
        self.on_prompt = None
        self.gpu = VramCoordinator(
            provider=self.vram_reading, total_provider=self.total_vram_reading,
        )
        self.ram = MemoryCoordinator(
            provider=self.ram_reading, total_provider=self.total_ram_reading,
        )
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        self.old = ComfyUIClient(
            "http://new.test" if shared else "http://old.test", target_name="old",
            client=self.http, vram_coordinator=self.gpu, memory_coordinator=self.ram,
        )
        self.new = ComfyUIClient(
            "http://new.test", target_name="new", client=self.http,
            vram_coordinator=self.gpu, memory_coordinator=self.ram,
        )
        self.old._mark_vram_reserved("old", 1 if shared else 16)
        self.old._mark_memory_reserved("old", 40)
        self.old._loaded_model.last_used_monotonic = time.monotonic() - 120
        self.old._last_used = time.monotonic() - 120

    async def vram_reading(self):
        return self.free_vram * 1024.0

    async def ram_reading(self):
        return self.free_ram * 1024.0

    async def total_vram_reading(self):
        return 24 * 1024.0

    async def total_ram_reading(self):
        return 96 * 1024.0

    async def handle(self, request):
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((host, path, body))
        if path == "/free":
            if self.on_free is not None:
                await self.on_free()
            if self.shared or host == "old.test":
                if self.release_status == 200:
                    self.free_vram = self.released_vram
                    self.free_ram = self.released_ram
                return httpx.Response(self.release_status, json={})
            return httpx.Response(200, json={})
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path == "/prompt":
            if self.on_prompt is not None:
                await self.on_prompt()
            return httpx.Response(200, json={"prompt_id": "ok"})
        if path == "/history/ok":
            return httpx.Response(200, json={"ok": {
                "outputs": {"9": {"images": [
                    {"filename": "out.png", "subfolder": "", "type": "output"},
                ]}},
                "status": {"status_str": "success"},
            }})
        if path == "/view":
            return httpx.Response(200, content=b"image")
        return httpx.Response(404)

    async def generate(self):
        return await self.new.generate_image(
            model="new", prompt="test", width=1024, height=1024, n=1, seed=1,
            steps=4, cfg=1, sampler_name="euler", scheduler="simple", denoise=1,
            estimated_vram_gb=20, estimated_memory_gb=40,
            request_vram_headroom_gb=3, min_free_vram_gb=2,
            vram_cleanup_policy="adaptive", exclusive_gpu=True,
        )

    def assert_no_pending_request(self):
        assert not self.gpu.has_pending(self.new.target_id, "new")
        assert not self.ram.has_pending(self.new.target_id, "new")
        assert not self.gpu.execution_snapshots()
        assert self.new.request_refs == 0

    async def close(self):
        await self.old.aclose()
        await self.new.aclose()
        await self.http.aclose()


@pytest.fixture
async def runtime_factory(monkeypatch):
    monkeypatch.setattr("fake_ollama.vram._POST_RELEASE_REFRESH_DELAYS_SECONDS", (0.0,))
    instances = []

    def create(**kwargs):
        runtime = _Runtime(**kwargs)
        instances.append(runtime)
        return runtime

    yield create
    for runtime in instances:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("free_vram", [8, 1])
async def test_cold_load_reclaims_other_runtime_after_local_cleanup(runtime_factory, free_vram):
    runtime = runtime_factory(shared=False, free_vram=free_vram, free_ram=80)

    images = await runtime.generate()

    assert images[0].data == b"image"
    assert [(host, path) for host, path, _ in runtime.calls if path in {"/free", "/prompt"}] == [
        ("new.test", "/free"), ("old.test", "/free"), ("new.test", "/prompt"),
    ]
    assert not runtime.old.has_vram_reservation("old")
    assert not runtime.gpu.execution_snapshots()


@pytest.mark.asyncio
@pytest.mark.parametrize("release_status,released_vram", [(500, 23), (200, 15)])
async def test_cold_load_rejects_when_other_runtime_cannot_free_enough(
    runtime_factory, release_status, released_vram,
):
    runtime = runtime_factory(shared=False, free_vram=8, free_ram=80)
    runtime.release_status = release_status
    runtime.released_vram = released_vram

    with pytest.raises(LocalTargetResourceError):
        await runtime.generate()

    assert any(host == "old.test" and path == "/free" for host, path, _ in runtime.calls)
    assert not any(path == "/prompt" for _, path, _ in runtime.calls)
    runtime.assert_no_pending_request()


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_old_request", [False, True])
async def test_model_switch_reclaims_shared_ram_before_prompt(runtime_factory, queued_old_request):
    runtime = runtime_factory(shared=True)
    if queued_old_request:
        runtime.old._begin_request_lifecycle()

    async def verify_reservations():
        # A full RAM/cache release also invalidates the earlier GPU admission.
        assert runtime.gpu.has_pending(runtime.new.target_id, "new")
        assert runtime.ram.has_pending(runtime.new.target_id, "new")

    runtime.on_prompt = verify_reservations
    try:
        images = await runtime.generate()
    finally:
        if queued_old_request:
            runtime.old._end_request_lifecycle()

    assert images[0].data == b"image"
    free_calls = [body for _, path, body in runtime.calls if path == "/free"]
    assert free_calls == [{"unload_models": True, "free_memory": True}]
    assert not runtime.old.has_memory_reservation("old")


@pytest.mark.asyncio
@pytest.mark.parametrize("release_status,released_ram", [(500, 80), (200, 32)])
async def test_shared_ram_reclaim_failure_never_submits_prompt(
    runtime_factory, release_status, released_ram,
):
    runtime = runtime_factory(shared=True)
    runtime.release_status = release_status
    runtime.released_ram = released_ram

    with pytest.raises(LocalTargetResourceError, match="Insufficient system RAM"):
        await runtime.generate()

    assert any(path == "/free" for _, path, _ in runtime.calls)
    assert not any(path == "/prompt" for _, path, _ in runtime.calls)
    runtime.assert_no_pending_request()


@pytest.mark.asyncio
async def test_cancelled_shared_ram_reclaim_releases_admission(runtime_factory):
    runtime = runtime_factory(shared=True)
    entered = asyncio.Event()

    async def block_release():
        entered.set()
        await asyncio.Event().wait()

    runtime.on_free = block_release
    task = asyncio.create_task(runtime.generate())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    runtime.assert_no_pending_request()
    runtime.on_free = None
    assert (await runtime.generate())[0].data == b"image"


@pytest.mark.asyncio
async def test_shared_ram_release_rechecks_invalidated_vram(runtime_factory):
    runtime = runtime_factory(shared=True)
    runtime.released_vram = 8

    with pytest.raises(LocalTargetResourceError, match="Insufficient GPU VRAM"):
        await runtime.generate()

    assert any(path == "/free" for _, path, _ in runtime.calls)
    assert not any(path == "/prompt" for _, path, _ in runtime.calls)
    runtime.assert_no_pending_request()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["vram", "memory"])
async def test_admission_reclaim_permission_is_limited_to_owner_task(runtime_factory, resource):
    runtime = runtime_factory(shared=True)
    release = getattr(runtime.old, f"_release_for_{resource}")
    runtime.new._begin_request_lifecycle()
    lease = await runtime.gpu.acquire_execution(
        runtime.new, model="new", workload_key="new", exclusive=True,
    )
    try:
        async with lease:
            assert not await release()
            with lease.admission():
                # A monitor or unrelated request must not borrow this permission.
                assert not await asyncio.create_task(release())
                runtime.old._active = 1
                assert not await release()
                runtime.old._active = 0
                runtime.old._idle_timeout = 60
                runtime.old._started_by_us = True

                async def unexpected_stop():
                    pytest.fail("idle shutdown must remain blocked during admission")

                runtime.old.stop_if_owned = unexpected_stop
                await runtime.old.stop_if_idle()
                assert not runtime.calls
            assert not await release()
            with lease.admission():
                assert await release()
    finally:
        runtime.new._end_request_lifecycle()
        runtime.old._started_by_us = False
        # Restore the method before fixture cleanup.
        runtime.old.__dict__.pop("stop_if_owned", None)


@pytest.mark.asyncio
async def test_shared_runtime_queues_generation_until_active_request_finishes(runtime_factory):
    runtime = runtime_factory(shared=True)
    runtime.old._begin_request_lifecycle()
    lease = await runtime.gpu.acquire_execution(
        runtime.old, model="old", workload_key="old", exclusive=True,
    )
    task = None
    try:
        async with lease:
            task = asyncio.create_task(runtime.generate())
            # The lease serializes the runtime even before _active is incremented.
            await asyncio.sleep(0)
            assert not task.done()
            assert not runtime.calls
            runtime.old._active = 1
            assert not await runtime.new._release_for_memory()
            assert not await runtime.new._release_for_vram()
            runtime.old._active = 0
            runtime.old._end_request_lifecycle()
        assert (await asyncio.wait_for(task, timeout=1))[0].data == b"image"
    finally:
        runtime.old._active = 0
        if runtime.old.request_refs:
            runtime.old._end_request_lifecycle()
        if task is not None and not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("released", [True, False])
async def test_image_api_returns_success_only_after_resource_admission(
    runtime_factory, shared, released,
):
    from fake_ollama.config import Settings
    from fake_ollama.server import create_app

    runtime = runtime_factory(
        shared=shared, free_vram=22 if shared else 8, free_ram=32 if shared else 80,
    )
    runtime.release_status = 200 if released else 500
    settings = Settings(
        comfyui_targets=[{"name": "new", "model": "new", "base_url": "http://new.test"}],
        api_interfaces=[{
            "name": "api", "host": "127.0.0.1", "port": 21435,
            "access_tokens": ["test-token"],
            "exposed_models": [{"model": "new", "target": "new", "alias": "new"}],
        }],
        model_profiles=[{
            "model": "new", "target": "new", "capabilities": ["image_generation"],
            "estimated_vram_gb": 20, "estimated_memory_gb": 40,
            "request_vram_headroom_gb": 3, "min_free_vram_gb": 2,
            "vram_cleanup_policy": "adaptive", "exclusive_gpu": True,
        }],
    )
    app = create_app(settings)
    app.state.comfyui_clients = {"new": runtime.new}
    app.state.vram_coordinator = runtime.gpu
    app.state.memory_coordinator = runtime.ram
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver:21435",
    ) as client:
        response = await client.post(
            "/v1/images/generations", headers={"x-api-key": "test-token"},
            json={"model": "new", "prompt": "test"},
        )

    assert response.status_code == (200 if released else 503), response.text
    if released:
        assert response.json()["data"][0]["b64_json"] == "aW1hZ2U="
    else:
        assert not any(path == "/prompt" for _, path, _ in runtime.calls)
        runtime.assert_no_pending_request()
