from __future__ import annotations

import json
import asyncio
from typing import Any, Optional

import httpx
import pytest

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.vram import LocalTargetResourceError, MemoryCoordinator, VramCoordinator


def _provider(state: dict[str, float]):
    async def read() -> Optional[float]:
        return state["mib"]

    return read


def _constant(value: float):
    async def read() -> Optional[float]:
        return value

    return read


@pytest.mark.asyncio
async def test_resident_request_with_low_headroom_adaptively_unloads() -> None:
    free = {"mib": 5 * 1024.0}
    coordinator = VramCoordinator(
        provider=_provider(free), total_provider=_constant(24 * 1024.0)
    )

    class Participant:
        target_id = "comfyui:h3"
        vram_runtime_group = "gpu:0|comfyui:http://127.0.0.1:21481"
        active_requests = 0
        resident = True

        def has_vram_reservation(self, model: str) -> bool:
            return self.resident

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            return []

    participant = Participant()
    coordinator.register(participant)
    cleanup_calls = 0

    async def cleanup() -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        participant.resident = False
        free["mib"] = 22 * 1024.0
        return True

    lease = await coordinator.acquire_execution(
        participant,
        model="minimax-h3-768p",
        workload_key="h3-default",
        request_headroom_gb=6,
        min_free_vram_gb=2,
        cleanup_policy="adaptive",
        exclusive=True,
        cleanup=cleanup,
    )
    async with lease:
        assert lease.cleaned_for_headroom is True
    assert cleanup_calls == 1


@pytest.mark.asyncio
async def test_resident_request_with_enough_headroom_keeps_model() -> None:
    coordinator = VramCoordinator(
        provider=_constant(10 * 1024.0),
        total_provider=_constant(24 * 1024.0),
    )

    class Participant:
        target_id = "comfyui:h3"
        vram_runtime_group = "gpu:0|h3"
        active_requests = 0

        def has_vram_reservation(self, model: str) -> bool:
            return True

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            return []

    participant = Participant()
    coordinator.register(participant)
    cleanup_calls = 0

    async def cleanup() -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return True

    lease = await coordinator.acquire_execution(
        participant,
        model="minimax-h3-768p",
        workload_key="h3-default",
        request_headroom_gb=6,
        min_free_vram_gb=2,
        cleanup_policy="adaptive",
        cleanup=cleanup,
    )
    async with lease:
        assert lease.cleaned_for_headroom is False
    assert cleanup_calls == 0


@pytest.mark.asyncio
async def test_cleanup_failure_rejects_before_comfy_prompt() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/free":
            return httpx.Response(500, text="cannot unload")
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "must-not-run"})
        return httpx.Response(404)

    coordinator = VramCoordinator(
        provider=_constant(5 * 1024.0),
        total_provider=_constant(24 * 1024.0),
    )
    client = ComfyUIClient(
        "http://comfy.test",
        target_name="h3",
        workflow_config={
            "default_width": 1024,
            "default_height": 1024,
        },
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client._mark_vram_reserved("z-image-turbo", 16.0)
    try:
        with pytest.raises(LocalTargetResourceError, match="No prompt was submitted"):
            await client.generate_image(
                model="z-image-turbo",
                prompt="test",
                width=1024,
                height=1024,
                n=1,
                seed=1,
                steps=8,
                cfg=1.0,
                sampler_name="res_multistep",
                scheduler="simple",
                denoise=1.0,
                estimated_vram_gb=16.0,
                request_vram_headroom_gb=6.0,
                min_free_vram_gb=2.0,
                vram_cleanup_policy="adaptive",
            )
    finally:
        await client.aclose()

    assert "/free" in calls
    assert "/prompt" not in calls


@pytest.mark.asyncio
async def test_shared_comfy_runtime_unload_invalidates_all_target_reservations() -> None:
    free_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/free":
            free_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={})
        return httpx.Response(404)

    vram = VramCoordinator(
        provider=_constant(20 * 1024.0),
        total_provider=_constant(24 * 1024.0),
    )
    memory = MemoryCoordinator(
        provider=_constant(64 * 1024.0),
        total_provider=_constant(96 * 1024.0),
    )
    transport = httpx.MockTransport(handler)
    qwen = ComfyUIClient(
        "http://127.0.0.1:21480",
        target_name="qwen",
        vram_coordinator=vram,
        memory_coordinator=memory,
        client=httpx.AsyncClient(transport=transport),
    )
    sense = ComfyUIClient(
        "http://127.0.0.1:21480",
        target_name="sense",
        vram_coordinator=vram,
        memory_coordinator=memory,
        client=httpx.AsyncClient(transport=transport),
    )
    qwen._mark_vram_reserved("qwen-image", 20.0)
    qwen._mark_memory_reserved("qwen-image", 10.0)
    sense._mark_vram_reserved("sensenova", 13.0)
    sense._mark_memory_reserved("sensenova", 30.0)

    try:
        assert qwen.vram_runtime_group == sense.vram_runtime_group
        assert await qwen._adaptive_unload_gpu()
        assert not qwen.has_vram_reservation("qwen-image")
        assert not sense.has_vram_reservation("sensenova")
        assert qwen.has_memory_reservation("qwen-image")
        assert sense.has_memory_reservation("sensenova")

        assert await sense._free_memory()
        assert not qwen.has_memory_reservation("qwen-image")
        assert not sense.has_memory_reservation("sensenova")
    finally:
        await qwen.aclose()
        await sense.aclose()

    assert free_bodies == [
        {"unload_models": True, "free_memory": False},
        {"unload_models": True, "free_memory": True},
    ]


@pytest.mark.asyncio
async def test_native_batch_scales_headroom_but_client_loop_does_not() -> None:
    client = ComfyUIClient(
        "http://comfy.test",
        workflow_config={
            "default_width": 1024,
            "default_height": 1024,
        },
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    try:
        native = client._workflows["t2i"]
        looped = client._workflows["i2i"]
        assert native is not None and looped is not None
        native_headroom, _ = client._vram_workload(
            native,
            mode="t2i",
            model="z-image-turbo",
            n=2,
            params={"width": 1024, "height": 1024},
            base_headroom_gb=2.0,
        )
        looped_headroom, _ = client._vram_workload(
            looped,
            mode="i2i",
            model="z-image-turbo",
            n=2,
            params={"width": 1024, "height": 1024},
            base_headroom_gb=2.0,
        )
    finally:
        await client.aclose()

    assert native_headroom == 4.0
    assert looped_headroom == 2.0


@pytest.mark.asyncio
async def test_exclusive_video_lease_queues_other_runtime() -> None:
    coordinator = VramCoordinator(
        provider=_constant(20 * 1024.0),
        total_provider=_constant(24 * 1024.0),
    )

    class Participant:
        active_requests = 0

        def __init__(self, name: str) -> None:
            self.target_id = name
            self.vram_runtime_group = f"gpu:0|{name}"

        def has_vram_reservation(self, model: str) -> bool:
            return False

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            return []

    first = Participant("h3")
    second = Participant("joyai")
    coordinator.register(first)
    coordinator.register(second)
    lease1 = await coordinator.acquire_execution(
        first,
        model="h3",
        workload_key="h3",
        request_headroom_gb=6,
        min_free_vram_gb=2,
        exclusive=True,
    )
    await lease1.__aenter__()
    waiting = asyncio.create_task(
        coordinator.acquire_execution(
            second,
            model="joyai",
            workload_key="joyai",
            request_headroom_gb=4,
            min_free_vram_gb=2,
            exclusive=True,
        )
    )
    await asyncio.sleep(0)
    assert not waiting.done()
    await lease1.__aexit__(None, None, None)
    lease2 = await asyncio.wait_for(waiting, timeout=1.0)
    async with lease2:
        pass


@pytest.mark.asyncio
async def test_exclusive_execution_blocks_other_local_model_load_admission() -> None:
    coordinator = VramCoordinator(
        provider=_constant(20 * 1024.0),
        total_provider=_constant(24 * 1024.0),
    )

    class Participant:
        active_requests = 0

        def __init__(self, name: str, group: str) -> None:
            self.target_id = name
            self.vram_runtime_group = group
            self.resident = False

        def has_vram_reservation(self, model: str) -> bool:
            return self.resident

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            return []

    h3 = Participant("h3", "gpu:0|h3")
    text = Participant("llama", "llama")
    text.resident = True
    coordinator.register(h3)
    coordinator.register(text)
    lease = await coordinator.acquire_execution(
        h3,
        model="h3",
        workload_key="h3",
        request_headroom_gb=6,
        min_free_vram_gb=2,
        exclusive=True,
    )
    await lease.__aenter__()
    with pytest.raises(LocalTargetResourceError):
        await coordinator.ensure_available(
            text, model="text", estimated_vram_gb=1
        )
    await lease.__aexit__(None, None, None)
    await coordinator.ensure_available(text, model="text", estimated_vram_gb=1)


@pytest.mark.asyncio
async def test_safety_floor_interrupts_active_execution() -> None:
    free = {"mib": 10 * 1024.0}
    coordinator = VramCoordinator(
        provider=_provider(free), total_provider=_constant(24 * 1024.0)
    )

    class Participant:
        target_id = "h3"
        vram_runtime_group = "gpu:0|h3"
        active_requests = 0

        def has_vram_reservation(self, model: str) -> bool:
            return True

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            return []

    participant = Participant()
    coordinator.register(participant)
    interrupted = asyncio.Event()

    async def interrupt() -> bool:
        interrupted.set()
        return True

    lease = await coordinator.acquire_execution(
        participant,
        model="h3",
        workload_key="h3",
        request_headroom_gb=6,
        min_free_vram_gb=2,
        interrupt=interrupt,
    )
    await lease.__aenter__()
    free["mib"] = 1024.0
    await asyncio.wait_for(interrupted.wait(), timeout=1.5)
    assert lease.breached
    await lease.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_two_h3_videos_unload_between_runs_when_first_left_low_headroom() -> None:
    free = {"mib": 22 * 1024.0}
    events: list[str] = []
    prompt_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal prompt_count
        path = request.url.path
        if request.method == "GET" and path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.method == "POST" and path == "/free":
            events.append("free")
            body = json.loads(request.content)
            assert body == {"unload_models": True, "free_memory": False}
            free["mib"] = 22 * 1024.0
            return httpx.Response(200, json={})
        if request.method == "POST" and path == "/prompt":
            prompt_count += 1
            events.append(f"prompt-{prompt_count}")
            if prompt_count == 1:
                free["mib"] = 5 * 1024.0
            return httpx.Response(200, json={"prompt_id": f"pid-{prompt_count}"})
        if request.method == "GET" and path.startswith("/history/pid-"):
            prompt_id = path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "outputs": {
                            "999": {
                                "videos": [
                                    {
                                        "filename": f"{prompt_id}.mp4",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                        "status": {"status_str": "success"},
                    }
                },
            )
        if request.method == "GET" and path == "/view":
            return httpx.Response(200, content=b"video")
        return httpx.Response(404)

    coordinator = VramCoordinator(
        provider=_provider(free), total_provider=_constant(24 * 1024.0)
    )
    client = ComfyUIClient(
        "http://127.0.0.1:21481",
        target_name="minimax-h3",
        workflow_config={
            "preset": "minimax_h3",
            "default_width": 1344,
            "default_height": 768,
            "default_num_frames": 124,
            "poll_interval_seconds": 0.001,
        },
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def run_once(seed: int) -> None:
        videos = await client.generate_video(
            model="minimax-h3-768p",
            prompt="structured prompt",
            width=1344,
            height=768,
            n=1,
            seed=seed,
            steps=20,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            num_frames=124,
            frame_rate=24.0,
            prefetch_count=1,
            estimated_vram_gb=21.0,
            request_vram_headroom_gb=6.0,
            min_free_vram_gb=2.0,
            vram_cleanup_policy="adaptive",
            exclusive_gpu=True,
            video_mode="t2v",
        )
        assert videos[0].data == b"video"

    try:
        await run_once(1)
        await run_once(2)
    finally:
        await client.aclose()

    assert events == ["prompt-1", "free", "prompt-2"]
