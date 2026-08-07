"""Regression test for idle llama.cpp VRAM reclamation.
the coordinator must reclaim it via release_for_vram.

Production failure (Windows, before fix): llama.cpp targets were started
through ``create_subprocess_shell``, so ``self._process`` was the cmd.exe
wrapper. Once the wrapper detached the captured PID became dead while
``llama-server.exe`` lived on, and ``stop_if_owned`` bailed on the
"launcher process already exited" branch — coordinator then reported
0 GiB reclaimable and refused new admissions even though ~20 GiB was
clearly recoverable.

After the fix llama.cpp targets are launched via
``create_subprocess_exec`` (no wrapper); ``self._process`` is the real
server PID and termination succeeds.
"""

from __future__ import annotations

import time

import httpx
import pytest

from fake_ollama.llama_cpp_client import LlamaCppClient, _LoadedModel
from fake_ollama.vram import VramCoordinator


class _FakeProc:
    def __init__(self) -> None:
        self.returncode = None

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = 1

    async def wait(self) -> int:
        return self.returncode or 0


@pytest.mark.asyncio
async def test_idle_gemma_is_reclaimed_when_qwen_admits(monkeypatch: pytest.MonkeyPatch):
    free_mib = {"v": 4.45 * 1024.0}

    async def provider() -> float:
        return free_mib["v"]

    coord = VramCoordinator(provider=provider)

    async def fake_terminate(proc, *, timeout=10.0):
        proc.returncode = 0
        # Simulate the GPU memory being released.
        free_mib["v"] = 24.0 * 1024.0
        return True

    monkeypatch.setattr(
        "fake_ollama.llama_cpp_client.terminate_process_tree", fake_terminate
    )

    def qwen_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    gemma = LlamaCppClient(
        "http://127.0.0.1:21412",
        target_name="gemma",
        idle_timeout_seconds=900.0,
        vram_coordinator=coord,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200))
        ),
    )
    qwen = LlamaCppClient(
        "http://127.0.0.1:21441",
        target_name="qwen",
        idle_timeout_seconds=900.0,
        vram_coordinator=coord,
        client=httpx.AsyncClient(transport=httpx.MockTransport(qwen_handler)),
    )

    try:
        # Pre-load gemma state to mirror "already running for a while".
        gemma._process = _FakeProc()
        gemma._started_by_us = True
        idle_past = time.monotonic() - 150.0
        gemma._last_used = idle_past
        gemma._loaded_model = _LoadedModel(
            model="gemma-4-26B-A4B-it-uncensored-Q4_K_M",
            estimated_vram_gb=20.0,
            last_used_monotonic=idle_past,
        )

        # qwen request → coordinator must reclaim gemma.
        await qwen.chat(
            {"model": "qwen3.6-27b-uncensored-iq4xs", "messages": []},
            estimated_vram_gb=20.0,
        )
        assert qwen._loaded_model is not None
        assert gemma._loaded_model is None, (
            "gemma should have been reclaimed before qwen could admit"
        )
        assert free_mib["v"] == pytest.approx(24.0 * 1024.0)
    finally:
        await gemma.aclose()
        await qwen.aclose()
