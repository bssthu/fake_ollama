"""Tests for the lifecycle-aware Ollama target client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from fake_ollama.ollama_client import OllamaClient
from fake_ollama.llama_cpp_client import LlamaCppClient
from fake_ollama.vram import LocalTargetResourceError, VramCoordinator


class _FakeProcess:
    returncode = None

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:  # pragma: no cover - fallback path
        self.returncode = 1

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.mark.asyncio
async def test_ollama_client_auto_starts_before_chat(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {"started": False, "commands": []}

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FakeProcess:
        state["commands"].append(command)
        state["started"] = True
        return _FakeProcess()

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )
    monkeypatch.setattr(
        "fake_ollama.ollama_client.terminate_process_tree",
        lambda process, *, timeout=10.0: _async_true(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            if not state["started"]:
                raise httpx.ConnectError("daemon down", request=request)
            return httpx.Response(200, json={"version": "0.0.0-test"})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            assert body["stream"] is False
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        "http://127.0.0.1:11434",
        auto_start=True,
        start_command="ollama serve",
        startup_timeout_seconds=1,
        client=async_client,
    )

    try:
        resp = await client.chat({"model": "qwen3.5:2b", "messages": []})
    finally:
        await client.aclose()

    assert resp["message"]["content"] == "ok"
    assert state["commands"] == ["ollama serve"]


@pytest.mark.asyncio
async def test_ollama_client_does_not_start_when_disabled(monkeypatch: pytest.MonkeyPatch):
    commands: list[str] = []

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FakeProcess:
        commands.append(command)
        return _FakeProcess()

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon down", request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        "http://127.0.0.1:11434",
        auto_start=False,
        start_command="ollama serve",
        client=async_client,
    )

    try:
        with pytest.raises(httpx.ConnectError):
            await client.chat({"model": "qwen3.5:2b", "messages": []})
    finally:
        await client.aclose()

    assert commands == []


@pytest.mark.asyncio
async def test_ollama_client_does_not_auto_start_during_shutdown(monkeypatch: pytest.MonkeyPatch):
    commands: list[str] = []

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FakeProcess:
        commands.append(command)
        return _FakeProcess()

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon down", request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        "http://127.0.0.1:11434",
        auto_start=True,
        start_command="ollama serve",
        client=async_client,
    )
    client.begin_shutdown()

    try:
        with pytest.raises(httpx.ConnectError, match="shutting down"):
            await client.chat({"model": "qwen3.5:2b", "messages": []})
    finally:
        await client.aclose()

    assert commands == []


@pytest.mark.asyncio
async def test_llama_cpp_client_does_not_auto_start_during_shutdown(monkeypatch: pytest.MonkeyPatch):
    commands: list[str] = []

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FakeProcess:
        commands.append(command)
        return _FakeProcess()

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon down", request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LlamaCppClient(
        "http://127.0.0.1:21436",
        auto_start=True,
        start_command="llama-server --model qwen.gguf",
        client=async_client,
    )
    client.begin_shutdown()

    try:
        with pytest.raises(httpx.ConnectError, match="shutting down"):
            await client.chat({"model": "qwen", "messages": []})
    finally:
        await client.aclose()

    assert commands == []


@pytest.mark.asyncio
async def test_llama_cpp_idle_stop_ignores_request_waiting_for_startup(
    monkeypatch: pytest.MonkeyPatch,
):
    state = {"started": False, "chat_called": False}
    health_gate = asyncio.Event()
    stopped: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(argv, **kwargs):
        state["started"] = True
        return _FakeProcess()

    async def fake_terminate_process_tree(
        process: _FakeProcess, *, timeout: float = 10.0
    ) -> bool:
        stopped.append(process)
        process.returncode = 0
        return True

    monkeypatch.setattr(
        "fake_ollama.llama_cpp_client.create_managed_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "fake_ollama.llama_cpp_client.terminate_process_tree",
        fake_terminate_process_tree,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if not state["started"]:
                raise httpx.ConnectError("daemon down", request=request)
            await health_gate.wait()
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions":
            state["chat_called"] = True
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = LlamaCppClient(
        "http://127.0.0.1:21436",
        auto_start=True,
        start_argv=["llama-server", "--model", "qwen.gguf"],
        idle_timeout_seconds=1.0,
        startup_timeout_seconds=5.0,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client._last_used = time.monotonic() - 60.0

    task = asyncio.create_task(client.chat({"model": "qwen", "messages": []}))
    try:
        for _ in range(100):
            if state["started"]:
                break
            await asyncio.sleep(0.01)
        assert state["started"] is True

        await client.stop_if_idle()
        assert stopped == []
        assert not task.done()

        health_gate.set()
        resp = await task
        assert resp["choices"][0]["message"]["content"] == "ok"
    finally:
        if not health_gate.is_set():
            health_gate.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        client._started_by_us = False
        await client.aclose()

    assert state["chat_called"] is True


async def _async_true() -> bool:
    return True


async def _async_false() -> bool:
    return False


@pytest.mark.asyncio
async def test_llama_cpp_client_stops_owned_process_tree(monkeypatch: pytest.MonkeyPatch):
    stopped: list[_FakeProcess] = []

    async def fake_terminate_process_tree(process: _FakeProcess, *, timeout: float = 10.0) -> bool:
        stopped.append(process)
        process.returncode = 0
        return True

    monkeypatch.setattr(
        "fake_ollama.llama_cpp_client.terminate_process_tree",
        fake_terminate_process_tree,
    )
    client = LlamaCppClient("http://127.0.0.1:21436")
    fake_process = _FakeProcess()
    client._process = fake_process
    client._started_by_us = True
    client._mark_vram_reserved("qwen", 20)

    try:
        stopped_ok = await client.stop_if_owned()
    finally:
        await client.aclose()

    assert stopped_ok is True
    assert stopped == [fake_process]
    assert client.has_vram_reservation("qwen") is False


@pytest.mark.asyncio
async def test_llama_cpp_client_keeps_vram_reservation_when_process_tree_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "fake_ollama.llama_cpp_client.terminate_process_tree",
        lambda process, *, timeout=10.0: _async_false(),
    )
    client = LlamaCppClient("http://127.0.0.1:21436")
    client._process = _FakeProcess()
    client._started_by_us = True
    client._mark_vram_reserved("qwen", 20)

    try:
        stopped_ok = await client.stop_if_owned()
    finally:
        client._started_by_us = False
        await client.aclose()

    assert stopped_ok is False
    assert client.has_vram_reservation("qwen") is True


@pytest.mark.asyncio
async def test_ollama_client_unloads_idle_model_before_loading_new_model(
    monkeypatch: pytest.MonkeyPatch,
):
    free_vram_values = iter([1024.0, 3072.0])

    async def free_vram_mib() -> float:
        return next(free_vram_values, 3072.0)

    monkeypatch.setattr(
        "fake_ollama.vram._POST_RELEASE_REFRESH_DELAYS_SECONDS", (0.0,)
    )

    coordinator = VramCoordinator(provider=free_vram_mib)
    unloaded: list[str] = []

    def old_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            unloaded.append(json.loads(request.content)["model"])
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        raise AssertionError(f"unexpected old path: {request.url.path}")

    def new_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                },
            )
        raise AssertionError(f"unexpected new path: {request.url.path}")

    old_client = OllamaClient(
        "http://old.local",
        target_name="old",
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(old_handler)),
    )
    new_client = OllamaClient(
        "http://new.local",
        target_name="new",
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(new_handler)),
    )
    old_client._mark_vram_reserved("old-model:latest", 2)
    old_client._loaded_models["old-model:latest"].last_used_monotonic = (
        time.monotonic() - 61.0
    )

    try:
        resp = await new_client.chat(
            {"model": "new-model", "messages": []}, estimated_vram_gb=2
        )
    finally:
        await new_client.aclose()
        await old_client.aclose()

    assert resp["message"]["content"] == "ok"
    assert unloaded == ["old-model:latest"]
    assert old_client.has_vram_reservation("old-model:latest") is False
    assert new_client.has_vram_reservation("new-model") is True


@pytest.mark.asyncio
async def test_ollama_client_rechecks_vram_after_release_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
):
    async def free_vram_mib() -> float:
        return 1024.0

    monkeypatch.setattr(
        "fake_ollama.vram._POST_RELEASE_REFRESH_DELAYS_SECONDS", (0.0,)
    )
    coordinator = VramCoordinator(provider=free_vram_mib)
    chat_called = False

    def old_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        raise AssertionError(f"unexpected old path: {request.url.path}")

    def new_handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_called
        if request.url.path == "/api/chat":
            chat_called = True
        return httpx.Response(200, json={"version": "0.0.0-test"})

    old_client = OllamaClient(
        "http://old.local",
        target_name="old",
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(old_handler)),
    )
    new_client = OllamaClient(
        "http://new.local",
        target_name="new",
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(new_handler)),
    )
    old_client._mark_vram_reserved("old-model", 2)
    old_client._loaded_models["old-model"].last_used_monotonic = time.monotonic() - 61.0

    try:
        with pytest.raises(LocalTargetResourceError, match="rechecked current free VRAM"):
            await new_client.chat(
                {"model": "new-model", "messages": []}, estimated_vram_gb=2
            )
    finally:
        await new_client.aclose()
        await old_client.aclose()

    assert chat_called is False


@pytest.mark.asyncio
async def test_ollama_client_reports_insufficient_vram_before_request():
    async def free_vram_mib() -> float:
        return 512.0

    coordinator = VramCoordinator(provider=free_vram_mib)
    chat_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_called
        if request.url.path == "/api/chat":
            chat_called = True
        return httpx.Response(200, json={"version": "0.0.0-test"})

    client = OllamaClient(
        "http://new.local",
        target_name="new",
        vram_coordinator=coordinator,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        with pytest.raises(LocalTargetResourceError, match="Insufficient GPU VRAM"):
            await client.chat(
                {"model": "new-model", "messages": []}, estimated_vram_gb=4
            )
    finally:
        await client.aclose()

    assert chat_called is False
