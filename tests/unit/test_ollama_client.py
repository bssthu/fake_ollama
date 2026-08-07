"""Unit tests for the lifecycle-aware Ollama target client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from fake_ollama.generic_openai_client import GenericOpenAIClient
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
async def test_ollama_stream_counts_output_as_recent_vram_activity(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "fake_ollama.ollama_client.time.monotonic", lambda: clock["now"]
    )

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"message":{"role":"assistant","content":"a"},"done":false}\n'
            clock["now"] = 250.0
            yield b'{"done":true}\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        if request.url.path == "/api/chat":
            return httpx.Response(200, stream=_Stream())
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = OllamaClient(
        "http://127.0.0.1:11434",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        chunks = [
            line
            async for line in client.stream_chat(
                {"model": "m", "messages": []}, estimated_vram_gb=1.0
            )
        ]
        assert chunks
        assert client._loaded_models["m"].last_used_monotonic == 250.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_llama_cpp_stream_counts_output_as_recent_vram_activity(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "fake_ollama.llama_cpp_client.time.monotonic", lambda: clock["now"]
    )

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            clock["now"] = 250.0
            yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, stream=_Stream())
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = LlamaCppClient(
        "http://127.0.0.1:21436",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        chunks = [
            line
            async for line in client.stream_chat(
                {"model": "m", "messages": []}, estimated_vram_gb=1.0
            )
        ]
        assert chunks
        assert client._loaded_model is not None
        assert client._loaded_model.last_used_monotonic == 250.0
    finally:
        await client.aclose()


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
async def test_unused_generic_openai_target_does_not_run_stop_command_on_close(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FakeProcess:
        calls.append(command)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create_subprocess_shell)
    client = GenericOpenAIClient(
        "http://127.0.0.1:8062",
        target_name="unused-wsl-target",
        stop_command="wsl -d Ubuntu-24.04 -- bash stop.sh",
    )

    await client.aclose()

    assert calls == []


@pytest.mark.asyncio
async def test_llama_cpp_idle_stop_command_does_not_repeat_after_success(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FakeProcess:
        calls.append(command)
        proc = _FakeProcess()
        proc.returncode = 0
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create_subprocess_shell)
    client = LlamaCppClient(
        "http://127.0.0.1:21436",
        stop_command="stop-vllm",
        idle_timeout_seconds=1.0,
    )
    client._mark_vram_reserved("qwen", 4.0)
    assert client._loaded_model is not None
    client._loaded_model.last_used_monotonic = time.monotonic() - 60.0

    try:
        await client.stop_if_idle()
        assert calls == ["stop-vllm"]
        assert client._loaded_model is None

        await client.stop_if_idle()
        assert calls == ["stop-vllm"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_shutdown_stop_command_is_not_repeated_by_concurrent_close(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    stop_waiting = asyncio.Event()
    finish_stop = asyncio.Event()

    class _FailingStopProcess:
        returncode = 1

        async def wait(self) -> int:
            stop_waiting.set()
            await finish_stop.wait()
            return self.returncode

    async def fake_create_subprocess_shell(command: str, **kwargs: Any) -> _FailingStopProcess:
        calls.append(command)
        return _FailingStopProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create_subprocess_shell)
    client = GenericOpenAIClient(
        "http://127.0.0.1:8062",
        target_name="used-wsl-target",
        stop_command="wsl -d Ubuntu-24.04 -- bash stop.sh",
    )
    client._begin_request_lifecycle()
    client._end_request_lifecycle()
    client.begin_shutdown()

    signal_stop = asyncio.create_task(client.stop_if_owned())
    await stop_waiting.wait()
    lifespan_close = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)
    finish_stop.set()

    assert await signal_stop is False
    await lifespan_close
    assert calls == ["wsl -d Ubuntu-24.04 -- bash stop.sh"]


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
        with pytest.raises(LocalTargetResourceError, match="rechecked current free resource"):
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


# ---------------------------------------------------------------------------
# pre-mark VRAM: model visible in dashboard during in-flight non-stream chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llama_cpp_chat_marks_vram_before_response_arrives():
    """_loaded_model must appear in loaded_model_snapshots() while the POST is
    still in-flight, before any response headers arrive.

    Regression: llama-server's non-stream path withholds all response headers
    until after prefill+decode finishes (needs Content-Length). Any
    mark-after-status-code scheme stays invisible to the dashboard for the
    entire generation — which can be minutes on a large-prompt analysis task.
    """
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        in_flight.set()
        await release.wait()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        task = asyncio.create_task(
            client.chat({"model": "m", "messages": []}, estimated_vram_gb=10.0)
        )
        await asyncio.wait_for(in_flight.wait(), timeout=2.0)

        # POST is blocked inside the transport — no response headers yet.
        # The dashboard snapshot must already show the model as loaded.
        snaps = client.loaded_model_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["model"] == "m"
        assert snaps[0]["estimated_vram_gb"] == 10.0

        release.set()
        await asyncio.wait_for(task, timeout=2.0)
        # Snapshot still present after completion.
        assert len(client.loaded_model_snapshots()) == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_chat_marks_vram_before_response_arrives():
    """_loaded_models must be populated before the HTTP response arrives.

    Same class of bug as the llama.cpp variant: non-stream Ollama chat also
    withholds response headers on long generations.
    """
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        in_flight.set()
        await release.wait()
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}, "done": True},
        )

    client = OllamaClient(
        "http://127.0.0.1:11434",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        task = asyncio.create_task(
            client.chat({"model": "m", "messages": []}, estimated_vram_gb=10.0)
        )
        await asyncio.wait_for(in_flight.wait(), timeout=2.0)

        snaps = client.loaded_model_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["model"] == "m"
        assert snaps[0]["estimated_vram_gb"] == 10.0

        release.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert len(client.loaded_model_snapshots()) == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_llama_cpp_chat_keeps_vram_reservation_on_upstream_error():
    """After an upstream error the VRAM reservation must be kept, not discarded.

    _ensure_ready confirmed the server was healthy (model in VRAM) before the
    POST. Clearing the reservation on a 5xx would cause the dashboard to show
    no model even though VRAM is still occupied.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(500, json={"error": "overloaded"})

    client = LlamaCppClient(
        "http://127.0.0.1:21441",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat({"model": "m", "messages": []}, estimated_vram_gb=8.0)
        # Reservation must survive the error — model is still in VRAM.
        snaps = client.loaded_model_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["model"] == "m"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_chat_keeps_vram_reservation_on_upstream_error():
    """Same reservation-survives-error guarantee for the Ollama client."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        return httpx.Response(500, json={"error": "overloaded"})

    client = OllamaClient(
        "http://127.0.0.1:11434",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat({"model": "m", "messages": []}, estimated_vram_gb=8.0)
        snaps = client.loaded_model_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["model"] == "m"
    finally:
        await client.aclose()
