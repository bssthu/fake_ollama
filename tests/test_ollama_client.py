"""Tests for the lifecycle-aware Ollama target client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from fake_ollama.ollama_client import OllamaClient
from fake_ollama.llama_cpp_client import LlamaCppClient


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
        "fake_ollama.ollama_client.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
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
        "fake_ollama.ollama_client.asyncio.create_subprocess_shell",
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
        "fake_ollama.ollama_client.asyncio.create_subprocess_shell",
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
        "fake_ollama.llama_cpp_client.asyncio.create_subprocess_shell",
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