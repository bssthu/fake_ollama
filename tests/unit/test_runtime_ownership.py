import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from fake_ollama.ollama_client import OllamaClient
from fake_ollama.llama_cpp_client import LlamaCppClient
from fake_ollama.comfyui_client import ComfyUIClient


@pytest.mark.parametrize("kind", [OllamaClient, LlamaCppClient, ComfyUIClient])
async def test_configured_stop_command_does_not_own_external_runtime(kind, monkeypatch):
    commands = []
    async def spawn(command, **kwargs):
        commands.append(command)
        return type("Process", (), {"returncode": 0, "wait": AsyncMock(return_value=0)})()
    monkeypatch.setattr(asyncio, "create_subprocess_shell", spawn)
    client = kind("http://external.test", stop_command="synthetic-stop")
    client._used_since_last_stop = True
    await client.aclose()
    assert commands == []


@pytest.mark.parametrize("kind", [OllamaClient, LlamaCppClient, ComfyUIClient])
async def test_closing_one_shared_client_preserves_owned_runtime(kind, monkeypatch):
    stopped = []
    async def terminate(process, **kwargs):
        stopped.append(process)
        return True
    monkeypatch.setattr(kind.__module__ + ".terminate_process_tree", terminate)
    first = kind("http://shared.test", target_name="one")
    second = kind("http://shared.test", target_name="two")
    process = type("Process", (), {"returncode": None})()
    first._process, first._started_by_us = process, True
    second._active = 1
    await first.aclose()
    assert stopped == []
    assert not second._client.is_closed
    second._active = 0
    await second.aclose()
    assert stopped == [process]


async def test_close_failure_keeps_resources_available_for_recovery(monkeypatch):
    client = LlamaCppClient("http://failed-stop.test")
    client._started_by_us = True
    client._process = type("Process", (), {"returncode": None})()
    monkeypatch.setattr("fake_ollama.llama_cpp_client.terminate_process_tree", AsyncMock(return_value=False))
    try:
        with pytest.raises(Exception, match="cleanup|stop"):
            await client.aclose()
        assert not client._client.is_closed
        assert client._started_by_us
    finally:
        monkeypatch.setattr("fake_ollama.llama_cpp_client.terminate_process_tree", AsyncMock(return_value=True))
        await client.aclose()


@pytest.mark.parametrize("kind", [OllamaClient, LlamaCppClient, ComfyUIClient])
async def test_external_management_requires_explicit_opt_in_and_use(kind, monkeypatch):
    calls = []
    async def stop(command, **kwargs):
        calls.append(command)
        return True
    monkeypatch.setattr(kind.__module__ + ".run_stop_command", stop)
    client = kind("http://managed.test", manage_external_runtime=True, stop_command="synthetic-stop")
    assert not await client.stop_if_owned()
    client._begin_request_lifecycle()
    client._end_request_lifecycle()
    assert await client.stop_if_owned()
    assert not await client.stop_if_owned()
    await client.aclose()
    assert calls == ["synthetic-stop"]


async def test_runtime_group_protects_sibling_on_another_url(monkeypatch):
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("fake_ollama.comfyui_client.terminate_process_tree", stop)
    first = ComfyUIClient("http://group-one.test", runtime_group="shared", target_name="one")
    second = ComfyUIClient("http://group-two.test", runtime_group="shared", target_name="two")
    first._process = object()
    first._started_by_us = True
    second._active = 1
    assert not await first.stop_if_owned()
    await first.aclose()
    assert not stop.called
    second._active = 0
    await second.aclose()
    assert stop.await_count == 1


@pytest.mark.parametrize("cancel", [False, True])
async def test_stop_command_timeout_or_cancel_joins_cleanup(monkeypatch, cancel):
    from fake_ollama import process_utils
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Process:
        async def wait(self):
            entered.set()
            await asyncio.Event().wait()
    process = Process()
    async def terminate(value, **kwargs):
        assert value is process
        cleaning.set()
        await release.wait()
        return True
    monkeypatch.setattr(process_utils, "create_managed_subprocess_shell", AsyncMock(return_value=process))
    monkeypatch.setattr(process_utils, "terminate_process_tree", terminate)
    task = asyncio.create_task(process_utils.run_stop_command("test-only", timeout=60 if cancel else .01))
    await entered.wait()
    if cancel:
        task.cancel()
    await cleaning.wait()
    if cancel:
        task.cancel()
    release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await task is False


async def test_unknown_process_identity_never_means_tree_is_gone():
    from fake_ollama.process_utils import terminate_process_tree
    process = type("Unknown", (), {"returncode": 0, "pid": 1})()
    assert not await terminate_process_tree(process)
