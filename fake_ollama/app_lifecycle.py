"""Application resource construction, monitoring and shutdown."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from fastapi import FastAPI
from .anthropic_client import AnthropicClient
from .comfyui_client import ComfyUIClient
from .config import Settings
from .llama_cpp_client import LlamaCppClient
from .generic_openai_client import GenericOpenAIClient
from .decision_client import DecisionClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .vram import MemoryCoordinator, VramCoordinator
from .dashboard import DashboardState, RequestMetrics, run_runtime_monitor

import logging

logger = logging.getLogger("fake_ollama")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    owned_names: list[str] = []
    owned_target_names: list[str] = []
    owned_llama_cpp_names: list[str] = []
    owned_generic_openai_names: list[str] = []
    owned_decision_names: list[str] = []
    owned_openai_names: list[str] = []
    owned_comfyui_names: list[str] = []
    if not getattr(app.state, "clients", None):
        app.state.clients = {}
    if not getattr(app.state, "ollama_clients", None):
        app.state.ollama_clients = {}
    if not getattr(app.state, "llama_cpp_clients", None):
        app.state.llama_cpp_clients = {}
    if not getattr(app.state, "generic_openai_clients", None):
        app.state.generic_openai_clients = {}
    if not getattr(app.state, "decision_clients", None):
        app.state.decision_clients = {}
    if not getattr(app.state, "openai_clients", None):
        app.state.openai_clients = {}
    if not getattr(app.state, "comfyui_clients", None):
        app.state.comfyui_clients = {}
    if not getattr(app.state, "vram_coordinator", None):
        app.state.vram_coordinator = VramCoordinator()
    if not getattr(app.state, "memory_coordinator", None):
        app.state.memory_coordinator = MemoryCoordinator()
    if not getattr(app.state, "dashboard_state", None):
        app.state.dashboard_state = DashboardState()
    if not getattr(app.state, "request_metrics", None):
        app.state.request_metrics = RequestMetrics()
    for up in app.state.settings.anthropic_upstreams:
        if up.name in app.state.clients:
            continue
        app.state.clients[up.name] = AnthropicClient(
            up.base_url,
            up.auth_token,
            timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
        )
        owned_names.append(up.name)
    for up in app.state.settings.openai_upstreams:
        if up.name in app.state.openai_clients:
            continue
        app.state.openai_clients[up.name] = OpenAIClient(
            up.base_url,
            auth_token=up.auth_token,
            timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
            upstream_name=up.name,
        )
        owned_openai_names.append(up.name)
    for tgt in app.state.settings.ollama_targets:
        if tgt.name in app.state.ollama_clients:
            continue
        app.state.ollama_clients[tgt.name] = OllamaClient(
            tgt.base_url,
            timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            stop_command=tgt.stop_command,
            manage_external_runtime=tgt.manage_external_runtime,
            stop_timeout_seconds=tgt.stop_timeout_seconds,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            target_name=tgt.name,
            vram_coordinator=app.state.vram_coordinator,
            memory_coordinator=app.state.memory_coordinator,
        )
        owned_target_names.append(tgt.name)
    for raw_tgt in app.state.settings.llama_cpp_targets:
        tgt = app.state.settings.effective_llama_cpp_target(raw_tgt)
        if tgt.name in app.state.llama_cpp_clients:
            continue
        app.state.llama_cpp_clients[tgt.name] = LlamaCppClient(
            tgt.base_url,
            auth_token=tgt.auth_token,
            timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            start_argv=tgt.synthesize_start_argv(),
            stop_command=tgt.stop_command,
            manage_external_runtime=tgt.manage_external_runtime,
            stop_timeout_seconds=tgt.stop_timeout_seconds,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            launch_env=tgt.effective_env(),
            target_name=tgt.name,
            vram_coordinator=app.state.vram_coordinator,
            memory_coordinator=app.state.memory_coordinator,
            max_concurrent_requests=tgt.effective_max_concurrent_requests,
            request_read_timeout_seconds=tgt.request_read_timeout_seconds,
        )
        owned_llama_cpp_names.append(tgt.name)
    for tgt in app.state.settings.generic_openai_targets:
        if tgt.name in app.state.generic_openai_clients:
            continue
        app.state.generic_openai_clients[tgt.name] = GenericOpenAIClient(
            tgt.base_url,
            auth_token=tgt.auth_token,
            timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            stop_command=tgt.stop_command,
            manage_external_runtime=tgt.manage_external_runtime,
            stop_timeout_seconds=tgt.stop_timeout_seconds,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            target_name=tgt.name,
            vram_coordinator=app.state.vram_coordinator,
            memory_coordinator=app.state.memory_coordinator,
            max_concurrent_requests=tgt.max_concurrent_requests,
            request_read_timeout_seconds=tgt.request_read_timeout_seconds,
        )
        owned_generic_openai_names.append(tgt.name)
    for tgt in app.state.settings.decision_targets:
        if tgt.name in app.state.decision_clients:
            continue
        app.state.decision_clients[tgt.name] = DecisionClient(
            tgt, timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
            vram_coordinator=app.state.vram_coordinator,
            memory_coordinator=app.state.memory_coordinator,
        )
        owned_decision_names.append(tgt.name)
    for tgt in app.state.settings.comfyui_targets:
        if tgt.name in app.state.comfyui_clients:
            continue
        app.state.comfyui_clients[tgt.name] = ComfyUIClient(
            tgt.base_url,
            auth_token=tgt.auth_token,
            timeout=app.state.settings.timeout_seconds,
            trust_env=app.state.settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            stop_command=tgt.stop_command,
            manage_external_runtime=tgt.manage_external_runtime,
            stop_timeout_seconds=tgt.stop_timeout_seconds,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            target_name=tgt.name,
            runtime_group=tgt.vram_runtime_group,
            gpu_device=tgt.gpu_device,
            workflow_config=tgt.workflow_config(),
            vram_coordinator=app.state.vram_coordinator,
            memory_coordinator=app.state.memory_coordinator,
        )
        owned_comfyui_names.append(tgt.name)
    for pool, names in (("clients", owned_names), ("openai_clients", owned_openai_names),
                        ("ollama_clients", owned_target_names), ("llama_cpp_clients", owned_llama_cpp_names),
                        ("generic_openai_clients", owned_generic_openai_names), ("decision_clients", owned_decision_names), ("comfyui_clients", owned_comfyui_names)):
        app.state.runtime_resources.managed.update(getattr(app.state, pool)[name] for name in names)
    _sync_local_target_idle_monitor(app)
    _sync_runtime_monitor(app)
    try:
        yield
    finally:
        runtime_monitor = getattr(app.state, "runtime_monitor", None)
        if runtime_monitor is not None:
            runtime_monitor.cancel()
            try:
                await runtime_monitor
            except asyncio.CancelledError:
                pass
            app.state.runtime_monitor = None
        idle_monitor = getattr(app.state, "local_target_idle_monitor", None)
        if idle_monitor is not None:
            idle_monitor.cancel()
            try:
                await idle_monitor
            except asyncio.CancelledError:
                pass
            app.state.local_target_idle_monitor = None
        manager = app.state.runtime_resources
        await manager.retire(set(manager.managed))
        for name in ("clients", "openai_clients", "ollama_clients", "llama_cpp_clients", "generic_openai_clients", "decision_clients", "comfyui_clients"):
            pool = getattr(app.state, name, {})
            for key, client in list(pool.items()):
                if client not in manager.managed and getattr(client, "_owns_client", False):
                    pool.pop(key, None)


def request_shutdown(app: FastAPI) -> None:
    """Mark the app as shutting down and disable local target auto-starts.

    Also schedules ``stop_if_owned`` for every owned local target client so
    in-flight requests against a still-loading llama.cpp / ollama child get
    aborted (owned Job Object on Windows). Without this the upstream HTTP
    connection from fake-ollama to the local server stays open until the
    huge model finishes loading, which prevents uvicorn from exiting after
    CTRL+C ("Waiting for connections to close").
    """
    if getattr(app.state, "shutdown_requested", False):
        return
    app.state.shutdown_requested = True
    owned_clients: list[Any] = []
    for client_group in (
        getattr(app.state, "ollama_clients", {}),
        getattr(app.state, "llama_cpp_clients", {}),
        getattr(app.state, "generic_openai_clients", {}),
        getattr(app.state, "decision_clients", {}),
        getattr(app.state, "comfyui_clients", {}),
    ):
        for client in list(client_group.values()):
            begin_shutdown = getattr(client, "begin_shutdown", None)
            if callable(begin_shutdown):
                begin_shutdown()
            if getattr(client, "stop_if_owned", None) is not None:
                owned_clients.append(client)

    if not owned_clients:
        return

    async def _stop_all() -> None:
        for client in owned_clients:
            try:
                await client.stop_if_owned()
            except Exception:
                logger.exception(
                    "failed to stop owned local target during shutdown"
                )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.create_task(_stop_all())
    else:
        # No running loop (e.g. uvicorn already finished serving). Run the
        # cleanup synchronously so the caller still kills the children.
        try:
            asyncio.run(_stop_all())
        except RuntimeError:
            pass


async def _local_target_idle_monitor(app: FastAPI) -> None:
    while True:
        try:
            await asyncio.sleep(5.0)
            client_groups = (
                getattr(app.state, "ollama_clients", {}),
                getattr(app.state, "llama_cpp_clients", {}),
                getattr(app.state, "generic_openai_clients", {}),
                getattr(app.state, "decision_clients", {}),
                getattr(app.state, "comfyui_clients", {}),
            )
            for clients in client_groups:
                for client in list(clients.values()):
                    stop_if_idle = getattr(client, "stop_if_idle", None)
                    if stop_if_idle is not None:
                        await stop_if_idle()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("local target idle lifecycle check failed")


def _sync_local_target_idle_monitor(app: FastAPI) -> None:
    if getattr(app.state, "shutdown_requested", False):
        task = getattr(app.state, "local_target_idle_monitor", None)
        if task is not None and not task.done():
            task.cancel()
        app.state.local_target_idle_monitor = None
        return

    client_groups = (
        getattr(app.state, "ollama_clients", {}),
        getattr(app.state, "llama_cpp_clients", {}),
        getattr(app.state, "generic_openai_clients", {}),
        getattr(app.state, "decision_clients", {}),
        getattr(app.state, "comfyui_clients", {}),
    )
    needs_monitor = any(
        getattr(c, "idle_timeout_seconds", None)
        for clients in client_groups
        for c in clients.values()
    )
    task = getattr(app.state, "local_target_idle_monitor", None)
    task_running = task is not None and not task.done()
    if needs_monitor and not task_running:
        app.state.local_target_idle_monitor = asyncio.create_task(
            _local_target_idle_monitor(app)
        )
    elif not needs_monitor and task_running:
        task.cancel()
        app.state.local_target_idle_monitor = None


def _sync_runtime_monitor(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    needs_monitor = (
        settings.dashboard_listener_enabled or settings.vram_low_free_reclaim_enabled
    )
    task = getattr(app.state, "runtime_monitor", None)
    task_running = task is not None and not task.done()
    if needs_monitor and not task_running:
        app.state.runtime_monitor = asyncio.create_task(run_runtime_monitor(app))
    elif not needs_monitor and task_running:
        task.cancel()
        app.state.runtime_monitor = None
