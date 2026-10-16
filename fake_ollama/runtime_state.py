"""Request snapshots and reference-counted retirement of runtime resources."""
import asyncio
from contextvars import ContextVar
import logging
import time

from fastapi import HTTPException
from starlette.datastructures import State

POOL_NAMES = ("clients", "openai_clients", "ollama_clients", "llama_cpp_clients", "generic_openai_clients", "comfyui_clients")
SNAPSHOT_FIELDS = ("settings", "desired_settings", *POOL_NAMES)
REQUEST_SNAPSHOT = ContextVar("runtime_snapshot", default=None)
logger = logging.getLogger("fake_ollama")


class RuntimeState(State):
    """Keep established app.state accessors pinned for the whole ASGI response."""
    def __getattr__(self, name):
        snapshot = REQUEST_SNAPSHOT.get()
        if snapshot is not None and snapshot[0] is self and name in snapshot[1]:
            return snapshot[1][name]
        return super().__getattr__(name)


class RuntimeResources:
    def __init__(self, app):
        self.app = app
        self.lock = asyncio.Lock()
        self.references = {}
        self.retired = {}
        self.blocked = {}
        self.managed = set()
        self.version = 0
        self.drain_seconds = 30.0

    def check(self, client):
        dependencies = self.blocked.get(client, set())
        if dependencies:
            raise HTTPException(503, "Previous backend generation is draining or awaits cleanup")
        return client

    def attach(self, client):
        if hasattr(client, "_vram_coordinator"):
            for field in ("vram_coordinator", "memory_coordinator"):
                coordinator = self.app.state._state.get(field)
                setattr(client, "_" + field, coordinator)
                if coordinator is not None:
                    coordinator.register(client)

    async def retire(self, clients):
        for client in clients:
            self.retired.setdefault(client, {"since": time.monotonic(), "status": "draining", "task": None})
        await self.close_ready()

    async def close_ready(self):
        ready = []
        for client, state in list(self.retired.items()):
            if self.references.get(client, 0):
                if time.monotonic() - state["since"] >= self.drain_seconds:
                    state["status"] = "waiting_for_requests"
                continue
            if state["task"] is None:
                state["status"] = "closing"
                state["task"] = asyncio.create_task(client.aclose())
                state["task"].add_done_callback(lambda task: None if task.cancelled() else task.exception())
            ready.append((client, state))
        if ready:
            await asyncio.wait({state["task"] for _, state in ready}, timeout=self.drain_seconds)
        for client, state in ready:
            task = state["task"]
            if not task.done():
                state["status"] = "cleanup_pending"
                continue
            try:
                task.result()
            except (Exception, asyncio.CancelledError):
                state["status"] = "cleanup_failed"
                logger.warning("Retired backend cleanup failed; replacement remains gated")
                continue
            self.retired.pop(client, None)
            self.managed.discard(client)
            for replacement, dependencies in list(self.blocked.items()):
                dependencies.discard(client)
                if not dependencies:
                    del self.blocked[replacement]
                    if any(replacement in self.app.state._state.get(name, {}).values() for name in POOL_NAMES):
                        self.attach(replacement)

    async def retry_cleanup(self):
        for state in self.retired.values():
            if state["status"] == "cleanup_failed":
                state["task"] = None
        await self.close_ready()

    def snapshots(self):
        return [{"target": getattr(client, "target_id", "remote"), "status": (
                    "waiting_for_requests" if self.references.get(client, 0) and time.monotonic() - state["since"] >= self.drain_seconds else state["status"]),
                 "references": self.references.get(client, 0)} for client, state in self.retired.items()]


class RuntimeSnapshotMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        state = scope["app"].state
        manager = state.runtime_resources
        # Copy the mappings; all fields are published without an intervening await.
        fields = {key: (dict(state._state[key]) if key in POOL_NAMES else state._state[key])
                  for key in SNAPSHOT_FIELDS if key in state._state}
        clients = {c for key in POOL_NAMES for c in fields.get(key, {}).values()}
        for client in clients:
            manager.references[client] = manager.references.get(client, 0) + 1
        token = REQUEST_SNAPSHOT.set((state, fields))
        try:
            await self.app(scope, receive, send)
        finally:
            REQUEST_SNAPSHOT.reset(token)
            for client in clients:
                count = manager.references[client] - 1
                if count:
                    manager.references[client] = count
                else:
                    manager.references.pop(client, None)
            # No await before ownership is transferred to a tracked retirement.
            await manager.close_ready()
