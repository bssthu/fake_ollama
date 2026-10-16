"""Shared ownership and closure contracts for heterogeneous local backends."""
import asyncio
import weakref
from urllib.parse import urlsplit

from .lifecycle import BackendShuttingDownError, finish_cleanup
from .vram import LocalTargetResourceError

_OWNERS = weakref.WeakValueDictionary()


class RuntimeOwner:
    def __init__(self):
        self.members = weakref.WeakSet()
        self.lock = asyncio.Lock()
        self.process = None
        self.command = None
        self.cwd = None
        self.authorized = False
        self.stopping = False
        self.shutdown_result = None

    def capture(self, client):
        if client._started_by_us and client._process is not None:
            self.process = client._process
            self.command, self.cwd = client._stop_command, client._cwd
        if client._manage_external_runtime and client._used_since_last_stop:
            self.authorized = True
            if self.process is None:
                self.command, self.cwd = client._stop_command, client._cwd

    def can_manage(self):
        for member in self.members:
            self.capture(member)
        return self.process is not None or self.authorized

    def busy(self):
        return any(member._active or member._request_refs or getattr(member, "_queued", 0)
                   or getattr(getattr(member, "_runtime", None), "pending", None)
                   for member in self.members)

    def activate(self, client):
        if not self.members and self.process is None:
            self.authorized, self.command, self.shutdown_result = False, None, None
        self.members.add(client)
        client._ownership_active = True


def register(client, *, group=None, active=True):
    url = urlsplit(client._base)
    endpoint = (url.scheme, (url.hostname or "").lower(), url.port or (443 if url.scheme == "https" else 80))
    keys = [("endpoint", endpoint)]
    if group:
        keys.append(("group", group))
    existing = {_OWNERS[key] for key in keys if key in _OWNERS}
    if len(existing) > 1:
        raise ValueError("Conflicting runtime ownership groups; drain existing runtimes before merging")
    owner = next(iter(existing), None) or RuntimeOwner()
    for key in keys:
        _OWNERS[key] = owner
    client._ownership = owner
    client._ownership_active = False
    client._closed = False
    if active:
        owner.activate(client)
    return owner


def begin_request(client):
    if client._closed or not client._ownership_active or client._ownership.stopping:
        raise BackendShuttingDownError(client.target_id)
    client._used_since_last_stop = True


async def _stop_locked(client, terminate, run_command):
    owner = client._ownership
    if not owner.can_manage():
        return False
    shutting_down = all(getattr(member, "_shutdown_requested", False) for member in owner.members)
    if owner.busy() and not shutting_down:
        return False
    if any(getattr(getattr(member, "_runtime", None), "pending", None) for member in owner.members):
        return False
    if shutting_down and owner.shutdown_result is not None:
        return owner.shutdown_result
    owner.stopping = True
    try:
        result = False
        if owner.command:
            result = await run_command(owner.command, cwd=owner.cwd, timeout=client._stop_timeout)
            # An owned process tree must also be verified, even when the stop
            # command returns zero. Detached descendants remain part of its job.
            if result and owner.process is not None:
                result = await terminate(owner.process, timeout=min(10.0, client._stop_timeout))
        elif owner.process is not None:
            result = await terminate(owner.process, timeout=min(10.0, client._stop_timeout))
        if result:
            owner.process, owner.command, owner.authorized = None, None, False
            for member in list(owner.members):
                member._started_by_us = False
                member._process = None
                member._used_since_last_stop = False
                if hasattr(member, "_invalidate_runtime_resources"):
                    member._invalidate_runtime_resources(full=True)
                else:
                    member._clear_all_vram_state()
        if shutting_down:
            owner.shutdown_result = result
        return result
    finally:
        owner.stopping = False


async def stop_owned(client, *, terminate, run_command):
    async with client._ownership.lock:
        return await _stop_locked(client, terminate, run_command)


async def close_client(client, *, terminate, run_command):
    async def close():
        owner = client._ownership
        async with owner.lock:
            if client._closed:
                return
            owner.capture(client)
            # Registered siblings retain collective ownership when this client
            # retires. Prepared reload replacements register only after drain.
            siblings = set(owner.members) - {client}
            owner.can_manage()
            if not siblings and (owner.process is not None or (owner.authorized and owner.command)):
                if not await _stop_locked(client, terminate, run_command):
                    raise LocalTargetResourceError("Backend stop cleanup failed; resources retained", error_type="runtime_quarantined")
            if client._owns_client:
                await client._client.aclose()
            for field in ("_vram_coordinator", "_memory_coordinator"):
                coordinator = getattr(client, field, None)
                if coordinator is not None:
                    coordinator.unregister(client)
            owner.members.discard(client)
            if not owner.members and owner.process is None:
                owner.authorized, owner.command, owner.shutdown_result = False, None, None
            client._started_by_us = False
            client._process = None
            client._closed = True
            client._ownership_active = False
    await finish_cleanup(close())
