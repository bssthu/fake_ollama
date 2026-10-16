"""Shared endpoint quarantine and conservative prompt reconciliation.

Never globally interrupt a shared ComfyUI. A cancelled HTTP request is not proof
that GPU work stopped. Missing acknowledgement is reconciled using a unique
client ID as well as the proposed prompt ID (older servers may ignore the latter).
"""
import asyncio
from dataclasses import dataclass, field
import uuid

import httpx

from .vram import LocalTargetResourceError


@dataclass
class PromptRecord:
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = field(default_factory=lambda: "fake-ollama-" + uuid.uuid4().hex)
    status: str = "submitting"
    uncertain: bool = True
    recovery_task: asyncio.Task | None = None
    poll_task: asyncio.Task | None = None
    tracking: set | None = None


class ComfyRuntime:
    def __init__(self):
        self.pending: dict[str, PromptRecord] = {}

    def check(self):
        if self.pending:
            raise LocalTargetResourceError("ComfyUI runtime awaits prompt reconciliation; new admission is paused",
                                           error_type="runtime_quarantined")


# Persist unresolved runtime protection across client replacements/reloads.
_RUNTIMES: dict[str, ComfyRuntime] = {}


def runtime_for(base_url):
    return _RUNTIMES.setdefault(base_url, ComfyRuntime())


def _matches(item, record):
    return (isinstance(item, (list, tuple)) and len(item) >= 4 and
            (item[1] == record.prompt_id or isinstance(item[3], dict) and item[3].get("client_id") == record.client_id))


def _terminal(entry):
    if not isinstance(entry, dict):
        return False
    status = entry.get("status") or {}
    return bool(status.get("completed") or status.get("status_str") in ("success", "error") or entry.get("outputs"))


async def reconcile(client, record) -> bool:
    async def get(path):
        response = await client._client.get(client._base + path, headers=client._headers(), timeout=3.0)
        response.raise_for_status()
        return response.json()
    history = await get("/history/" + record.prompt_id)
    if _terminal(history.get(record.prompt_id)):
        return True
    queue = await get("/queue")
    # A non-ComfyUI/unsupported response cannot establish safe completion.
    if not all(isinstance(queue.get(k), list) for k in ("queue_running", "queue_pending")):
        return False
    running = [item for item in queue["queue_running"] if _matches(item, record)]
    pending = [item for item in queue["queue_pending"] if _matches(item, record)]
    if running:
        record.prompt_id = str(running[0][1])
        record.status = "running"
        return False
    if pending:
        record.prompt_id = str(pending[0][1])
        record.status = "deleting"
        response = await client._client.post(client._base + "/queue", headers=client._headers(),
                                              json={"delete": [record.prompt_id]}, timeout=3.0)
        response.raise_for_status()
        checked = await get("/queue")
        if not all(isinstance(checked.get(k), list) for k in ("queue_running", "queue_pending")):
            return False
        # Deletion can race a worker taking the item: check both queues again.
        return not any(_matches(item, record) for k in ("queue_running", "queue_pending") for item in checked[k])
    if record.uncertain:
        # A server that ignored client-generated prompt_id can have already
        # completed. Match unique request client_id in retained history.
        all_history = await get("/history?max_items=1000")
        for identifier, entry in all_history.items():
            if isinstance(entry, dict) and _matches(entry.get("prompt"), record) and _terminal(entry):
                record.prompt_id = str(identifier)
                return True
    # No queue/history evidence is NOT proof that a lost submission never ran.
    return False


async def recover(client, key, record):
    async def poll():
        while True:
            try:
                if await reconcile(client, record):
                    return True
            except (httpx.HTTPError, ValueError, TypeError):
                record.status = "unreachable"
            await asyncio.sleep(client._recovery_poll_seconds)
    try:
        if record.poll_task is not None and not record.poll_task.done():
            done, _ = await asyncio.wait({record.poll_task}, timeout=client._recovery_timeout_seconds)
            if not done:
                record.status = "quarantined"
                return False
        record.poll_task = asyncio.create_task(poll())
        record.poll_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        done, _ = await asyncio.wait({record.poll_task}, timeout=client._recovery_timeout_seconds)
        if not done:
            record.poll_task.cancel()
            record.status = "quarantined"
            return False
        confirmed = record.poll_task.result()
    except (Exception, asyncio.CancelledError):
        record.status = "quarantined"
        return False
    if confirmed:
        record.status = "finished"
        client._active_prompt_ids.discard(key)
        client._active_prompt_ids.discard(record.prompt_id)
        if record.tracking is not None:
            record.tracking.discard(key)
            record.tracking.discard(record.prompt_id)
        client._runtime.pending.pop(key, None)
        if client._vram_coordinator is not None and not client._runtime.pending:
            await client._vram_coordinator.recover_runtime(client.vram_runtime_group)
    return confirmed
