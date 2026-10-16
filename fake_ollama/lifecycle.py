"""Small shared lifecycle contracts for local backend cleanup."""
import asyncio

from .vram import LocalTargetResourceError


class BackendShuttingDownError(LocalTargetResourceError):
    def __init__(self, target_id):
        super().__init__(f"{target_id} is shutting down", error_type="backend_shutting_down")


async def finish_cleanup(awaitable):
    """Join finite owned cleanup even if the caller is cancelled repeatedly."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
