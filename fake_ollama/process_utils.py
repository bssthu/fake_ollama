"""Helpers for subprocesses owned by local model targets."""

from __future__ import annotations

import asyncio
import os
import json
import sys
import time
import subprocess
from pathlib import Path
from typing import IO, Optional, Sequence, Tuple, Union

from .lifecycle import finish_cleanup

StderrTarget = Union[None, str, os.PathLike, int, IO]


def _resolve_stderr(stderr: StderrTarget) -> Tuple[object, Optional[IO]]:
    """Translate a caller-provided stderr target into a subprocess kwarg.

    Returns ``(kwarg_value, owned_file)``. ``owned_file`` is the file
    object the parent process opened on the caller's behalf; the caller
    must close it once the child has inherited the fd. Path-like targets
    are opened in append mode so each restart appends to (rather than
    truncates) the file. ``None`` keeps the historical DEVNULL behaviour.
    """
    if stderr is None:
        return asyncio.subprocess.DEVNULL, None
    if isinstance(stderr, (str, os.PathLike)):
        path = Path(stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered text mode so partial writes from the child still
        # land on disk while the server is running — that is the whole
        # point of capturing stderr instead of /dev/null.
        fh = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        return fh, fh
    return stderr, None


def _common_kwargs(
    cwd: Optional[str],
    env: Optional[dict],
    stderr_value: object,
) -> dict:
    # Normalize empty/whitespace cwd to None — on Windows, subprocess
    # treats cwd="" as an invalid path and raises WinError 123.
    if isinstance(cwd, str) and not cwd.strip():
        cwd = None
    kwargs: dict = {
        "cwd": cwd,
        "stdout": asyncio.subprocess.DEVNULL,
        "stderr": stderr_value,
    }
    if env is not None:
        kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | 0x4  # CREATE_SUSPENDED
    else:
        kwargs["start_new_session"] = True
    return kwargs


async def _spawn(command, *, shell, cwd=None, env=None, stderr=None):
    stderr_value, owned = _resolve_stderr(stderr)
    kwargs = _common_kwargs(cwd, env, stderr_value)
    async def launch():
        if os.name == "nt":
            from .windows_job import WindowsJob
            job = WindowsJob()
            process = None
            try:
                process = (await asyncio.create_subprocess_shell(command, **kwargs) if shell
                           else await asyncio.create_subprocess_exec(*command, **kwargs))
                job.assign_and_resume(process.pid)
                process._managed_job = job
                return process
            except BaseException:
                # The process is still suspended if job assignment failed.
                # kill() uses its retained process handle, never a name/PID tree.
                if process is not None:
                    process.kill()
                    await process.wait()
                job.close()
                raise
        from .posix_process import PosixProcess
        kwargs["stdin"] = asyncio.subprocess.PIPE
        kwargs["stdout"] = asyncio.subprocess.PIPE
        guardian = await asyncio.create_subprocess_exec(sys.executable, "-u",
            str(Path(__file__).with_name("posix_guardian.py")),
            json.dumps({"command": command, "shell": shell}), **kwargs)
        return PosixProcess(guardian)
    task = asyncio.create_task(launch())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        async def cleanup_spawn():
            process = await task
            if not await terminate_process_tree(process):
                raise OSError("Cancelled process spawn cleanup failed")
        await finish_cleanup(cleanup_spawn())
        raise
    finally:
        if owned is not None:
            owned.close()


async def create_managed_subprocess_shell(command, *, cwd=None, env=None, stderr=None):
    return await _spawn(command, shell=True, cwd=cwd, env=env, stderr=stderr)


async def create_managed_subprocess_exec(argv, *, cwd=None, env=None, stderr=None):
    if not argv:
        raise ValueError("argv must not be empty")
    return await _spawn(list(argv), shell=False, cwd=cwd, env=env, stderr=stderr)


async def terminate_process_tree(process, *, timeout=10.0):
    """Confirm the owned tree is empty, including after launcher completion.

    Unknown process objects have no tree identity and fail closed. Cancellation
    joins the finite cleanup instead of leaving a separate killer unobserved.
    """
    async def terminate():
        if os.name != "nt":
            method = getattr(process, "terminate_tree", None)
            return await method(timeout) if method is not None else False
        job = getattr(process, "_managed_job", None)
        if job is None:
            return False
        handles = []
        try:
            handles = job.process_handles()
            if job.active():
                job.terminate()
            deadline = time.monotonic() + timeout
            while job.active() or any(job.kernel.WaitForSingleObject(handle, 0) != 0 for handle in handles):
                if time.monotonic() >= deadline:
                    return False
                await asyncio.sleep(.01)
            await asyncio.wait_for(process.wait(), max(.1, timeout))
            job.close()
            return True
        except (OSError, asyncio.TimeoutError):
            return False
    return await finish_cleanup(terminate())


async def process_tree_running(process):
    """True/False for a verified owned identity, None if status is uncertain."""
    try:
        if os.name == "nt":
            job = getattr(process, "_managed_job", None)
            return job.running() if job is not None else None
        from .posix_process import PosixProcess, group_has_live_members
        if isinstance(process, PosixProcess):
            return await group_has_live_members(process.pid, exclude={process.pid})
    except OSError:
        pass
    return None


async def run_stop_command(command, *, cwd=None, timeout=30.0):
    """Bound command lifetime and always reclaim its owned process tree."""
    process = await create_managed_subprocess_shell(command, cwd=cwd)
    try:
        try:
            code = await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return code == 0
    finally:
        if not await finish_cleanup(terminate_process_tree(process, timeout=min(10.0, timeout))):
            raise OSError("Stop command process tree cleanup failed")
