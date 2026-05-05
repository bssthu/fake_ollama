"""Helpers for subprocesses owned by local model targets."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Optional, Sequence


def _common_kwargs(cwd: Optional[str], env: Optional[dict]) -> dict:
    # Normalize empty/whitespace cwd to None — on Windows, subprocess
    # treats cwd="" as an invalid path and raises WinError 123.
    if isinstance(cwd, str) and not cwd.strip():
        cwd = None
    kwargs: dict = {
        "cwd": cwd,
        "stdout": asyncio.subprocess.DEVNULL,
        "stderr": asyncio.subprocess.DEVNULL,
    }
    if env is not None:
        kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


async def create_managed_subprocess_shell(
    command: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> asyncio.subprocess.Process:
    # Normalize empty/whitespace cwd to None — on Windows, subprocess
    # treats cwd="" as an invalid path and raises WinError 123.
    if isinstance(cwd, str) and not cwd.strip():
        cwd = None
    kwargs = {
        "cwd": cwd,
        "stdout": asyncio.subprocess.DEVNULL,
        "stderr": asyncio.subprocess.DEVNULL,
    }
    if env is not None:
        kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return await asyncio.create_subprocess_shell(command, **kwargs)


async def create_managed_subprocess_exec(
    argv: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> asyncio.subprocess.Process:
    """Spawn a process directly (no shell wrapper).

    Prefer this over ``create_managed_subprocess_shell`` whenever the caller
    holds the argv list, because the captured ``Process`` is the actual
    target program. With the shell variant on Windows the captured PID is
    the ``cmd.exe`` wrapper, whose lifetime is decoupled from the child;
    once the wrapper exits we can no longer reliably terminate the real
    child via ``taskkill /T /F`` (its parent is gone), so resource cleanup
    silently fails.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    kwargs = _common_kwargs(cwd, env)
    return await asyncio.create_subprocess_exec(*argv, **kwargs)


async def terminate_process_tree(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 10.0,
) -> bool:
    if process.returncode is not None:
        return True
    if os.name == "nt":
        return await _terminate_windows_process_tree(process, timeout=timeout)
    return await _terminate_posix_process_group(process, timeout=timeout)


async def _terminate_windows_process_tree(
    process: asyncio.subprocess.Process,
    *,
    timeout: float,
) -> bool:
    killer = await asyncio.create_subprocess_exec(
        "taskkill",
        "/PID",
        str(process.pid),
        "/T",
        "/F",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(killer.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        killer.kill()
        await killer.wait()

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
    return process.returncode is not None


async def _terminate_posix_process_group(
    process: asyncio.subprocess.Process,
    *,
    timeout: float,
) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        process.terminate()

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
    return process.returncode is not None