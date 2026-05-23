"""Helpers for subprocesses owned by local model targets."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import IO, Optional, Sequence, Tuple, Union

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
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


async def create_managed_subprocess_shell(
    command: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    stderr: StderrTarget = None,
) -> asyncio.subprocess.Process:
    stderr_value, owned = _resolve_stderr(stderr)
    kwargs = _common_kwargs(cwd, env, stderr_value)
    try:
        return await asyncio.create_subprocess_shell(command, **kwargs)
    finally:
        # The child has inherited the underlying fd; the parent no longer
        # needs its own handle. Closing here avoids leaking one file
        # descriptor per server restart.
        if owned is not None:
            owned.close()


async def create_managed_subprocess_exec(
    argv: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    stderr: StderrTarget = None,
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
    stderr_value, owned = _resolve_stderr(stderr)
    kwargs = _common_kwargs(cwd, env, stderr_value)
    try:
        return await asyncio.create_subprocess_exec(*argv, **kwargs)
    finally:
        if owned is not None:
            owned.close()


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