"""Isolated subprocess acceptance; opt in separately from default offline tests."""
import asyncio
import os
from pathlib import Path
import sys
import time

import pytest

from fake_ollama.process_utils import create_managed_subprocess_exec, terminate_process_tree


def alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.e2e
@pytest.mark.parametrize("parent_exits", [True, False])
async def test_tree_termination_survives_launcher_exit(tmp_path, parent_exits):
    marker, stop = tmp_path / "child.pid", tmp_path / "stop"
    script = tmp_path / "child.py"
    script.write_text("import os,time,signal\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "end=time.monotonic()+20\n"
        f"while not Path({str(stop)!r}).exists() and time.monotonic()<end: time.sleep(.02)\n")
    code = f"import subprocess,sys,time; subprocess.Popen([sys.executable, {str(script)!r}]); " + ("pass" if parent_exits else "time.sleep(20)")
    process = await create_managed_subprocess_exec([sys.executable, "-c", code])
    pid = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        pid = int(marker.read_text())
        assert alive(pid)
        if parent_exits:
            await asyncio.wait_for(process.wait(), 5)
        assert await terminate_process_tree(process, timeout=.3)
        assert not alive(pid), "child survived despite successful tree termination"
        assert await terminate_process_tree(process, timeout=.3)
    finally:
        stop.touch()
        if process.returncode is None:
            await terminate_process_tree(process, timeout=.3)
        deadline = time.monotonic() + 3
        while pid and alive(pid) and time.monotonic() < deadline:
            await asyncio.sleep(.01)


@pytest.mark.e2e
@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group and TERM/KILL semantics")
async def test_posix_term_ignoring_child(tmp_path):
    await test_tree_termination_survives_launcher_exit(tmp_path, False)


@pytest.mark.e2e
@pytest.mark.parametrize("code", [0, 7])
async def test_normal_exit_retains_verifiable_empty_tree(code):
    process = await create_managed_subprocess_exec([sys.executable, "-c", f"raise SystemExit({code})"])
    assert await process.wait() == code
    assert await terminate_process_tree(process, timeout=1)
