"""Async facade separating launcher completion from owned group lifetime."""
import asyncio
import json
from pathlib import Path


class PosixProcess:
    def __init__(self, guardian):
        self.guardian = guardian
        self.pid = guardian.pid
        self.returncode = None
        self.completed = asyncio.Event()
        self.stopping = False
        self.closed = False
        self.reader = asyncio.create_task(self._read_status())

    async def _read_status(self):
        try:
            line = await self.guardian.stdout.readline()
            if line:
                self.returncode = json.loads(line)["returncode"]
            else:
                self.returncode = await self.guardian.wait()
        finally:
            self.completed.set()

    async def wait(self):
        await self.completed.wait()
        return self.returncode

    async def terminate_tree(self, timeout):
        if self.closed:
            return True
        if not self.stopping:
            if self.guardian.returncode is not None:
                # Lost anchor: no authority to signal a possibly reused PGID.
                self.closed = not await group_has_live_members(self.pid)
                return self.closed
            self.stopping = True
            self.guardian.stdin.write((json.dumps({"stop": timeout}) + "\n").encode())
            await self.guardian.stdin.drain()
        try:
            await asyncio.wait_for(self.guardian.wait(), timeout + 2)
            await self.reader
            self.closed = not await group_has_live_members(self.pid)
            self.guardian.stdin.close()
            return self.closed
        except (asyncio.TimeoutError, OSError):
            return False


async def group_has_live_members(pgid, *, exclude=()):
    if Path("/proc/self/stat").exists():
        for path in Path("/proc").glob("[0-9]*/stat"):
            if int(path.parent.name) in exclude:
                continue
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except FileNotFoundError:
                continue
            except (OSError, ValueError, IndexError):
                return True
        return False
    # BSD/macOS: inspect state without signalling any PID. Failure is unsafe.
    process = await asyncio.create_subprocess_exec("ps", "-axo", "pid=,pgid=,stat=", stdout=asyncio.subprocess.PIPE)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 2)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return True
    if process.returncode:
        return True
    return any(int(parts[1]) == pgid and int(parts[0]) not in exclude and not parts[2].startswith("Z")
               for line in output.decode().splitlines() if len(parts := line.split()) >= 3)
