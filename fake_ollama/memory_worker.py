"""Bounded, authenticated, memory-only IPC for gateway-owned workers.

The ephemeral listener is loopback-only and closes after the child connects.
JSON + raw bytes (never pickle); existing process-tree ownership is preserved.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import secrets
import socket
import struct

from .privacy import worker_environment

ROOT = Path(__file__).resolve().parent.parent
MAX_JSON = 48 * 1024**2
MAX_BYTES = 4 * 1024**2


def packet(message, data=b''):
    encoded = json.dumps(message, ensure_ascii=False, allow_nan=False).encode('utf-8')
    lengths = (len(encoded), len(data))
    validate_lengths(*lengths)
    return struct.pack('!II', *lengths) + encoded + data


def validate_lengths(json_length, data_length):
    if not 0 < json_length <= MAX_JSON or not 0 <= data_length <= MAX_BYTES:
        raise ValueError('Worker message exceeds memory limit')


class Channel:
    """Synchronous child endpoint. No payloads in command-line arguments."""
    def __init__(self, sock):
        self.socket = sock

    @classmethod
    def connect(cls):
        from .privacy import install_worker_write_guard
        install_worker_write_guard()
        channel = cls(socket.create_connection(('127.0.0.1', int(os.environ.pop('FAKE_WORKER_PORT'))), timeout=30))
        # A listening speech session may remain silent for hours. Request
        # deadlines and client liveness belong to the owner; an idle worker
        # must survive until its owner closes the channel or its Windows Job.
        channel.socket.settimeout(None)
        channel.send({'secret': os.environ.pop('FAKE_WORKER_SECRET')})
        return channel

    def exact(self, size):
        result = bytearray()
        while len(result) < size:
            block = self.socket.recv(size - len(result))
            if not block:
                raise EOFError('Worker owner disconnected')
            result.extend(block)
        return bytes(result)

    def receive(self):
        a, b = struct.unpack('!II', self.exact(8))
        validate_lengths(a, b)
        return json.loads(self.exact(a)), self.exact(b)

    def send(self, message, data=b''):
        self.socket.sendall(packet(message, data))

    def event(self, message, data=b''):
        self.send({'event': message}, data)
        return self.receive()

    def close(self):
        self.socket.close()


class MemoryWorker:
    def __init__(self, python, module):
        self.python, self.module = python, module
        self.process = self.reader = self.writer = None
        self.server = None
        self.lock = asyncio.Lock()

    async def start(self):
        from .process_utils import create_managed_subprocess_exec
        if self.process is not None:
            return
        secret = secrets.token_urlsafe(32)
        ready = asyncio.get_running_loop().create_future()
        connections = set()

        async def accept(reader, writer):
            if len(connections) >= 4:
                writer.close()
                return
            task = asyncio.current_task()
            connections.add(task)
            accepted = False
            try:
                length, binary = struct.unpack('!II', await asyncio.wait_for(reader.readexactly(8), 5))
                if not 0 < length <= 256 or binary:
                    return
                hello = json.loads(await asyncio.wait_for(reader.readexactly(length), 5))
                if secrets.compare_digest(str(hello.get('secret', '')), secret) and not ready.done():
                    accepted = True
                    ready.set_result((reader, writer))
            except (TypeError, ValueError, OSError, asyncio.IncompleteReadError, asyncio.TimeoutError):
                pass
            finally:
                if not accepted:
                    writer.close()
                connections.discard(task)

        server = await asyncio.start_server(accept, '127.0.0.1', 0, limit=64 * 1024)
        self.server = server
        try:
            env = worker_environment()
            env.update(FAKE_WORKER_PORT=str(server.sockets[0].getsockname()[1]), FAKE_WORKER_SECRET=secret)
            self.process = await create_managed_subprocess_exec(
                [self.python, '-B', '-m', self.module, '--memory'], cwd=str(ROOT), env=env)
            exited = asyncio.create_task(self.process.wait())
            try:
                done, _ = await asyncio.wait({ready, exited}, timeout=30, return_when=asyncio.FIRST_COMPLETED)
                if ready not in done:
                    if exited in done:
                        raise RuntimeError(f'隔离模型进程启动失败（退出码 {exited.result()}），请检查运行环境依赖')
                    raise TimeoutError('隔离模型进程连接超时')
                self.reader, self.writer = ready.result()
            finally:
                exited.cancel()
                await asyncio.gather(exited, return_exceptions=True)
        except BaseException:
            await self.stop()
            raise
        finally:
            server.close()
            # Python 3.13 wait_closed also waits for accepted clients. The
            # authenticated connection must stay alive until stop(), not here.
            for task in list(connections):
                task.cancel()
            if connections:
                await asyncio.gather(*connections, return_exceptions=True)

    async def receive(self):
        a, b = struct.unpack('!II', await self.reader.readexactly(8))
        validate_lengths(a, b)
        return json.loads(await self.reader.readexactly(a)), await self.reader.readexactly(b)

    async def call(self, message, data=b'', *, on_event=None, timeout=180):
        async with self.lock:
            await self.start()
            return await asyncio.wait_for(self.exchange(message, data, on_event), timeout)

    async def exchange(self, message, data, on_event):
        self.writer.write(packet(message, data))
        await self.writer.drain()
        while True:
            response, binary = await self.receive()
            if 'event' in response:
                if on_event is None:
                    raise RuntimeError('Unexpected worker event')
                reply = await on_event(response['event'], binary)
                meta, content = reply if isinstance(reply, tuple) else (reply or {}, b'')
                self.writer.write(packet(meta, content))
                await self.writer.drain()
                continue
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response.get('result', response)

    async def stop(self):
        from .process_utils import terminate_process_tree
        if self.writer is not None:
            self.writer.close()
            try:
                await asyncio.wait_for(self.writer.wait_closed(), 2)
            except (OSError, asyncio.TimeoutError):
                pass
            self.writer = None
        if self.process is not None:
            if not await terminate_process_tree(self.process):
                raise RuntimeError('Worker process tree cleanup failed')
            self.process = None
        self.reader = None
        if self.server is not None:
            self.server.close()
            await asyncio.wait_for(self.server.wait_closed(), 5)
            self.server = None
