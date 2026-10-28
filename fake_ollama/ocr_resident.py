"""Gateway-owned OCR process, using a private single-request file mailbox."""
import asyncio
import json
from pathlib import Path
import tempfile
import time

from fastapi import HTTPException
from .process_utils import create_managed_subprocess_exec, terminate_process_tree
from .lifecycle import finish_cleanup


class ResidentOCR:
    def __init__(self):
        self.process = None
        self.directory = None
        self.config_key = None
        self.loaded = []
        self.shutting_down = False
        self.stop_lock = asyncio.Lock()

    def begin_shutdown(self):
        self.shutting_down = True

    def status(self):
        alive = self.process is not None and self.process.returncode is None
        return {'worker_running': alive, 'loaded_models': self.loaded if alive else []}

    async def stop_if_owned(self):
        async with self.stop_lock:
            await self._stop()

    async def _stop(self):
        if self.process is not None:
            if not await terminate_process_tree(self.process):
                raise RuntimeError('OCR resident process cleanup failed')
            self.process = None
        self.loaded = []
        self.config_key = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    async def run(self, payload, config, request):
        started = time.perf_counter()
        key = json.dumps(config, sort_keys=True)
        try:
            if self.shutting_down:
                raise HTTPException(503, 'OCR is shutting down')
            if self.process is not None and (self.process.returncode is not None or self.config_key != key):
                await self.stop_if_owned()
            if self.process is None:
                self.directory = tempfile.TemporaryDirectory(prefix='fake-ocr-resident-')
                work = Path(self.directory.name)
                config_path = work / 'config.json'
                config_path.write_text(key, encoding='utf-8')
                self.process = await create_managed_subprocess_exec([
                    config['python'], '-m', 'fake_ollama.ocr_worker', '--config', str(config_path),
                    '--serve', str(work),
                ], cwd=str(Path(__file__).resolve().parent.parent), stderr='logs/ocr-worker.err.log')
                self.config_key = key
            if self.shutting_down:
                raise HTTPException(503, 'OCR is shutting down')
            work = Path(self.directory.name)
            source, output = work / 'request.json', work / 'result.json'
            output.unlink(missing_ok=True)
            partial = work / 'request.partial'
            partial.write_text(payload.model_dump_json(), encoding='utf-8')
            partial.replace(source)
            deadline = time.monotonic() + 900
            while not output.is_file():
                if self.process is None or self.process.returncode is not None:
                    raise HTTPException(502, 'OCR resident worker exited')
                if time.monotonic() >= deadline:
                    raise HTTPException(504, 'OCR timed out')
                if await request.is_disconnected():
                    raise HTTPException(499, 'OCR request cancelled')
                await asyncio.sleep(0.02)
            if await request.is_disconnected():
                raise HTTPException(499, 'OCR request cancelled')
            result = json.loads(output.read_text(encoding='utf-8'))
            output.unlink()
            if 'fatal_error' in result:
                raise HTTPException(422 if result.get('error_code') == 'invalid_image' else 502, result['fatal_error'])
            self.loaded = result.get('loaded_models', [])
            result['request_seconds'] = round(time.perf_counter() - started, 4)
            return result
        except BaseException:
            await finish_cleanup(self.stop_if_owned())
            raise
