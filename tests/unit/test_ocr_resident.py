import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fake_ollama.ocr_pipeline import OCRRequest
from fake_ollama import ocr_resident


class Request:
    def __init__(self, disconnected=False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


@pytest.fixture
def worker(monkeypatch):
    spawned, killed = [], []

    async def spawn(argv, **kwargs):
        directory = Path(argv[argv.index('--serve') + 1])
        process = SimpleNamespace(returncode=None, directory=directory)
        spawned.append(process)

        async def serve():
            while True:
                source = directory / 'request.json'
                if source.exists():
                    payload = json.loads(source.read_text())
                    source.unlink()
                    if payload['image'] != 'wait':
                        result = {'results':[], 'loaded_models':['fixture'], 'keep_loaded':True}
                        if payload['image'] == 'error':
                            result = {'fatal_error':'fixture failure'}
                        (directory / 'result.json').write_text(json.dumps(result))
                await asyncio.sleep(.001)
        process.task = asyncio.create_task(serve())
        return process

    async def terminate(process):
        killed.append(process)
        process.task.cancel()
        try:
            await process.task
        except asyncio.CancelledError:
            pass
        process.returncode = 0
        return True

    monkeypatch.setattr(ocr_resident, 'create_managed_subprocess_exec', spawn)
    monkeypatch.setattr(ocr_resident, 'terminate_process_tree', terminate)
    return spawned, killed


@pytest.mark.asyncio
async def test_retains_worker_reuses_then_releases_and_reloads(worker):
    spawned, killed = worker
    manager = ocr_resident.ResidentOCR()
    req = OCRRequest(image='x', keep_loaded=True)
    config = {'python':'fixture'}
    for _ in range(2):
        result = await manager.run(req, config, Request())
        assert result['keep_loaded'] and result['request_seconds'] >= 0
    assert len(spawned) == 1 and not killed
    assert manager.status()['loaded_models'] == ['fixture']
    assert not (spawned[0].directory / 'request.json').exists()
    assert not (spawned[0].directory / 'result.json').exists()
    await manager.stop_if_owned()
    assert not spawned[0].directory.exists()
    assert manager.status() == {'worker_running':False, 'loaded_models':[]}
    await manager.run(req, config, Request())
    assert len(spawned) == 2
    # Config changes must discard cached runtime choices.
    await manager.run(req, {**config, 'version':2}, Request())
    assert len(spawned) == 3 and len(killed) == 2
    manager.begin_shutdown()
    await manager.stop_if_owned()
    with pytest.raises(HTTPException, match='shutting down'):
        await manager.run(req, config, Request())
    assert len(spawned) == len(killed)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['disconnect', 'cancel', 'error'])
async def test_cancel_disconnect_and_failure_release_entire_resident(worker, mode):
    spawned, killed = worker
    manager = ocr_resident.ResidentOCR()
    req = OCRRequest(image='error' if mode == 'error' else 'wait', keep_loaded=True)
    task = asyncio.create_task(manager.run(req, {'python':'fixture'}, Request(mode=='disconnect')))
    if mode == 'cancel':
        while not spawned:
            await asyncio.sleep(.001)
        task.cancel()
    with pytest.raises(asyncio.CancelledError if mode == 'cancel' else HTTPException):
        await task
    assert len(spawned) == len(killed) == 1
    assert manager.process is None and not spawned[0].directory.exists()
