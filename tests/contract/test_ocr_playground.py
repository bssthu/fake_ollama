from fastapi.testclient import TestClient
from fake_ollama.server import create_app
from tests.contract.test_playground import _settings
import json
from pathlib import Path
import pytest
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor


def test_ocr_page_is_static_but_api_requires_auth():
    with TestClient(create_app(_settings()), base_url='http://127.0.0.1:21431', client=('127.0.0.1', 50000)) as client:
        page = client.get('/playground/ocr')
        assert page.status_code == 200 and 'ocrSection' in page.text
        assert page.url.path == '/playground/'
        assert "frame-ancestors 'none'" in page.headers['content-security-policy']
        assert client.get('/playground/api/ocr/models').status_code in (401, 403)
        assert client.post('/playground/api/ocr/run', json={'image':'x'}).status_code in (401, 403)
        assert client.post('/playground/api/ocr/release').status_code in (401, 403)
        assert client.post('/playground/api/ocr/release', headers={'Authorization':'Bearer key-a'}).json()['released']
        assert client.get('/playground/api/ocr/models', headers={'Authorization':'Bearer key-a'}).status_code == 503


@pytest.mark.parametrize('worker_result,status', [({'results': [], 'decision':'review'}, 200), ({'fatal_error':'model load failed'}, 502), ({'fatal_error':'invalid image','error_code':'invalid_image'},422)])
def test_worker_boundary_and_cleanup(tmp_path, monkeypatch, worker_result, status):
    from fake_ollama import ocr_playground
    config = tmp_path / 'ocr.json'
    config.write_text(json.dumps({'python':'isolated-python', 'engines':{'minicpm-ocr':{'required_files':[]}},
                                 'verifiers':{'jev':{'kind':'minicpm_jev','model':'missing','mmproj':'missing'}}}))
    settings = _settings().model_copy(update={'playground_ocr_config':str(config)})
    launches, cleaned = [], []
    async def spawn(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index('--output')+1])
        assert 'Authorization' not in Path(argv[argv.index('--input')+1]).read_text()
        output.write_text(json.dumps(worker_result))
        return type('Process', (), {'returncode':0})()
    async def cleanup(process):
        cleaned.append(process)
        return True
    monkeypatch.setattr(ocr_playground, 'create_managed_subprocess_exec', spawn)
    monkeypatch.setattr(ocr_playground, 'terminate_process_tree', cleanup)
    with TestClient(create_app(settings), base_url='http://127.0.0.1:21431', client=('127.0.0.1',50000)) as client:
        headers={'Authorization':'Bearer key-a'}
        assert client.get('/playground/api/ocr/models',headers=headers).status_code == 200
        discovery = client.get('/playground/api/models', headers=headers).json()
        assert discovery['authentication'] == 'token'
        models = {model['id']: model for model in discovery['models']}
        assert models['ocr:minicpm-ocr']['operations'][0]['configured']
        assert models['ocr:verify']['operations'][0]['configured'] is False
        assert models['ocr:compare']['operations'][0]['ocr_mode'] == 'compare'
        assert not launches  # discovery must never load a model
        assert client.post('/playground/api/ocr/run',headers=headers,json={'image':'x','engines':['unknown']}).status_code == 422
        response=client.post('/playground/api/ocr/run',headers=headers,json={'image':'x'})
        assert response.status_code == status
    assert len(launches) == len(cleaned) == 1
    assert not Path(launches[0][launches[0].index('--input')+1]).exists()


def test_verifier_selection_and_validation_before_spawn(tmp_path, monkeypatch):
    from fake_ollama import ocr_playground
    weight = tmp_path / 'weights.gguf'
    weight.write_bytes(b'fixture')
    spec = {'kind':'minicpm_jev', 'model':str(weight), 'mmproj':str(weight), 'renderer_source':str(tmp_path)}
    config = tmp_path / 'ocr.json'
    config.write_text(json.dumps({'python':'isolated-python', 'engines':{},
                                 'verifiers':{'first':{**spec,'label':'First Jev'},
                                              'second':{**spec,'label':'Second Jev'},
                                              'missing':{**spec,'model':'missing.gguf'}}}))
    settings = _settings().model_copy(update={'playground_ocr_config':str(config)})
    received = []
    async def spawn(argv, **kwargs):
        payload = json.loads(Path(argv[argv.index('--input')+1]).read_text())
        received.append(payload)
        Path(argv[argv.index('--output')+1]).write_text(json.dumps({'verifier':payload['verifier'], 'results':[]}))
        return type('Process', (), {'returncode':0})()
    async def cleanup(process):
        return True
    monkeypatch.setattr(ocr_playground, 'create_managed_subprocess_exec', spawn)
    monkeypatch.setattr(ocr_playground, 'terminate_process_tree', cleanup)
    with TestClient(create_app(settings), base_url='http://127.0.0.1:21431', client=('127.0.0.1',50000)) as client:
        headers = {'Authorization':'Bearer key-a'}
        discovery = client.get('/playground/api/models', headers=headers).json()
        assert [v['id'] for v in discovery['ocr']['verifiers']] == ['first','second','missing']
        assert next(m for m in discovery['models'] if m['id']=='ocr:verify')['operations'][0]['configured']
        payload = {'image':'x','mode':'score','engines':[],'candidates':['candidate']}
        for verifier, status in [(None,422), ('unknown',422), ('missing',503), ('second',200), ('first',200)]:
            response = client.post('/playground/api/ocr/run', headers=headers, json={**payload,'verifier':verifier})
            assert response.status_code == status
            if status == 200:
                assert response.json()['verifier'] == verifier
    assert [p['verifier'] for p in received] == ['second','first']


def test_retained_release_serialization_default_transition_and_shutdown(tmp_path, monkeypatch):
    from fake_ollama import ocr_playground
    config = tmp_path / 'ocr.json'
    config.write_text(json.dumps({'python':'fixture','engines':{'minicpm-ocr':{'required_files':[]}}}))
    app = create_app(_settings().model_copy(update={'playground_ocr_config':str(config)}))
    manager = app.state.ocr_resident
    entered, resume = threading.Event(), threading.Event()
    stopped = []

    async def retained(*args):
        entered.set()
        assert await asyncio.to_thread(resume.wait, 5)
        return {'keep_loaded':True, 'results':[]}

    async def stop():
        stopped.append(True)

    async def spawn(argv, **kwargs):
        assert stopped  # release-mode requests first clear the resident cache
        Path(argv[argv.index('--output')+1]).write_text('{"results":[]}')
        return type('Process', (), {'returncode':0})()

    async def cleanup(process):
        return True

    monkeypatch.setattr(manager, 'run', retained)
    monkeypatch.setattr(manager, 'stop_if_owned', stop)
    monkeypatch.setattr(ocr_playground, 'create_managed_subprocess_exec', spawn)
    monkeypatch.setattr(ocr_playground, 'terminate_process_tree', cleanup)
    headers = {'Authorization':'Bearer key-a'}
    with TestClient(app, base_url='http://127.0.0.1:21431', client=('127.0.0.1',50000)) as client:
        with ThreadPoolExecutor() as pool:
            task = pool.submit(client.post, '/playground/api/ocr/run', headers=headers, json={'image':'x','keep_loaded':True})
            assert entered.wait(3)
            try:
                assert client.post('/playground/api/ocr/release', headers=headers).status_code == 409
                assert not stopped
                assert client.post('/playground/api/ocr/run', headers=headers, json={'image':'x'}).status_code == 409
            finally:
                resume.set()
            assert task.result().status_code == 200
        assert client.post('/playground/api/ocr/release', headers=headers).status_code == 200
        assert len(stopped) == 1
        assert client.post('/playground/api/ocr/run', headers=headers, json={'image':'x'}).status_code == 200
        assert len(stopped) == 2
    assert manager.shutting_down and len(stopped) == 3
