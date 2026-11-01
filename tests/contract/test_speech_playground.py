import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

from fake_ollama.server import create_app
from tests.contract.test_playground import _settings


def configured(tmp_path, enabled=False):
    (tmp_path / 'model.bin').write_bytes(b'fixture, never load')
    config = {'python': sys.executable, 'inference_enabled': enabled,
              'asr': {'asr': {'path': str(tmp_path), 'required_files': ['model.bin']}},
              'translators': {'translation': {'path': str(tmp_path), 'required_files': ['model.bin']}}}
    path = tmp_path / 'speech.json'
    path.write_text(json.dumps(config))
    return _settings().model_copy(update={'playground_speech_config': str(path)})


def client_for(app):
    return TestClient(app, base_url='http://127.0.0.1:21431', client=('127.0.0.1', 50000))


def test_auth_discovery_and_disabled_inference_never_spawn(tmp_path, monkeypatch):
    from fake_ollama import speech_runtime
    async def forbidden(*args, **kwargs): raise AssertionError('must not launch any process')
    monkeypatch.setattr(speech_runtime, 'create_managed_subprocess_exec', forbidden)
    with client_for(create_app(configured(tmp_path))) as client:
        assert client.get('/playground/speech.js').status_code == 200
        assert client.get('/playground/api/speech/models').status_code in (401, 403)
        headers = {'Authorization': 'Bearer key-a'}
        result = client.get('/playground/api/speech/models', headers=headers).json()
        assert not result['inference_enabled'] and result['asr'][0]['available']
        response = client.post('/playground/api/speech/sessions', headers=headers,
                               json={'asr': 'asr', 'translator': 'translation', 'source': 'file'})
        assert response.status_code == 409


def test_upload_order_limit_session_identity_and_cancel_cleanup(tmp_path):
    app = create_app(configured(tmp_path, enabled=True))
    with client_for(app) as client:
        headers = {'Authorization': 'Bearer key-a'}
        payload = {'asr': 'asr', 'translator': 'translation', 'source': 'file'}
        assert client.post('/playground/api/speech/sessions', headers=headers, json={**payload, 'asr': 'bad'}).status_code == 422
        session = client.post('/playground/api/speech/sessions', headers=headers, json=payload).json()
        url = '/playground/api/speech/sessions/' + session['id']
        assert client.get(url, headers=headers).headers['cache-control'] == 'no-store'
        assert client.post('/playground/api/speech/sessions', headers=headers, json=payload).status_code == 409
        assert client.get('/playground/api/speech/sessions/wrong', headers=headers).status_code == 404
        assert client.post(url + '/start', headers=headers).status_code == 422
        assert client.put(url + '/file?offset=0', headers=headers, content=b'first').json()['uploaded_bytes'] == 5
        assert client.put(url + '/file?offset=0', headers=headers, content=b'duplicate').status_code == 409
        assert client.put(url + '/file?offset=5', headers=headers, content=b'x' * (4 * 1024**2 + 1)).status_code == 413
        assert client.put(url + '/file?offset=-1', headers=headers, content=b'x').status_code == 422
        session_state = app.state.speech_manager.session
        assert session_state.upload_buffer == b'first'
        result = client.delete(url, headers=headers).json()
        assert result['status'] == 'cancelled' and result['resources_released']
        assert not session_state.upload_buffer
        # A subsequent session is independent and no stale source can be read.
        newer = client.post('/playground/api/speech/sessions', headers=headers, json=payload).json()
        assert newer['id'] != session['id']
        assert client.get(url, headers=headers).status_code == 404
    assert app.state.speech_manager.shutting_down
    assert app.state.speech_manager.session.cleaned


def test_speech_routes_are_not_available_on_api_listener(tmp_path):
    app = create_app(configured(tmp_path))
    with TestClient(app, base_url='http://127.0.0.1:21435', client=('127.0.0.1', 50000)) as client:
        assert client.get('/playground/api/speech/models', headers={'Authorization': 'Bearer key-a'}).status_code == 404


def test_system_capture_rejects_remote_client(tmp_path):
    app = create_app(configured(tmp_path, enabled=True))
    with TestClient(app, base_url='http://127.0.0.1:21431', client=('192.0.2.25', 50000)) as client:
        response = client.post('/playground/api/speech/sessions', headers={'Authorization': 'Bearer key-a'},
                               json={'asr': 'asr', 'translator': 'translation', 'source': 'loopback'})
        assert response.status_code == 403


def test_remote_audio_switch_and_proxy_boundary(tmp_path):
    settings = configured(tmp_path, enabled=True)
    headers = {'Authorization': 'Bearer key-a'}
    body = {'asr': 'asr', 'translator': 'translation', 'source': 'loopback'}
    with client_for(create_app(settings)) as client:
        assert client.post('/playground/api/speech/sessions', headers={**headers, 'X-Forwarded-For': '192.0.2.5'}, json=body).status_code == 403
        assert client.post('/playground/api/speech/sessions', headers={**headers, 'Origin': 'https://proxy.example'}, json=body).status_code == 403
    settings = settings.model_copy(update={'playground_server_audio_remote_enabled': True})
    with TestClient(create_app(settings), base_url='http://127.0.0.1:21431', client=('192.0.2.5', 5000)) as client:
        result = client.post('/playground/api/speech/sessions', headers=headers, json=body)
        assert result.status_code == 200


def test_session_is_bound_to_credential_and_page_key(tmp_path):
    app = create_app(configured(tmp_path, enabled=True))
    headers = {'Authorization': 'Bearer key-a', 'X-Playground-Session-Key': 'page-a'}
    with client_for(app) as client:
        session = client.post('/playground/api/speech/sessions', headers=headers,
            json={'asr': 'asr', 'translator': 'translation', 'source': 'browser'}).json()
        url = '/playground/api/speech/sessions/' + session['id']
        assert client.get(url, headers={**headers, 'X-Playground-Session-Key': 'page-b'}).status_code == 404
        assert client.get(url, headers=headers).status_code == 200
        assert client.delete(url, headers=headers).json()['resources_released']


def test_websocket_idle_commits_tail_and_duplicate_connection_cannot_cancel(tmp_path, monkeypatch):
    from fake_ollama import speech_runtime
    from tests.unit.test_speech import frame
    async def call(worker, payload, pcm=None):
        if payload['op'] == 'transcribe': return {'text': 'public fixture', 'language': 'en', 'seconds': .01}
        if payload['op'] == 'translate': return {'translation': 'fixture translation', 'seconds': .01}
        return {'loaded': True}
    monkeypatch.setattr(speech_runtime.ModelWorker, 'call', call)
    app = create_app(configured(tmp_path, enabled=True))
    headers = {'Authorization': 'Bearer key-a'}
    with client_for(app) as client:
        data = client.post('/playground/api/speech/sessions', headers=headers,
            json={'asr': 'asr', 'translator': 'translation', 'source': 'browser'}).json()
        url = '/playground/api/speech/sessions/' + data['id']
        with client.websocket_connect('ws://127.0.0.1:21431' + url + '/stream', headers={'Origin': 'http://127.0.0.1:21431'}) as ws:
            ws.send_json({'authorization': 'Bearer key-a'})
            assert ws.receive_json()['type'] == 'ready'
            with client.websocket_connect('ws://127.0.0.1:21431' + url + '/stream') as duplicate:
                duplicate.send_json({'authorization': 'Bearer key-a'})
                assert duplicate.receive_json()['type'] == 'error'
            assert not app.state.speech_manager.session.cleaned
            ws.send_json({'type': 'start'})
            while ws.receive_json().get('status') != 'listening': pass
            ws.send_bytes(frame())
            ws.send_json({'type': 'idle'})
            while True:
                update = ws.receive_json()
                rows = [event['segment'] for event in update.get('events', [])]
                if any(row['state'] == 'complete' for row in rows): break
            # No manual finish was needed to expose the short final utterance.
            assert app.state.speech_manager.session.count == 1
            ws.send_json({'type': 'finish'})
            while not ws.receive_json().get('resources_released'): pass


@pytest.mark.parametrize('trusted,allowed', [(True, True), (True, False), (False, True)])
def test_websocket_reuses_explicit_https_proxy_origin_policy(tmp_path, trusted, allowed):
    from starlette.websockets import WebSocketDisconnect
    settings = configured(tmp_path, enabled=True).model_copy(update={
        'playground_trusted_proxy_ips': ['127.0.0.1'] if trusted else [],
        'playground_allowed_origins': ['https://playground.example'] if allowed else []})
    app = create_app(settings)
    with client_for(app) as client:
        response = client.post('/playground/api/speech/sessions', headers={'Authorization': 'Bearer key-a'},
                               json={'asr': 'asr', 'translator': 'translation', 'source': 'browser'})
        url = 'ws://127.0.0.1:21431/playground/api/speech/sessions/' + response.json()['id'] + '/stream'
        headers = {'Host': 'playground.example', 'Origin': 'https://playground.example'}
        if trusted and allowed:
            with client.websocket_connect(url, headers=headers) as ws:
                ws.send_json({'authorization': 'Bearer key-a'})
                assert ws.receive_json()['type'] == 'ready'
                ws.send_json({'type': 'cancel'})
                while not ws.receive_json().get('resources_released'): pass
        else:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(url, headers=headers): pass
