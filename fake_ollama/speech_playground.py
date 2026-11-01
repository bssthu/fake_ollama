"""Speech routes inherit the Playground listener's authentication and origin checks."""
import os
import asyncio
import hashlib
import json
from contextlib import suppress
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from .security import is_loopback_host, playground_anonymous_allowed
from .speech_config import SpeechRequest, catalog, files_ready, read_config
from .speech_runtime import SpeechManager


class SpeechJSONResponse(JSONResponse):
    def __init__(self, content, **kwargs):
        headers = dict(kwargs.pop('headers', None) or {})
        headers.update({'Cache-Control': 'no-store', 'Pragma': 'no-cache'})
        super().__init__(content, headers=headers, **kwargs)


def is_local_page(request):
    expected = request.scope.get('scheme', 'http') + '://' + request.headers.get('host', '')
    origin = request.headers.get('origin')
    return bool(request.client and is_loopback_host(request.client.host)
                and is_loopback_host(request.url.hostname or '')
                and request.client.host not in request.app.state.settings.playground_trusted_proxy_ips
                and (origin is None or origin == expected)
                and not any(request.headers.get(k) for k in ('forwarded', 'x-forwarded-for', 'x-forwarded-host')))


def local_capture_only(request):
    if not is_local_page(request) and not request.app.state.settings.playground_server_audio_remote_enabled:
        raise HTTPException(403, '服务器系统音频未向非 localhost 用户开放；请选择浏览器共享音频或文件')


def session_owner(request):
    credential = request.headers.get('authorization', '') or request.headers.get('x-api-key', '')
    client = request.headers.get('x-playground-session-key', '')
    return hashlib.sha256((credential + '\0' + client).encode()).hexdigest()


def register_speech_routes(app: FastAPI):
    manager = SpeechManager()
    app.state.speech_manager = manager

    @app.get('/playground/speech.js', include_in_schema=False)
    async def script():
        return Response(Path(__file__).with_name('static').joinpath('speech.js').read_text('utf-8'),
                        media_type='text/javascript', headers={'Cache-Control': 'no-store'})

    @app.get('/playground/speech-capture-worklet.js', include_in_schema=False)
    async def worklet():
        return Response(Path(__file__).with_name('static').joinpath('speech-capture-worklet.js').read_text('utf-8'),
                        media_type='text/javascript', headers={'Cache-Control': 'no-store'})

    @app.get('/playground/api/speech/models', include_in_schema=False, response_class=SpeechJSONResponse)
    async def models(request: Request):
        return {**catalog(read_config(request.app.state.settings)),
                'server_audio_allowed': is_local_page(request) or request.app.state.settings.playground_server_audio_remote_enabled,
                'busy': bool(manager.session and not manager.session.cleaned)}

    @app.get('/playground/api/speech/devices', include_in_schema=False, response_class=SpeechJSONResponse)
    async def devices(request: Request):
        local_capture_only(request)
        if os.name != 'nt':
            raise HTTPException(422, '系统回放采集需要 Windows WASAPI')
        return await manager.devices(read_config(request.app.state.settings))

    @app.post('/playground/api/speech/sessions', include_in_schema=False, response_class=SpeechJSONResponse)
    async def create(payload: SpeechRequest, request: Request):
        config = read_config(request.app.state.settings)
        if not config.get('inference_enabled', False):
            raise HTTPException(409, '语音模型推理尚未启用，请在本机语音配置中设置 inference_enabled=true')
        if payload.source == 'loopback':
            local_capture_only(request)
            if os.name != 'nt':
                raise HTTPException(422, '系统回放采集需要 Windows WASAPI')
        for group, key in [('asr', payload.asr), ('translators', payload.translator)]:
            spec = config[group].get(key)
            if spec is None:
                raise HTTPException(422, '未知的语音识别或翻译模型')
            if not files_ready(spec) or not Path(spec.get('python', config['python'])).is_file():
                raise HTTPException(503, '模型文件或隔离 Python 环境未就绪')
        if payload.asr_mode == 'native_streaming' and config['asr'][payload.asr].get('kind') != 'voxtral':
            raise HTTPException(422, '所选模型的本机运行时不支持原生流式输入')
        if payload.file_size > config.get('max_upload_bytes', 2 * 1024**3):
            raise HTTPException(413, '文件超过配置的大小限制')
        session = await manager.create(config, payload, session_owner(request))
        return session.snapshot()

    @app.put('/playground/api/speech/sessions/{session_id}/file', include_in_schema=False, response_class=SpeechJSONResponse)
    async def upload(session_id: str, request: Request, offset: int = Query(ge=0)):
        session = manager.get(session_id, session_owner(request))
        async with session.io_lock:
            if session.status != 'created' or session.payload.source != 'file':
                raise HTTPException(409, '当前会话不能上传文件')
            if offset != session.uploaded:
                raise HTTPException(409, f'上传偏移不匹配，服务端已接收 {session.uploaded} bytes')
            content = bytearray()
            async for block in request.stream():
                if session.status != 'created':
                    raise HTTPException(409, '上传已取消')
                content.extend(block)
                session.last_seen = time.monotonic()
                if len(content) > 4 * 1024**2:
                    raise HTTPException(413, '每次最多上传 4 MiB')
            if not content:
                raise HTTPException(422, '空文件块')
            if session.status != 'created' or session.cleaned:
                raise HTTPException(409, '上传已取消')
            if session.uploaded + len(content) > session.config.get('max_upload_bytes', 2 * 1024**3):
                raise HTTPException(413, '文件超过配置的上传大小限制')
            # Commit only complete HTTP chunks, so an interrupted upload is resumable.
            if session.uploaded + len(content) > 32 * 1024**2:
                raise HTTPException(413, 'HTTP 兼容上传的内存上限为 32 MiB；大文件请使用页面按需读取')
            session.upload_buffer.extend(content)
            session.uploaded += len(content)
            return {'uploaded_bytes': session.uploaded}

    @app.post('/playground/api/speech/sessions/{session_id}/start', include_in_schema=False, response_class=SpeechJSONResponse)
    async def start(session_id: str, request: Request):
        session = manager.get(session_id, session_owner(request))
        if session.payload.source == 'loopback': local_capture_only(request)
        async with session.io_lock:
            await session.start()
        return session.snapshot()

    @app.get('/playground/api/speech/sessions/{session_id}', include_in_schema=False, response_class=SpeechJSONResponse)
    async def poll(session_id: str, request: Request, after: int = Query(default=0, ge=0)):
        session = manager.get(session_id, session_owner(request))
        if session.payload.source == 'loopback':
            try: local_capture_only(request)
            except HTTPException:
                await session.cancel('服务器音频远程访问已关闭')
                raise
        return session.snapshot(after)

    @app.post('/playground/api/speech/sessions/{session_id}/finish', include_in_schema=False, response_class=SpeechJSONResponse)
    async def finish(session_id: str, request: Request):
        session = manager.get(session_id, session_owner(request))
        await session.finish()
        return session.snapshot()

    @app.delete('/playground/api/speech/sessions/{session_id}', include_in_schema=False, response_class=SpeechJSONResponse)
    async def cancel(session_id: str, request: Request):
        session = manager.get(session_id, session_owner(request))
        await session.cancel()
        return session.snapshot()

    @app.websocket('/playground/api/speech/sessions/{session_id}/stream')
    async def stream(websocket: WebSocket, session_id: str):
        from .routing import _playground_interface_for
        actual = getattr(app.state, 'actual_listener_settings', app.state.settings)
        server = websocket.scope.get('server')
        origin = websocket.headers.get('origin')
        expected = ('https' if websocket.url.scheme == 'wss' else 'http') + '://' + websocket.headers.get('host', '')
        scope = dict(websocket.scope, type='http', method='GET', scheme='https' if websocket.url.scheme == 'wss' else 'http')
        # Reuse the existing explicit proxy IP + public-origin allowlist when
        # TLS terminates at a proxy. Forwarded headers alone grant no access.
        allowed_origin = origin is None or origin == expected or playground_anonymous_allowed(Request(scope))
        if (not actual.playground_listener_enabled or not server or server[1] != actual.playground_port
                or not app.state.settings.playground_listener_enabled or server[1] != app.state.settings.playground_port
                or not allowed_origin):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = sender = None
        attached = False
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), 10)
            if len(raw) > 4096: raise ValueError('Authentication frame too large')
            hello = json.loads(raw)
            scope['headers'] = [(k, v) for k, v in scope['headers'] if k not in (b'authorization', b'x-playground-session-key')]
            scope['headers'].extend([(b'authorization', str(hello.get('authorization', '')).encode()),
                                     (b'x-playground-session-key', str(hello.get('client', '')).encode())])
            request = Request(scope)
            _playground_interface_for(request, app.state.settings)
            session = manager.get(session_id, session_owner(request))
            if session.payload.source == 'loopback': local_capture_only(request)
            if session.websocket is not None: raise ValueError('Session already connected')
            session.websocket = websocket
            attached = True
            await session.send({'type': 'ready'})

            async def publish():
                cursor = 0
                while True:
                    snapshot = session.snapshot(cursor)
                    await session.send({'type': 'snapshot', **snapshot})
                    cursor = snapshot['cursor']
                    if session.cleaned: return
                    await asyncio.sleep(.2)
            sender = asyncio.create_task(publish())
            while not session.cleaned:
                message = await websocket.receive()
                if message['type'] == 'websocket.disconnect': break
                session.last_seen = time.monotonic()
                if session.payload.source == 'loopback': local_capture_only(request)
                binary = message.get('bytes')
                if binary is not None:
                    if len(binary) > 1024**2 + 16: raise ValueError('Frame too large')
                    if session.payload.source == 'file':
                        session.range_reply(binary[:16].decode('ascii'), binary[16:])
                    elif session.payload.source == 'browser' and session.status not in ('created', 'loading_asr', 'loading_translation'):
                        await session.feed(binary, captured_at=time.monotonic())
                    else: raise ValueError('Audio source is not ready')
                else:
                    raw = message.get('text', '')
                    if len(raw) > 4096: raise ValueError('Control frame too large')
                    command = json.loads(raw).get('type')
                    if command == 'start': await session.start()
                    elif command == 'finish': await session.finish()
                    elif command == 'cancel': await session.cancel()
                    elif command == 'idle' and session.payload.source == 'browser': await session.feed(kind='idle')
                    elif command != 'heartbeat': raise ValueError('Unknown speech command')
            if session.cleaned: await session.send({'type': 'snapshot', **session.snapshot()})
        except (WebSocketDisconnect, asyncio.CancelledError): pass
        except Exception as exc:
            with suppress(Exception): await websocket.send_json({'type': 'error', 'error': str(exc)[:2000]})
        finally:
            if sender:
                sender.cancel()
                with suppress(asyncio.CancelledError, Exception): await sender
            if session and attached:
                session.websocket = None
                if not session.cleaned: await session.cancel('浏览器音频连接已断开')
            with suppress(Exception): await websocket.close()
