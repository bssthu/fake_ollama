"""Isolated real Playground with a local file-triggered graceful shutdown probe."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import uvicorn
from fake_ollama.config import load_settings
from fake_ollama.server import create_app, request_shutdown
from fake_ollama.privacy import configure_privacy_mode


async def main(args):
    configure_privacy_mode(args.privacy_mode)
    settings = load_settings(str(args.config)).model_copy(update={
        'playground_enabled':True, 'playground_host':'127.0.0.1',
        'playground_port':args.port,'admin_enabled':False,'dashboard_enabled':False})
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,
        proxy_headers=False,access_log=False,log_level='warning'))
    args.stop_file.unlink(missing_ok=True)
    args.report.write_text(json.dumps({'pid':os.getpid(),'port':args.port}),encoding='utf-8')
    async def monitor():
        while not args.stop_file.exists(): await asyncio.sleep(.2)
        request_shutdown(app)
        server.should_exit = True
    task=asyncio.create_task(monitor())
    try:
        print('SPEECH_TEST_SERVER '+str(os.getpid()),flush=True)
        await server.serve()
    finally:
        task.cancel()
        session=app.state.speech_manager.session
        result={'pid':os.getpid(),'port':args.port,'stopped':True,
                'resources_released':session.cleaned if session else True,
                'temporary_audio_removed':not session.upload_buffer if session else True}
        args.report.write_text(json.dumps(result),encoding='utf-8')
        print('SERVER_SHUTDOWN '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--port',type=int,default=21439)
    parser.add_argument('--privacy-mode', action='store_true')
    parser.add_argument('--stop-file',type=Path,default=ROOT/'.tmp/speech-live/stop-server')
    parser.add_argument('--report',type=Path,default=ROOT/'.tmp/speech-live/server.json')
    asyncio.run(main(parser.parse_args()))
