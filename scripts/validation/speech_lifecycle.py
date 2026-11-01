"""Real local API failure/cleanup checks after the sustained browser run.

Loads GPU models and finally stops only the dedicated speech_server.py instance
through its private stop file. Never targets the normal Playground service.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'.tmp/speech-live'


async def main(args):
    report={}
    def save(): (WORK/'additional-lifecycle.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    async with httpx.AsyncClient(base_url=args.base+'/playground/api/speech',trust_env=False,timeout=60) as client:
        async def create(source):
            r=await client.post('/sessions',json={'source':source,'asr':'whisper-large-v3-turbo',
                  'translator':'qwen3.5-2b','target_language':'zh'})
            r.raise_for_status()
            return r.json()['id']
        async def wait(sid, condition, timeout=180):
            began=time.monotonic()
            while time.monotonic()-began < timeout:
                r=await client.get('/sessions/'+sid)
                r.raise_for_status()
                data=r.json()
                if condition(data): return data
                await asyncio.sleep(.2)
            raise TimeoutError('Session condition timed out')
        sid=await create('file')
        sending=asyncio.Event()
        hold=asyncio.Event()
        async def body():
            yield b'partial upload'
            sending.set()
            await hold.wait()
            yield b' tail'
        task=asyncio.create_task(client.put('/sessions/'+sid+'/file?offset=0',content=body(),headers={'Content-Type':'application/octet-stream'}))
        await sending.wait()
        await asyncio.sleep(.3)
        began=time.monotonic()
        r=await client.delete('/sessions/'+sid)
        r.raise_for_status()
        assert r.json()['resources_released'],r.json()
        report['cancel_incomplete_http_upload_seconds']=round(time.monotonic()-began,3)
        hold.set()
        r=await task
        assert r.status_code==409,r.text
        save()
        print('SLOW_UPLOAD_CANCEL_PASS',flush=True)

        sid=await create('file')
        r=await client.put('/sessions/'+sid+'/file?offset=0',content=b'not a media container',headers={'Content-Type':'application/octet-stream'})
        r.raise_for_status()
        (await client.post('/sessions/'+sid+'/start')).raise_for_status()
        data=await wait(sid,lambda d:d['status']=='error' and d['resources_released'])
        assert '无法解码音频' in data['error'],data
        report['invalid_media_after_model_load']={'error':data['error'],'resources_released':True}
        save()
        print('INVALID_MEDIA_GPU_CLEANUP_PASS',flush=True)

        # No per-session requests for 45 seconds. Catalog reads do not renew its lease.
        sid=await create('loopback')
        (await client.post('/sessions/'+sid+'/start')).raise_for_status()
        began=time.monotonic()
        while time.monotonic()-began<65:
            await asyncio.sleep(2)
            r=await client.get('/models')
            r.raise_for_status()
            if not r.json()['busy']: break
        data=(await client.get('/sessions/'+sid)).json()
        assert data['resources_released'] and data['status']=='cancelled' and '断开' in data['error'],data
        report['abandoned_client_watchdog_seconds']=round(time.monotonic()-began,3)
        save()
        print('WATCHDOG_45_SECONDS_PASS',flush=True)

        sid=await create('loopback')
        (await client.post('/sessions/'+sid+'/start')).raise_for_status()
        await wait(sid,lambda d:d['status']=='listening')
        await asyncio.sleep(2)
        began=time.monotonic()
        (WORK/'stop-server').touch()
        for _ in range(150):
            data=json.loads((WORK/'server.json').read_text('utf-8'))
            if data.get('stopped'): break
            await asyncio.sleep(.2)
        assert data.get('stopped') and data['resources_released'] and data['temporary_audio_removed'],data
        report['service_shutdown_with_loaded_models_seconds']=round(time.monotonic()-began,3)
        report['service_shutdown']=data
        save()
        print('SERVICE_EXIT_LOADED_MODELS_PASS '+json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:21439')
    asyncio.run(main(parser.parse_args()))
