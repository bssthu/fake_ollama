"""Real browser/API/GPU validation; requires an explicitly started local Playground.

The loopback check plays the provided 26-second WAV on the default output device.
The sustained check processes a repeated media file without playing it audibly.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
import winsound

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.tmp/speech-live/browser'


async def main(args):
    global OUT
    if args.output_dir: OUT = args.output_dir
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(ROOT / 'services/ocr/.browsers')
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'base':args.base, 'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'checks':{}}
    def save():
        (OUT / 'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width':1440,'height':1050})
        page = await context.new_page()
        errors, observed = [], {}
        page.on('pageerror', lambda e: errors.append(str(e)))
        async def response(response):
            if '/api/speech/sessions/' in response.url and response.request.method == 'GET' and response.ok:
                data = await response.json()
                for event in data.get('events', []):
                    row = event['segment']
                    observed.setdefault(data['id'], {})[row['id']] = row
                    if row['state'] == 'translating': report['original_before_translation'] = True
        page.on('response', response)
        async def open_page():
            await page.goto(args.base + '/playground/')
            await page.locator('#speechTab').click()
            await expect(page.locator('#speechStart')).to_be_enabled(timeout=15000)
            await page.locator('#speechAsr').select_option('whisper-large-v3-turbo')
            await page.locator('#speechTranslator').select_option('qwen3.5-2b')
        async def start(source, path=None):
            await page.locator('#speechSource').select_option(source)
            if path: await page.locator('#speechFile').set_input_files(str(path))
            async with page.expect_response(lambda r:r.url.endswith('/api/speech/sessions') and r.request.method=='POST') as response_info:
                await page.locator('#speechStart').click()
            created = await (await response_info.value).json()
            assert 'id' in created, created
            return created['id']
        async def snapshot(sid):
            result = await context.request.get(args.base + '/playground/api/speech/sessions/' + sid)
            assert result.ok, await result.text()
            return await result.json()
        async def wait_for(sid, condition, timeout=600):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                data = await snapshot(sid)
                if condition(data): return data
                assert data['status'] != 'error', data
                await asyncio.sleep(.2)
            raise TimeoutError('Speech session did not reach expected state')
        async def finished(sid, allow_translation_errors=False):
            data = await wait_for(sid, lambda d:d['status'] in ('finished','cancelled','error') and d['resources_released'])
            assert data['status'] != 'error', data
            await expect(page.locator('#speechStart')).to_be_enabled(timeout=10000)
            rows = {e['segment']['id']:e['segment'] for e in data['events']}
            if not allow_translation_errors:
                assert all(r['state']=='complete' for r in rows.values()), [r for r in rows.values() if r['state']!='complete']
            return data, list(rows.values())
        try:
            await open_page()
            await page.locator('#speechDevices').click()
            await expect(page.locator('#speechDevice option')).not_to_have_count(1)
            sid = await start('file', args.wav)
            data, rows = await finished(sid)
            assert len(rows) >= 3 and all(r['translation'] for r in rows)
            report['checks']['file'] = {'state':data,'rows':rows}
            await page.screenshot(path=str(OUT / 'file-desktop.png'), full_page=True)
            for kind in ('Srt','Json'):
                async with page.expect_download() as downloaded:
                    await page.locator('#speechExport'+kind).click()
                await (await downloaded.value).save_as(str(OUT / ('file.'+kind.lower())))
            assert rows[0]['translation'] in (OUT / 'file.srt').read_text('utf-8')
            await page.set_viewport_size({'width':390,'height':844})
            assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            await page.screenshot(path=str(OUT / 'file-mobile.png'),full_page=True)
            await page.set_viewport_size({'width':1440,'height':1050})
            save()
            print('LIVE_FILE_PASS ' + str(len(rows)) + ' segments; desktop/mobile/export',flush=True)
            if args.file_only:
                assert not errors, errors
                report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                save()
                return

            sid = await start('loopback')
            await wait_for(sid, lambda d:d['status']=='listening')
            # Wait for the WASAPI subprocess to open before beginning known audio.
            await asyncio.sleep(2)
            winsound.PlaySound(str(args.wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
            await asyncio.sleep(29)
            winsound.PlaySound(None, 0)
            await page.locator('#speechFinish').click()
            data, rows = await finished(sid)
            assert rows and all('translation_latency_seconds' in r for r in rows), rows
            report['checks']['loopback'] = {'state':data, 'rows':rows}
            await page.screenshot(path=str(OUT / 'loopback.png'),full_page=True)
            save()
            print('LIVE_LOOPBACK_PASS ' + str(len(rows)) + ' segments',flush=True)

            if args.soak_seconds:
                sid = await start('file', args.soak_file)
                await wait_for(sid, lambda d:d['segments'] > 0)
                began = time.monotonic()
                samples = []
                while time.monotonic() - began < args.soak_seconds:
                    data = await snapshot(sid)
                    assert data['status'] not in ('error','cancelled','finished'), {k:v for k,v in data.items() if k!='events'}
                    gpu = await asyncio.to_thread(subprocess.check_output,
                        ['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                        text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    samples.append({'elapsed':round(time.monotonic()-began,1),'segments':data['segments'],
                                    'backlog':data['backlog_seconds'],'gpu':gpu.strip()})
                    report['checks']['soak'] = {'seconds':round(time.monotonic()-began,1),'samples':samples}
                    save()
                    print('SOAK ' + json.dumps(samples[-1]),flush=True)
                    await asyncio.sleep(min(30,max(0,args.soak_seconds-(time.monotonic()-began))))
                await page.locator('#speechFinish').click()
                data, rows = await finished(sid, allow_translation_errors=True)
                report['checks']['soak'].update(final=data, retained_rows=len(rows), minimum_active_wall_seconds=args.soak_seconds,
                    translation_errors=[r for r in rows if r['state']!='complete'])
                await page.screenshot(path=str(OUT / 'soak.png'),full_page=True)
                save()
                print('LIVE_SOAK_PASS',flush=True)

            # Cancel an actual model load, refresh during another load, and test
            # the real 45-second disconnect watchdog with DELETE deliberately lost.
            sid = await start('loopback')
            await asyncio.sleep(3)
            began = time.monotonic()
            await page.locator('#speechCancel').click()
            data, _ = await finished(sid)
            report['checks']['cancel_loading'] = {'seconds':round(time.monotonic()-began,3),'state':data}
            sid = await start('loopback')
            await asyncio.sleep(3)
            await page.reload()
            data = await wait_for(sid, lambda d:d['resources_released'],timeout=20)
            assert data['status']=='cancelled', data
            report['checks']['refresh_releases'] = True
            await open_page()
            sid = await start('loopback')
            await asyncio.sleep(3)
            await page.route('**/api/speech/sessions/**', lambda route: route.abort() if route.request.method=='DELETE' else route.continue_())
            await page.close()
            began = time.monotonic()
            while time.monotonic()-began < 60:
                response = await context.request.get(args.base + '/playground/api/speech/models')
                if not (await response.json())['busy']: break
                await asyncio.sleep(2)
            data = await snapshot(sid)
            assert data['resources_released'] and data['status']=='cancelled', data
            report['checks']['disconnect_watchdog'] = {'seconds':round(time.monotonic()-began,3),'state':data}
            assert not errors, errors
            report['page_errors'] = errors
            report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            save()
            print('LIVE_BROWSER_PASS',flush=True)
        except Exception as exc:
            report['error'] = str(exc)
            save()
            raise
        finally:
            winsound.PlaySound(None, 0)
            await browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:21439')
    parser.add_argument('--wav',type=Path,default=ROOT / '.tmp/speech-live/japanese.wav')
    parser.add_argument('--soak-seconds',type=int,default=1200)
    parser.add_argument('--soak-file',type=Path,default=ROOT / '.tmp/speech-live/soak.mka')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--file-only',action='store_true')
    asyncio.run(main(parser.parse_args()))
