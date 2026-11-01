"""Real browser checks for function-first navigation and memory-only speech transport."""
import argparse
import asyncio
import json
import os
import time
import wave
import winsound
from pathlib import Path
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


async def main(args):
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(ROOT / 'services/ocr/.browsers')
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1050})
        page = await context.new_page()
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        await page.goto(args.base + '/playground/')
        await expect(page.locator('#privacyStatus')).to_contain_text('隐私模式')
        await page.locator('#loadModels').click()
        await expect(page.locator('#function')).to_be_enabled()
        tasks = await page.locator('#function option').evaluate_all('(items) => items.map(x=>x.value)')
        assert 'speech' in tasks and 'ocr_recognize' in tasks and 'chat' in tasks, tasks
        await page.locator('#function').select_option('ocr_recognize')
        models = await page.locator('#model option').evaluate_all('(items) => items.map(x=>x.value)')
        assert models and all(x.startswith('ocr:') for x in models)
        await page.locator('#function').select_option('speech')
        await expect(page.locator('#speechStart')).to_be_enabled()
        await expect(page.locator('#generalWorkspace')).to_be_hidden()
        assert await page.locator('#speechSource option').count() == 3
        await page.locator('#speechAsr').select_option('voxtral-mini-4b-realtime-2602')
        await expect(page.locator('#speechStreamingSettings')).to_be_visible()
        await page.locator('#speechAsr').select_option(args.asr)
        if args.file:
            await page.locator('#speechSource').select_option('file')
            await page.locator('#speechFile').set_input_files(str(args.file.resolve()))
            await page.locator('#speechStart').click()
            await expect(page.locator('#speechCancel')).to_be_enabled()
            await expect(page.locator('#speechStart')).to_be_enabled(timeout=args.timeout * 1000)
            error = await page.locator('#speechError').inner_text()
            assert not error, error
            rows = await page.locator('.speech-caption').count()
            assert rows > 0
            assert await page.locator('.speech-caption[data-state="complete"]').count() == rows
            private = '开启' in await page.locator('#privacyStatus').inner_text()
            assert await page.locator('#speechExportJson').is_disabled() == private
            print(json.dumps({'real_file': True, 'rows': rows, 'privacy': private,
                              'metrics': await page.locator('#speechMetrics').inner_text()}, ensure_ascii=True), flush=True)
        assert not errors, errors
        if args.loopback:
            await page.locator('#speechSource').select_option('loopback')
            async with page.expect_response(lambda r: r.url.endswith('/speech/sessions') and r.request.method == 'POST') as response:
                await page.locator('#speechStart').click()
            response = await response.value
            created = await response.json()
            key = response.request.headers.get('x-playground-session-key')
            headers = {'X-Playground-Session-Key': key}
            async def status():
                r = await context.request.get(args.base + '/playground/api/speech/sessions/' + created['id'], headers=headers)
                assert r.ok, await r.text()
                return await r.json()
            deadline = time.monotonic() + 120
            while (await status())['status'] != 'listening':
                assert time.monotonic() < deadline
                await asyncio.sleep(.25)
            await asyncio.sleep(2)
            with wave.open(str(args.loopback), 'rb') as media:
                duration = media.getnframes() / media.getframerate()
            winsound.PlaySound(str(args.loopback.resolve()), winsound.SND_FILENAME | winsound.SND_ASYNC)
            await asyncio.sleep(duration + 4)
            before = await status()
            rows = {event['segment']['id']: event['segment'] for event in before['events']}
            assert before['status'] == 'listening', {k:v for k,v in before.items() if k != 'events'}
            assert rows and all(row['state'] == 'complete' for row in rows.values())
            assert before['processed_seconds'] >= duration - .5, before['processed_seconds']
            print(json.dumps({'loopback_tail_before_stop': True, 'rows': len(rows), 'audio_seconds': duration,
                              'processed_seconds': before['processed_seconds']}), flush=True)
            await page.locator('#speechFinish').click()
            await expect(page.locator('#speechStart')).to_be_enabled(timeout=30000)
            assert not await page.locator('#speechError').inner_text()
            winsound.PlaySound(None, 0)
        await page.set_viewport_size({'width': 390, 'height': 844})
        await expect(page.locator('#privacyStatus')).to_be_visible()
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        print(json.dumps({'browser': 'passed', 'functions': tasks, 'console_errors': errors}), flush=True)
        await context.close()
        await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:21439')
    parser.add_argument('--file', type=Path)
    parser.add_argument('--asr', default='whisper-large-v3-turbo')
    parser.add_argument('--loopback', type=Path)
    parser.add_argument('--timeout', type=int, default=600)
    asyncio.run(main(parser.parse_args()))
