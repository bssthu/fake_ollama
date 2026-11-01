"""Real browser/GPU provisional translations before native ASR finalization.

Print timing/count metadata only; captions and media remain in memory.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


async def main(args):
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(ROOT / 'services/ocr/.browsers')
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = await context.new_page()
        errors, previews = [], []
        sources = {}
        first_text = None
        updating_preview = asyncio.Event()
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        def received(raw):
            nonlocal first_text
            if not isinstance(raw, str): return
            data = json.loads(raw)
            for event in data.get('events', []):
                row = event['segment']
                if row['state'] == 'recognizing' and row.get('text'):
                    if row.get('translation_in_progress') and row.get('translation'): updating_preview.set()
                    first_text = first_text or time.monotonic()
                    key = (row['id'], row.get('translation_source'))
                    if row.get('translation') and key not in sources:
                        sources[key] = True
                        assert row.get('translation_is_partial')
                        previews.append({'row': row['id'], 'audio_end': row['end'],
                                         'seconds_after_first_text': time.monotonic() - first_text,
                                         'source_chars': len(row['text']), 'translation_chars': len(row['translation'])})
        page.on('websocket', lambda ws: ws.on('framereceived', received))
        await page.goto(args.base + '/playground/')
        await page.locator('#loadModels').click()
        await page.locator('#function').select_option('speech')
        await expect(page.locator('#speechStart')).to_be_enabled()
        await page.locator('#speechAsr').select_option('voxtral-mini-4b-realtime-2602')
        await page.locator('#speechTranslator').select_option(args.translator)
        await expect(page.locator('#speechTranslationInterval')).to_have_value('2')
        await page.locator('#speechSource').select_option('file')
        await page.locator('#speechFile').set_input_files(str(args.file.resolve()))
        async with page.expect_response(lambda r: r.url.endswith('/speech/sessions') and r.request.method == 'POST') as created:
            await page.locator('#speechStart').click()
        created = await created.value
        session = await created.json()
        await expect(page.locator('#speechTranslationInterval')).to_be_disabled()
        if args.cancel:
            await asyncio.wait_for(updating_preview.wait(), 120)
            started = time.monotonic()
            await page.locator('#speechCancel').click()
            await expect(page.locator('#speechStart')).to_be_enabled(timeout=30000)
            response = await context.request.get(args.base + '/playground/api/speech/sessions/' + session['id'],
                       headers={'X-Playground-Session-Key': created.request.headers['x-playground-session-key']})
            assert response.ok
            snapshot = await response.json()
            assert snapshot['resources_released'] and snapshot['queued_frames'] == 0
            assert not errors, errors
            print(json.dumps({'cancel_during_preview_update': True, 'released': True,
                              'release_seconds': time.monotonic() - started, 'console_errors': errors}), flush=True)
            await context.close()
            await browser.close()
            return
        await expect(page.locator('#speechStart')).to_be_enabled(timeout=180000)
        assert not await page.locator('#speechError').inner_text()
        assert len(previews) >= 2, previews
        assert previews[0]['audio_end'] < 30, previews[0]
        assert len({p['row'] for p in previews}) < len(previews), 'Need multiple revisions of one caption'
        rows = await page.locator('.speech-caption').count()
        assert rows > 0 and await page.locator('.speech-caption[data-state="complete"]').count() == rows
        assert await page.locator('#speechExportJson').is_disabled()
        assert not errors, errors
        print(json.dumps({'translator': args.translator, 'preview_updates': len(previews),
                          'first_preview': previews[0], 'final_rows': rows, 'console_errors': errors}), flush=True)
        await context.close()
        await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:21439')
    parser.add_argument('--file', type=Path, default=ROOT / '.tmp/speech-live/japanese.wav')
    parser.add_argument('--translator', default='qwen3.5-2b')
    parser.add_argument('--cancel', action='store_true')
    asyncio.run(main(parser.parse_args()))
