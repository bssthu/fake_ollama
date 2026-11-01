"""Real native streaming cancellation and browser disconnect; metadata only."""
import asyncio
import json
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
BASE = 'http://127.0.0.1:21439'


async def main():
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(ROOT / 'services/ocr/.browsers')
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        for action in ('cancel', 'disconnect'):
            page = await context.new_page()
            await page.goto(BASE + '/playground/')
            await page.locator('#loadModels').click()
            await page.locator('#function').select_option('speech')
            await expect(page.locator('#speechStart')).to_be_enabled()
            await page.locator('#speechAsr').select_option('voxtral-mini-4b-realtime-2602')
            await page.locator('#speechSource').select_option('file')
            await page.locator('#speechFile').set_input_files('R:/测试视频.mkv')
            async with page.expect_response(lambda r: r.url.endswith('/speech/sessions') and r.request.method == 'POST') as created:
                await page.locator('#speechStart').click()
            response = await created.value
            session = await response.json()
            headers = {'X-Playground-Session-Key': response.request.headers['x-playground-session-key']}
            # The native generation thread must actually have emitted text.
            await expect(page.locator('.speech-caption[data-state="recognizing"]').first).to_be_visible(timeout=180000)
            started = time.monotonic()
            if action == 'cancel':
                await page.locator('#speechCancel').click()
                await expect(page.locator('#speechStart')).to_be_enabled(timeout=30000)
            else:
                await page.close()
            while True:
                response = await context.request.get(BASE + '/playground/api/speech/sessions/' + session['id'], headers=headers)
                assert response.ok
                snapshot = await response.json()
                if snapshot['resources_released']:
                    assert snapshot['queued_frames'] == 0
                    break
                assert time.monotonic() - started < 30
                await asyncio.sleep(.1)
            print(json.dumps({'native_stream': action, 'released': True, 'seconds': time.monotonic() - started}), flush=True)
            if not page.is_closed(): await page.close()
        await context.close()
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
