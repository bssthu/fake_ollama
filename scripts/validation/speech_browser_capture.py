"""Real Chromium tab-audio capture; consent chooser is automated by a test flag."""
import asyncio
import json
import os
from pathlib import Path
import time
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
BASE = 'http://127.0.0.1:21439'


async def main():
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(ROOT / 'services/ocr/.browsers')
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='chromium', headless=True, ignore_default_args=['--mute-audio'],
            args=['--auto-select-tab-capture-source-by-title=Speech Capture Fixture'])
        context = await browser.new_context()
        await context.route('**/capture-fixture', lambda route: route.fulfill(content_type='text/html',
            body='<title>Speech Capture Fixture</title><button onclick="document.querySelector(\'audio\').play()">Play</button><audio src="/fixture.wav" loop></audio>'))
        await context.route('**/fixture.wav', lambda route: route.fulfill(content_type='audio/wav',
            headers={'Cache-Control': 'no-store'}, body=(ROOT / '.tmp/speech-live/english.wav').read_bytes()))
        fixture = await context.new_page()
        await fixture.goto(BASE + '/capture-fixture')
        await fixture.locator('button').click()
        page = await context.new_page()
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        await page.add_init_script('''
          window.workletPackets = {count: 0, peak: 0};
          const Node = window.AudioWorkletNode;
          window.AudioWorkletNode = class extends Node {
            constructor(...args) {
              super(...args);
              this.port.addEventListener('message', event => {
                if (event.data instanceof ArrayBuffer) {
                  workletPackets.count++;
                  for (const value of new Int16Array(event.data)) workletPackets.peak = Math.max(workletPackets.peak, Math.abs(value));
                }
              });
            }
          };
          const capture = navigator.mediaDevices.getDisplayMedia.bind(navigator.mediaDevices);
          window.captureCalls = [];
          navigator.mediaDevices.getDisplayMedia = async options => {
            window.captureCalls.push({active: navigator.userActivation.isActive, options});
            const result = await capture(options);
            window.capturedTracks = result.getTracks();
            return result;
          };
        ''')
        await page.goto(BASE + '/playground/')
        await page.locator('#loadModels').click()
        await expect(page.locator('#function')).to_be_enabled()
        await page.locator('#function').select_option('speech')
        await expect(page.locator('#speechStart')).to_be_enabled()
        await page.locator('#speechSource').select_option('browser')
        await page.locator('#speechStart').click()
        deadline = time.monotonic() + 60
        while await page.locator('.speech-caption[data-state="complete"]').count() < 1:
            if time.monotonic() >= deadline:
                print(json.dumps({'status': await page.locator('#speechStatus').inner_text(),
                    'metrics': await page.locator('#speechMetrics').inner_text(),
                    'packets': await page.evaluate('workletPackets'),
                    'tracks': await page.evaluate('capturedTracks.map(t=>({kind:t.kind, state:t.readyState, muted:t.muted,settings:t.getSettings()}))'),
                    'fixture': await fixture.evaluate('({time:document.querySelector("audio").currentTime, paused:document.querySelector("audio").paused})')}, ensure_ascii=True), flush=True)
                raise AssertionError('No captured speech')
            error = await page.locator('#speechError').inner_text()
            assert not error, error
            await asyncio.sleep(.5)
        calls = await page.evaluate('captureCalls')
        assert len(calls) == 1 and calls[0]['active'] and calls[0]['options']['audio']
        await page.locator('#speechFinish').click()
        await expect(page.locator('#speechStart')).to_be_enabled(timeout=30000)
        tracks = await page.evaluate('capturedTracks.map(t=>({kind:t.kind,state:t.readyState}))')
        assert any(t['kind'] == 'audio' for t in tracks)
        assert all(t['state'] == 'ended' for t in tracks), tracks
        assert not errors, errors
        print(json.dumps({'native_tab_capture': True, 'explicit_click_activation': True,
                          'tracks_released': tracks, 'rows': await page.locator('.speech-caption').count(),
                          'consent_test_method': 'Chromium auto-select test flag'}), flush=True)
        await context.close()
        await browser.close()
        # Exercise native browser rejection without allocating a server session.
        browser = await p.chromium.launch(channel='chromium', headless=True, args=['--auto-reject-capture'])
        page = await browser.new_page()
        created = []
        page.on('request', lambda request: created.append(request.url) if request.url.endswith('/speech/sessions') and request.method == 'POST' else None)
        await page.goto(BASE + '/playground/')
        await page.locator('#loadModels').click()
        await expect(page.locator('#function')).to_be_enabled()
        await page.locator('#function').select_option('speech')
        await expect(page.locator('#speechStart')).to_be_enabled()
        await page.locator('#speechSource').select_option('browser')
        await page.locator('#speechStart').click()
        await expect(page.locator('#speechError')).to_be_visible(timeout=15000)
        await expect(page.locator('#speechStart')).to_be_enabled()
        assert not created
        print(json.dumps({'native_consent_rejection': True, 'sessions_created': 0}), flush=True)
        await browser.close()


if __name__ == '__main__': asyncio.run(main())
