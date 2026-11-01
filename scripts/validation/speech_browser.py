"""Render the real Playground with HTTP fixtures only. Never starts any ML backend."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


class StaticHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        assets = {'/playground/': ('playground.html', 'text/html'),
                  '/playground/playground.css': ('playground.css', 'text/css'),
                  '/playground/playground.js': ('playground.js', 'text/javascript'),
                  '/playground/speech.js': ('speech.js', 'text/javascript')}
        item = assets.get(self.path)
        if not item:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', item[1] + '; charset=utf-8')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:;")
        self.end_headers()
        self.wfile.write((ROOT / 'fake_ollama/static' / item[0]).read_bytes())

    def log_message(self, *args):
        pass


async def main():
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(ROOT / 'services/ocr/.browsers')
    server = ThreadingHTTPServer(('127.0.0.1', 0), StaticHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = ROOT / '.tmp/speech-browser'
    report.mkdir(parents=True, exist_ok=True)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={'width': 1440, 'height': 1050})
            errors, calls = [], []
            page.on('pageerror', lambda e: errors.append(str(e)))
            state = {'enabled': False, 'started': False, 'polls': 0, 'cancelled': False}
            async def route_handler(route):
                req = route.request
                path = req.url.split('/playground/api/speech', 1)[1]
                calls.append((req.method, path))
                if path == '/models':
                    result = {'asr': [{'id':'asr','label':'Qwen3-ASR 1.7B（模拟）','available':True,'runtime_ready':True}],
                              'translators': [{'id':'translation','label':'Qwen3.5 2B（模拟）','available':True,'runtime_ready':True}],
                              'inference_enabled':state['enabled'], 'max_upload_bytes':10000000}
                elif path == '/devices':
                    result = {'devices':[{'id':'7','label':'Fixture speakers'}]}
                elif path == '/sessions':
                    assert req.method == 'POST'
                    payload = req.post_data_json
                    assert payload['target_language'] == 'zh' and payload['source'] == 'file'
                    result = {'id':'fixture-session'}
                elif '/file?' in path:
                    assert req.method == 'PUT' and req.post_data_buffer == b'audio-fixture-not-a-real-model-input'
                    result = {'uploaded_bytes':len(req.post_data_buffer)}
                else:
                    if path.endswith('/start'): state['started'] = True
                    if req.method == 'DELETE': state['cancelled'] = True
                    if state['started']: state['polls'] += 1
                    completed = state['polls'] >= 5
                    row = {'id':0,'start':0,'end':3.5,'text':'Hello 你好 <img src=x onerror=alert(1)>',
                           'translation':'你好，世界' if completed else '', 'language':'English,Chinese',
                           'state':'complete' if completed else 'translating'}
                    result = {'id':'fixture-session','status':'cancelled' if state['cancelled'] else 'translating',
                              'error':'','input_seconds':4,'processed_seconds':3.5,'backlog_seconds':.5,'segments':1,
                              'cursor':2 if completed else 1,'events':[{'sequence':2 if completed else 1,'segment':row}] if state['started'] else [],
                              'resources_released':state['cancelled']}
                await route.fulfill(status=200, content_type='application/json', body=json.dumps(result))
            await page.route('**/playground/api/speech/**', route_handler)
            await page.goto(f'http://127.0.0.1:{server.server_port}/playground/')
            await page.locator('#speechTab').click()
            await expect(page.locator('#speechReadiness')).to_contain_text('暂未启用')
            await expect(page.locator('#speechStart')).to_be_disabled()
            assert not any(path == '/sessions' for _, path in calls)
            await page.locator('#speechDevices').click()
            await expect(page.locator('#speechDevice option')).to_have_count(2)
            state['enabled'] = True
            await page.locator('#speechRefresh').click()
            await expect(page.locator('#speechStart')).to_be_enabled()
            await page.locator('#speechSource').select_option('file')
            await page.locator('#speechFile').set_input_files({'name':'fixture.wav','mimeType':'audio/wav','buffer':b'audio-fixture-not-a-real-model-input'})
            await page.locator('#speechStart').click()
            await expect(page.locator('.speech-caption-text p').first).to_contain_text('Hello 你好')
            await expect(page.locator('.speech-caption-text p').last).to_have_text('翻译中…')
            await expect(page.locator('.speech-caption-text p').last).to_have_text('你好，世界')
            assert await page.locator('#speechCaptions img').count() == 0
            await page.screenshot(path=str(report / 'desktop.png'), full_page=True)
            async with page.expect_download() as downloaded:
                await page.locator('#speechExportSrt').click()
            download = await downloaded.value
            await download.save_as(str(report / 'fixture.srt'))
            assert '你好，世界' in (report / 'fixture.srt').read_text('utf-8')
            await page.locator('#speechCancel').click()
            await expect(page.locator('#speechStatus')).to_have_text('已停止')
            await expect(page.locator('#speechStart')).to_be_enabled()
            await page.set_viewport_size({'width':390,'height':844})
            await page.screenshot(path=str(report / 'mobile.png'), full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            await page.locator('#generalTab').click()
            await expect(page.locator('#generalWorkspace')).to_be_visible()
            assert not errors, errors
            await browser.close()
            print('SPEECH_BROWSER_FIXTURE_PASS: disabled inference, upload, original-before-translation, escaped text, SRT, cancel, mobile layout; no models loaded')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    asyncio.run(main())
