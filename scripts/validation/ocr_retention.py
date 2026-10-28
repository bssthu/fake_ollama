"""Real browser/model checks for OCR retention and sequential latency measurements."""
import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright, expect


async def main():
    root = Path.cwd()
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(root/'services/ocr/.browsers')
    key = json.loads((root/'config.json').read_text(encoding='utf-8'))['api_interfaces'][0]['access_tokens'][0]
    report = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width':1440,'height':1100})
        page = await context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            await page.goto('http://127.0.0.1:21431/playground/')
            await page.locator('#apiKey').fill(key)
            await page.locator('#loadModels').click()
            await expect(page.locator('#connectionStatus')).to_contain_text('个模型')
            for engine, verifier in [('ppocr-v6',''), ('hunyuan-ocr-current',''), ('minicpm-ocr','minicpm-ocr-jev')]:
                await page.locator('#model').select_option('ocr:' + engine)
                await page.locator('#fileInput').set_input_files(str(root/'.tmp/ocr-research/samples/mixed.png'))
                await page.locator('#ocrVerifier').select_option(verifier)
                await page.locator('#ocrLifetime').select_option('keep')
                await page.locator('#ocrRepeat').fill('3')
                await page.locator('#send').click()
                await expect(page.locator('#ocrBenchmark')).to_contain_text('完成 3/3', timeout=180000)
                await expect(page.locator('#stop')).to_be_hidden(timeout=15000)
                rows = [json.loads(text) for text in await page.locator('.ocr-results details pre').all_text_contents()]
                assert len(rows) == 3
                assert not rows[0]['results'][0]['model_reused']
                for result in rows[1:]:
                    assert result['results'][0]['model_reused']
                    assert all(event['reused'] and event['load_seconds']==0 for event in result['runtime_events'])
                assert not await page.locator('.ocr-failure').count()
                report[engine] = rows
                await expect(page.locator('#ocrRelease')).to_be_enabled()
                await page.screenshot(path=str(root/'.tmp/ocr-research/ocr-retention.png'), full_page=True)
                await page.locator('#ocrRelease').click()
                await expect(page.locator('#ocrRuntimeStatus')).to_have_text('当前无保留模型')
                print(engine, 'cold/warm roundtrip', [round(r['browser_seconds'],3) for r in rows], flush=True)
                await page.locator('#clear').click()

            # Default release mode must cold-load each round, including after keep mode.
            await page.locator('#model').select_option('ocr:hunyuan-ocr-current')
            await page.locator('#fileInput').set_input_files(str(root/'.tmp/ocr-research/samples/mixed.png'))
            await page.locator('#ocrVerifier').select_option('')
            await page.locator('#ocrLifetime').select_option('release')
            await page.locator('#ocrRepeat').fill('2')
            await page.locator('#send').click()
            await expect(page.locator('#ocrBenchmark')).to_contain_text('完成 2/2', timeout=180000)
            await expect(page.locator('#stop')).to_be_hidden(timeout=15000)
            rows = [json.loads(text) for text in await page.locator('.ocr-results details pre').all_text_contents()]
            assert len(rows)==2 and all(not r['keep_loaded'] and not r['results'][0]['model_reused'] for r in rows)
            report['release_mode'] = rows
            await page.locator('#ocrLifetime').select_option('keep')
            await page.locator('#ocrRepeat').fill('20')
            await page.locator('#send').click()
            await page.wait_for_timeout(1000)
            await page.locator('#stop').click()
            await expect(page.locator('#stop')).to_be_hidden(timeout=20000)
            await expect(page.locator('#ocrRuntimeStatus')).to_have_text('当前无保留模型')
            assert not errors, errors
            (root/'.tmp/ocr-research/retention-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        finally:
            await page.close()
            for _ in range(30):
                response = await context.request.post('http://127.0.0.1:21431/playground/api/ocr/release',headers={'Authorization':'Bearer '+key})
                if response.status != 409:
                    break
                await asyncio.sleep(.2)
            await browser.close()
        print('RETENTION_PASS: PP, Hunyuan, MiniCPM + Jev reuse; manual release; cold reload; cancel; no JS errors',flush=True)


if __name__=='__main__':
    asyncio.run(main())
