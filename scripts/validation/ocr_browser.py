"""Rendered browser + authenticated local OCR verification; never prints credentials."""
import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright, expect

async def main():
    root=Path.cwd();os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(root/'services/ocr/.browsers')
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    key=config['api_interfaces'][0]['access_tokens'][0]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True)
        page=await browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.goto('http://127.0.0.1:21431/playground/')
        await page.locator('#loadModels').click()
        await expect(page.locator('#connectionStatus')).to_contain_text('本机免密')
        await page.locator('#apiKey').fill(key)
        await page.locator('#loadModels').click()
        await expect(page.locator('#authenticationHint')).to_contain_text('使用接口 API key')
        options = await page.locator('#model option').evaluate_all('(items) => items.map(e => e.value)')
        assert 'ocr:hunyuan-ocr' not in options and 'ocr:ppocr-clipaste' not in options
        await page.locator('#model').select_option('ocr:ppocr-v6')
        await page.locator('#fileInput').set_input_files(str(root/'.tmp/ocr-research/samples/mixed.png'))
        assert await page.locator('#ocrVerifier').input_value() == ''
        await page.locator('#send').click()
        await expect(page.locator('.ocr-summary')).to_contain_text('未使用复核模型', timeout=180000)
        assert await page.locator('.ocr-result').count() == 1
        assert await page.locator('.ocr-failure').count() == 0
        await page.locator('#operation').select_option('ocr_auto')
        await expect(page.locator('#send')).to_be_disabled()
        await page.locator('#ocrVerifier').select_option('minicpm-ocr-jev')
        await expect(page.locator('#send')).to_be_enabled()
        await page.locator('#model').select_option('ocr:compare')
        await expect(page.locator('#send')).to_be_enabled()
        await page.locator('#send').click()
        await expect(page.locator('.ocr-summary')).to_be_visible(timeout=180000)
        assert await page.locator('.ocr-result').count()==3
        await expect(page.locator('.ocr-summary')).to_contain_text('MiniCPM-OCR-Jev / F16')
        assert await page.locator('.ocr-failure').count()==0
        await page.screenshot(path=str(root/'.tmp/ocr-research/ocr-playground.png'),full_page=True)
        print('BROWSER_COMPARE_PASS: standalone PP without verifier, three engines with selected Jev', flush=True)
        await page.locator('#model').select_option('ocr:minicpm-ocr')
        await page.locator('#operation').select_option('ocr_auto')
        await page.locator('#ocrThreshold').fill('1')
        await page.locator('#send').click()
        await expect(page.locator('.ocr-summary')).to_be_visible(timeout=180000)
        assert await page.locator('.ocr-result').count()==2
        assert '重试 1 次' in await page.locator('.ocr-summary').first.inner_text()
        assert '需要人工核对' in await page.locator('.ocr-summary').first.inner_text()
        print('BROWSER_RETRY_PASS: bounded fallback in shared history', flush=True)
        await page.locator('#model').select_option('ocr:verify')
        await page.locator('#ocrCandidates textarea').fill('MiniCPM OCR 识别测试')
        await page.locator('#ocrAddCandidate').click()
        await page.locator('#ocrCandidates textarea').nth(1).fill('另一个候选')
        await page.locator('#send').click()
        await expect(page.locator('.ocr-summary')).to_be_visible(timeout=180000)
        assert await page.locator('.ocr-result').count()==2
        assert await page.locator('.ocr-failure').count()==0
        print('BROWSER_JEV_PASS: two external candidates scored', flush=True)
        await page.locator('#model').select_option('ocr:compare')
        assert await page.locator('.ocr-result').count()==3  # history survives model switches
        await page.locator('#model').select_option('ocr:minicpm-ocr')
        await page.locator('#operation').select_option('ocr_recognize')
        await page.locator('#send').click()
        await page.wait_for_timeout(1000)
        await page.locator('#stop').click()
        await expect(page.locator('#errorBox')).to_contain_text('请求已停止')
        await page.wait_for_timeout(3000)
        assert not errors,errors
        print('BROWSER_PASS: main-page comparison, bounded retry, Jev candidates, history, cancel, auth status; no JS errors')
        await browser.close()

if __name__=='__main__':asyncio.run(main())
