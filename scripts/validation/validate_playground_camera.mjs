#!/usr/bin/env node
// End-to-end Playground camera validation.

/** Exercise the Playground camera pipeline with Chromium's fake camera.
 *
 * No npm packages are required. The API key is read only from
 * FAKE_OLLAMA_CAMERA_TEST_API_KEY and is never printed.
 */

import {existsSync, mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';

const DEFAULT_BROWSER_PATHS = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
];

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForJson(url, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
      lastError = new Error(`${url} returned HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(200);
  }
  throw lastError || new Error(`Timed out waiting for ${url}`);
}

class CdpClient {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.nextId = 1;
    this.pending = new Map();
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    await new Promise((resolve, reject) => {
      this.socket.addEventListener('open', resolve, {once: true});
      this.socket.addEventListener('error', reject, {once: true});
    });
    this.socket.addEventListener('message', (event) => {
      const message = JSON.parse(String(event.data));
      if (!message.id || !this.pending.has(message.id)) return;
      const {resolve, reject} = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message || JSON.stringify(message.error)));
      else resolve(message.result || {});
    });
    this.socket.addEventListener('close', () => {
      for (const {reject} of this.pending.values()) {
        reject(new Error('Browser debugging connection closed'));
      }
      this.pending.clear();
    });
  }

  call(method, params = {}) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('Browser debugging connection is not open'));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.socket.send(JSON.stringify({id, method, params}));
    });
  }

  close() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) this.socket.close();
  }
}

async function waitForPageReady(client, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  let stableChecks = 0;
  while (Date.now() < deadline) {
    try {
      const result = await client.call('Runtime.evaluate', {
        expression: "document.readyState === 'complete' && Boolean(document.getElementById('apiKey'))",
        returnByValue: true,
      });
      if (result.result?.value === true) {
        stableChecks += 1;
        if (stableChecks >= 2) return;
      } else {
        stableChecks = 0;
      }
    } catch {
      stableChecks = 0;
    }
    await delay(300);
  }
  throw new Error('Timed out waiting for the Playground page to finish loading');
}

async function main() {
  const pageUrl = argument('--url', 'http://127.0.0.1:21431/playground/');
  const modelId = argument('--model', 'mage-vl-local');
  const debugPort = Number(argument('--debug-port', '9223'));
  const browserPath = argument(
    '--browser',
    DEFAULT_BROWSER_PATHS.find((path) => existsSync(path)) || '',
  );
  const apiKey = process.env.FAKE_OLLAMA_CAMERA_TEST_API_KEY ?? '';
  if (!browserPath || !existsSync(browserPath)) throw new Error('Edge or Chrome was not found');
  if (!Number.isInteger(debugPort) || debugPort < 1024 || debugPort > 65535) {
    throw new Error('--debug-port must be an available TCP port from 1024 to 65535');
  }

  const profileDir = mkdtempSync(join(tmpdir(), 'mage-vl-camera-browser-'));
  const browser = spawn(browserPath, [
    '--headless=new',
    `--remote-debugging-port=${debugPort}`,
    '--remote-debugging-address=127.0.0.1',
    '--remote-allow-origins=*',
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    '--autoplay-policy=no-user-gesture-required',
    '--no-first-run',
    '--no-default-browser-check',
    `--user-data-dir=${profileDir}`,
    pageUrl,
  ], {stdio: 'ignore'});

  let pageClient;
  let browserClient;
  try {
    const version = await waitForJson(`http://127.0.0.1:${debugPort}/json/version`);
    browserClient = new CdpClient(version.webSocketDebuggerUrl);
    await browserClient.connect();
    const pages = await waitForJson(`http://127.0.0.1:${debugPort}/json/list`);
    const page = pages.find((entry) => entry.type === 'page' && entry.url.startsWith(pageUrl));
    if (!page) throw new Error(`Playground page was not found in Chromium targets: ${pageUrl}`);
    pageClient = new CdpClient(page.webSocketDebuggerUrl);
    await pageClient.connect();
    await waitForPageReady(pageClient);

    const expression = `
      (async () => {
        const apiKey = ${JSON.stringify(apiKey)};
        const modelId = ${JSON.stringify(modelId)};
        const byId = (id) => document.getElementById(id);
        const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        const waitFor = async (predicate, timeoutMs, label) => {
          const deadline = performance.now() + timeoutMs;
          while (performance.now() < deadline) {
            const value = predicate();
            if (value) return value;
            await sleep(200);
          }
          throw new Error('Timed out waiting for ' + label);
        };
        const change = (element, value) => {
          element.value = value;
          element.dispatchEvent(new Event('input', {bubbles: true}));
          element.dispatchEvent(new Event('change', {bubbles: true}));
        };

        await waitFor(() => byId('apiKey'), 15_000, 'Playground page DOM');
        change(byId('apiKey'), apiKey);
        byId('loadModels').click();
        await waitFor(
          () => Array.from(byId('model').options).some((option) => option.value === modelId),
          20_000,
          'Mage-VL discovery',
        );
        change(byId('model'), modelId);
        await waitFor(
          () => Array.from(byId('operation').options).some(
            (option) => option.value === 'video_analysis'
          ),
          5_000,
          'video analysis operation',
        );
        change(byId('operation'), 'video_analysis');
        await waitFor(
          () => document.querySelector('[data-parameter="segment_seconds"]'),
          5_000,
          'video analysis parameters',
        );
        change(byId('prompt'), '简要描述当前画面中可见的主体、动作和变化。');
        change(byId('maxTokens'), '32');
        change(document.querySelector('[data-parameter="segment_seconds"]'), '2');
        change(document.querySelector('[data-parameter="frames_per_segment"]'), '2');
        change(document.querySelector('[data-parameter="max_segments"]'), '1');
        byId('cameraFacing').value = '';
        await waitFor(() => !byId('cameraStart').disabled, 5_000, 'enabled camera button');
        byId('cameraStart').click();
        await waitFor(
          () => byId('cameraPreview').srcObject && byId('cameraStatus').textContent.includes('录制中'),
          15_000,
          'camera recording',
        );
        const trackSettings = byId('cameraPreview').srcObject.getVideoTracks()[0].getSettings();
        await waitFor(() => {
          const match = byId('cameraStats').textContent.match(/已分析 ([0-9]+)/);
          return match && Number(match[1]) >= 1;
        }, 120_000, 'first analyzed camera window');
        byId('cameraStop').click();
        await waitFor(
          () => byId('cameraStatus').textContent === '已停止',
          120_000,
          'camera analysis shutdown',
        );
        const analyzed = Number(byId('cameraStats').textContent.match(/已分析 ([0-9]+)/)?.[1] || 0);
        if (analyzed < 1) throw new Error('No camera window was analyzed');
        return {
          status: byId('cameraStatus').textContent,
          stats: byId('cameraStats').textContent,
          latest_result: byId('cameraLiveText').textContent,
          history_turns: document.querySelectorAll('.conversation-turn').length,
          track_settings: trackSettings,
          secure_context: window.isSecureContext,
        };
      })()
    `;
    const evaluation = await pageClient.call('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (evaluation.exceptionDetails) {
      const detail = evaluation.exceptionDetails.exception?.description
        || evaluation.exceptionDetails.text;
      throw new Error(detail || 'Browser evaluation failed');
    }
    console.log(JSON.stringify(evaluation.result.value, null, 2));
  } finally {
    try {
      if (browserClient) await browserClient.call('Browser.close');
    } catch {
      // The browser may have already closed after the page failed.
    }
    pageClient?.close();
    browserClient?.close();
    await Promise.race([
      new Promise((resolve) => browser.once('exit', resolve)),
      delay(5_000),
    ]);
    if (browser.exitCode === null) browser.kill();
    rmSync(profileDir, {recursive: true, force: true});
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});
