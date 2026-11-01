(() => {
  'use strict';
  function timestamp(seconds) {
    const ms = Math.max(0, Math.round(seconds * 1000));
    return `${String(Math.floor(ms / 3600000)).padStart(2, '0')}:${String(Math.floor(ms / 60000) % 60).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`;
  }
  function subtitles(rows) {
    return rows.filter(r => r.text).map((r, i) => `${i + 1}\n${timestamp(r.start)} --> ${timestamp(r.end)}\n${r.text}\n${r.translation || '[译文未完成]'}${r.translation && r.translation_is_partial ? '\n[临时译文，尚未确认]' : ''}\n`).join('\n');
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = { timestamp, subtitles };
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id);
  const base = '/playground/api/speech';
  const terminal = new Set(['finished', 'cancelled', 'error']);
  const labels = {created: '准备音频', loading_asr: '加载识别模型…', loading_translation: '加载翻译模型…',
    listening: '接收音频', transcribing: '识别中…', translating: '翻译中…', draining: '处理剩余音频…',
    finished: '已完成', cancelled: '已停止', error: '任务出错'};
  const state = {catalog: null, session: null, key: null, busy: false, rows: new Map(), nodes: new Map(),
    client: crypto.randomUUID(), socket: null, stream: null, audio: null, node: null, heartbeat: null,
    privacy: false, file: null, captureStarted: false, resultOptions: {}, flush: null, generation: 0, failing: false,
    finishing: false, lastHeartbeat: 0};
  function error(message = '') { $('speechError').textContent = message; $('speechError').hidden = !message; }
  function headers() {
    const key = state.key === null ? $('apiKey').value.trim() : state.key;
    return {...(key ? {Authorization: `Bearer ${key}`} : {}), 'X-Playground-Session-Key': state.client};
  }
  async function api(path, options = {}) {
    const response = await fetch(base + path, {...options, cache: 'no-store', headers: {...headers(), ...options.headers}});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result));
    return result;
  }
  function controls() {
    const ready = state.catalog?.inference_enabled && !state.catalog?.busy && $('speechAsr').selectedOptions[0]?.disabled === false && $('speechTranslator').selectedOptions[0]?.disabled === false;
    for (const id of ['speechAsr', 'speechTranslator', 'speechTarget', 'speechSource', 'speechDevice', 'speechDevices', 'speechFile', 'speechWindow', 'speechSilence', 'speechMode', 'speechDelay', 'speechContext', 'speechTranslationInterval', 'speechDetection']) $(id).disabled = state.busy;
    $('speechStart').disabled = state.busy || !ready;
    $('speechCancel').disabled = !state.busy;
    $('speechFinish').disabled = !state.busy || !state.session || state.finishing;
    $('speechExportSrt').disabled = $('speechExportJson').disabled = state.privacy || !state.rows.size;
    $('speechExportSrt').title = $('speechExportJson').title = state.privacy ? '隐私模式禁止将字幕保存到文件' : '';
    document.dispatchEvent(new CustomEvent('speech-busy', {detail: {busy: state.busy}}));
  }
  function notes(resetModes = false) {
    const asr = state.catalog?.asr.find(x => x.id === $('speechAsr').value);
    $('speechAsrNote').textContent = asr?.note || '';
    $('speechTranslatorNote').textContent = state.catalog?.translators.find(x => x.id === $('speechTranslator').value)?.note || '';
    if (resetModes) {
      $('speechWindow').value = asr?.recommended_segment_seconds || 8;
      $('speechMode').replaceChildren(new Option('模型推荐方式', 'auto'), new Option('分段识别', 'segments'));
      if (asr?.modes?.includes('native_streaming')) $('speechMode').append(new Option('原生流式识别', 'native_streaming'));
    }
    const native = asr?.modes?.includes('native_streaming') && $('speechMode').value !== 'segments';
    $('speechModeNote').textContent = asr?.streaming_note || '';
    $('speechStreamingSettings').hidden = !native;
    $('speechWindowField').hidden = native;
    $('speechDetectionField').hidden = asr?.kind !== 'whisper';
    const source = $('speechSource').value;
    $('speechDeviceField').hidden = source !== 'loopback';
    $('speechFileField').hidden = source !== 'file';
    $('speechBrowserNotice').hidden = source !== 'browser';
    controls();
  }
  async function refresh() {
    try {
      error();
      state.catalog = await api('/models');
      state.privacy = Boolean(state.catalog.privacy?.enabled);
      for (const [id, items] of [['speechAsr', state.catalog.asr], ['speechTranslator', state.catalog.translators]]) {
        const old = $(id).value || (id === 'speechAsr' ? state.catalog.default_asr : state.catalog.default_translator);
        $(id).replaceChildren(...items.map(item => {
          const option = new Option(item.label, item.id);
          option.disabled = !item.available || !item.runtime_ready;
          return option;
        }));
        if (items.some(x => x.id === old)) $(id).value = old;
      }
      const server = [...$('speechSource').options].find(x => x.value === 'loopback');
      server.disabled = !state.catalog.loopback_supported || !state.catalog.server_audio_allowed;
      if (server.disabled && $('speechSource').value === 'loopback') $('speechSource').value = 'browser';
      $('speechReadiness').textContent = state.catalog.busy ? '已有语音任务正在运行。结束后点击“连接并获取功能列表”刷新。' : state.catalog.inference_enabled ? '模型在开始后加载；停止或断开连接会释放模型。' : '服务端尚未启用语音模型推理。';
      notes(true);
    } catch (exc) { error(exc.message); }
  }
  function renderRow(row) {
    const list = $('speechCaptions');
    if (row.state === 'discarded') {
      state.rows.delete(row.id); state.nodes.get(row.id)?.remove(); state.nodes.delete(row.id);
      return;
    }
    const follow = list.scrollHeight - list.scrollTop - list.clientHeight < 90;
    if (!state.rows.size) list.replaceChildren();
    state.rows.set(row.id, row);
    let node = state.nodes.get(row.id);
    if (!node) {
      node = document.createElement('article'); node.className = 'speech-caption';
      const meta = document.createElement('div'); meta.className = 'speech-caption-meta';
      const body = document.createElement('div'); body.className = 'speech-caption-text';
      body.append(document.createElement('p'), document.createElement('p')); node.append(meta, body);
      state.nodes.set(row.id, node); list.append(node);
    }
    node.dataset.state = row.state;
    const progress = row.state === 'recognizing'
      ? ` · 流式识别中${row.translation_in_progress ? ' · 翻译更新中' : row.error ? ' · 临时翻译失败，等待后续更新' : row.translation ? row.translation_pending ? ' · 临时译文，新内容待翻译' : ' · 临时译文' : ' · 等待翻译间隔'}`
      : row.state === 'translating' ? ' · 确认最终译文中' : '';
    node.firstChild.textContent = `${timestamp(row.start)} — ${timestamp(row.end)} · ${row.language || 'unknown'}${progress}${row.translation_latency_seconds !== undefined ? ` · 段尾到译文 ${row.translation_latency_seconds}s` : ''}`;
    node.lastChild.children[0].textContent = row.text;
    node.lastChild.children[1].textContent = row.translation || (row.error ? `翻译失败：${row.error}` : row.translation_in_progress || row.state !== 'recognizing' ? '翻译中…' : '等待翻译间隔…');
    if (row.translation && row.state === 'translation_error') node.lastChild.children[1].textContent += `\n\n当前为临时译文，最终翻译失败：${row.error}`;
    node.lastChild.children[1].title = [...(row.warnings || []), ...(row.error ? [row.error] : [])].join('\n');
    while (state.rows.size > 2000) {
      const oldest = state.rows.keys().next().value;
      state.rows.delete(oldest); state.nodes.get(oldest)?.remove(); state.nodes.delete(oldest);
    }
    if (follow) list.scrollTop = list.scrollHeight;
  }
  async function stopHardware() {
    if (state.node) { state.node.disconnect(); state.node.port.close(); state.node = null; }
    if (state.audio) { const audio = state.audio; state.audio = null; await audio.close().catch(() => {}); }
    if (state.stream) { const stream = state.stream; state.stream = null; for (const track of stream.getTracks()) track.stop(); }
    state.captureStarted = false;
    state.flush = null;
  }
  function send(value) {
    if (state.socket?.readyState !== WebSocket.OPEN) throw new Error('语音连接已断开');
    if (state.socket.bufferedAmount > 2 * 1024 * 1024) throw new Error('网络发送积压超过内存上限，已停止采集');
    state.socket.send(typeof value === 'object' && !(value instanceof ArrayBuffer) ? JSON.stringify(value) : value);
  }
  async function startHardware() {
    if (!state.stream || state.captureStarted) return;
    state.captureStarted = true;
    const context = new AudioContext({sampleRate: 16000}); state.audio = context;
    if (context.sampleRate !== 16000) throw new Error('此浏览器不支持 16 kHz 音频采集');
    await context.audioWorklet.addModule('/playground/speech-capture-worklet.js');
    if (!state.busy || !state.stream) return;
    const node = new AudioWorkletNode(context, 'speech-capture'); state.node = node;
    node.port.onmessage = event => {
      if (event.data === 'flushed') { state.flush?.(); state.flush = null; return; }
      if (state.busy) {
        try { send(event.data); } catch (exc) { fail(exc.message); }
      }
    };
    const mute = context.createGain(); mute.gain.value = 0;
    context.createMediaStreamSource(state.stream).connect(node); node.connect(mute).connect(context.destination);
    await context.resume();
  }
  function snapshot(data) {
    if (data.id !== state.session) return;
    for (const event of data.events || []) renderRow(event.segment);
    $('speechStatus').textContent = (labels[data.status] || data.status) + (terminal.has(data.status) && !data.resources_released ? ' · 正在释放模型…' : '');
    $('speechMetrics').textContent = `${data.asr_mode === 'native_streaming' ? '原生流式' : '分段识别'} · 音频 ${data.input_seconds}s · 已处理 ${data.processed_seconds}s · ${data.segments} 段 · 积压 ${data.backlog_seconds}s`;
    if (data.empty_segments) $('speechMetrics').textContent += ` · 空结果 ${data.empty_segments} 段（可能为非语音或漏识别）`;
    if (data.error) error(data.error);
    if (data.history_truncated) error('部分早期字幕已超出服务端保留范围。');
    if (data.status === 'listening' && state.stream && !state.captureStarted) startHardware().catch(exc => fail(exc.message));
    if (terminal.has(data.status) && data.resources_released) {
      state.busy = false; state.key = null; state.file = null; state.finishing = false;
      clearInterval(state.heartbeat); state.heartbeat = null;
      stopHardware();
      const socket = state.socket; state.socket = null; socket?.close();
    }
    controls();
  }
  async function fail(message) {
    if (state.failing) return;
    state.failing = true;
    state.generation++;
    error(message);
    await stopHardware();
    if (state.session) {
      try { snapshot(await api(`/sessions/${state.session}`, {method: 'DELETE'})); }
      catch (_) { error(`${message}；服务端将通过断连超时释放资源。`); }
    }
    state.socket?.close(); state.socket = null; state.busy = false; state.file = null;
    clearInterval(state.heartbeat); state.heartbeat = null;
    state.key = null; state.failing = false; state.finishing = false; controls();
  }
  async function start() {
    error();
    const source = $('speechSource').value;
    const file = $('speechFile').files[0];
    if (source === 'file' && !file) return error('请选择视频或音频文件');
    if (source === 'file' && file.size > state.catalog.max_upload_bytes) return error('文件超过服务端配置的大小限制');
    state.busy = true; state.finishing = false; state.key = $('apiKey').value.trim(); state.session = null; controls();
    const generation = ++state.generation;
    const requestHeaders = headers();
    try {
      if (source === 'browser') {
        if (!navigator.mediaDevices?.getDisplayMedia) throw new Error('此浏览器或页面地址不支持音频共享；远程页面需要 HTTPS');
        // Called directly from the user's click: the browser owns explicit consent.
        const stream = await navigator.mediaDevices.getDisplayMedia({video: true, audio: true, systemAudio: 'include'});
        if (!state.busy || generation !== state.generation) { for (const track of stream.getTracks()) track.stop(); return; }
        state.stream = stream;
        if (!state.stream.getAudioTracks().length) throw new Error('未共享音频，请在浏览器选择框中勾选共享音频');
        for (const track of state.stream.getTracks()) track.addEventListener('ended', () => { if (state.busy && state.stream) finish(); }, {once: true});
      }
      const options = {asr: $('speechAsr').value, translator: $('speechTranslator').value,
        target_language: $('speechTarget').value.trim(), source, device: $('speechDevice').value,
        file_size: source === 'file' ? file.size : 0, segment_seconds: Number($('speechWindow').value),
        silence_threshold: Number($('speechSilence').value), asr_mode: $('speechMode').value,
        transcription_delay_ms: Number($('speechDelay').value), stream_context_seconds: Number($('speechContext').value),
        partial_translation_interval_seconds: Number($('speechTranslationInterval').value),
        speech_detection: $('speechDetection').checked};
      const data = await api('/sessions', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(options)});
      if (!state.busy || generation !== state.generation) {
        await fetch(`${base}/sessions/${data.id}`, {method: 'DELETE', headers: requestHeaders});
        return;
      }
      state.session = data.id; state.file = source === 'file' ? file : null; state.resultOptions = options;
      state.rows.clear(); state.nodes.clear(); $('speechCaptions').replaceChildren();
      const url = new URL(`${base}/sessions/${data.id}/stream`, location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(url); state.socket = socket;
      socket.onopen = () => socket.send(JSON.stringify({authorization: state.key ? `Bearer ${state.key}` : '', client: state.client}));
      socket.onmessage = async event => {
        try {
          const value = JSON.parse(event.data);
          if (value.type === 'ready') {
            send({type: 'start'});
            state.heartbeat = setInterval(() => { try { send({type: 'heartbeat'}); } catch (exc) { fail(exc.message); } }, 5000);
          } else if (value.type === 'snapshot') {
            snapshot(value);
            // Incoming activity also drives acknowledgements: background tabs
            // may throttle setInterval while still processing WebSocket events.
            if (state.busy && performance.now() - state.lastHeartbeat > 5000) {
              send({type: 'heartbeat'}); state.lastHeartbeat = performance.now();
            }
          }
          else if (value.type === 'error') await fail(value.error);
          else if (value.type === 'read') {
            if (!state.file || !Number.isSafeInteger(value.offset) || !Number.isSafeInteger(value.length) || value.offset < 0 || value.length > 1024 * 1024 || value.offset + value.length > state.file.size || !/^[0-9a-f]{16}$/.test(value.id)) throw new Error('服务端文件读取请求无效');
            const chunk = await state.file.slice(value.offset, value.offset + value.length).arrayBuffer();
            const frame = new Uint8Array(16 + chunk.byteLength);
            frame.set(new TextEncoder().encode(value.id)); frame.set(new Uint8Array(chunk), 16); send(frame.buffer);
          }
        } catch (exc) { await fail(exc.message); }
      };
      socket.onclose = () => { if (state.busy && state.socket === socket) fail('语音连接已断开，已停止采集'); };
      socket.onerror = () => { if (state.busy) error('无法建立语音连接'); };
      controls();
    } catch (exc) { await fail(exc.message); }
  }
  async function finish() {
    if (!state.busy || state.finishing) return;
    state.finishing = true;
    $('speechFinish').disabled = true;
    try {
      if (state.node) {
        await Promise.race([new Promise(resolve => {state.flush = resolve; state.node.port.postMessage('flush');}), new Promise(resolve => setTimeout(resolve, 1000))]);
      }
      await stopHardware();
      if (state.session) send({type: 'finish'});
    } catch (exc) { await fail(exc.message); }
  }
  function download(kind) {
    if (state.privacy) return;
    const rows = [...state.rows.values()];
    const text = kind === 'srt' ? subtitles(rows) : JSON.stringify({...state.resultOptions, segments: rows}, null, 2);
    const url = URL.createObjectURL(new Blob([text], {type: 'text/plain;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = `speech-subtitles.${kind}`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  document.addEventListener('playground-function', event => { if (event.detail.id === 'speech' && !state.busy) refresh(); });
  document.addEventListener('playground-privacy', event => {state.privacy = Boolean(event.detail.enabled); controls();});
  $('speechAsr').addEventListener('change', () => notes(true));
  for (const id of ['speechTranslator', 'speechMode', 'speechSource']) $(id).addEventListener('change', () => notes());
  $('speechDevices').addEventListener('click', async () => {
    try { const data = await api('/devices'); $('speechDevice').replaceChildren(...data.devices.map(x => new Option(x.label, String(x.id), x.default, x.default))); }
    catch (exc) { error(exc.message); }
  });
  $('speechStart').addEventListener('click', start);
  $('speechFinish').addEventListener('click', finish);
  $('speechCancel').addEventListener('click', () => fail('已立即停止'));
  $('speechExportSrt').addEventListener('click', () => download('srt'));
  $('speechExportJson').addEventListener('click', () => download('json'));
  window.addEventListener('pagehide', () => {
    stopHardware(); state.socket?.close();
    if (state.busy && state.session) fetch(`${base}/sessions/${state.session}`, {method: 'DELETE', headers: headers(), keepalive: true}).catch(() => {});
  });
})();
