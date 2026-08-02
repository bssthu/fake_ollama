(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const els = {
    apiKey: $('apiKey'),
    toggleKey: $('toggleKey'),
    model: $('model'),
    loadModels: $('loadModels'),
    connectionStatus: $('connectionStatus'),
    capabilityList: $('capabilityList'),
    operation: $('operation'),
    operationHint: $('operationHint'),
    uploadSection: $('uploadSection'),
    uploadRequirement: $('uploadRequirement'),
    fileInput: $('fileInput'),
    dropZone: $('dropZone'),
    attachmentList: $('attachmentList'),
    chatParams: $('chatParams'),
    mediaParams: $('mediaParams'),
    denoiseField: $('denoiseField'),
    videoParams: $('videoParams'),
    systemPrompt: $('systemPrompt'),
    temperature: $('temperature'),
    maxTokens: $('maxTokens'),
    mediaSize: $('mediaSize'),
    mediaCount: $('mediaCount'),
    mediaSeed: $('mediaSeed'),
    mediaSteps: $('mediaSteps'),
    mediaCfg: $('mediaCfg'),
    mediaDenoise: $('mediaDenoise'),
    videoFrames: $('videoFrames'),
    videoFps: $('videoFps'),
    videoPrefetch: $('videoPrefetch'),
    resultTitle: $('resultTitle'),
    modelChip: $('modelChip'),
    modelVram: $('modelVram'),
    ttftMetric: $('ttftMetric'),
    timeMetric: $('timeMetric'),
    emptyState: $('emptyState'),
    errorBox: $('errorBox'),
    reasoning: $('reasoning'),
    reasoningText: $('reasoningText'),
    answer: $('answer'),
    answerText: $('answerText'),
    mediaGallery: $('mediaGallery'),
    prompt: $('prompt'),
    clear: $('clear'),
    stop: $('stop'),
    send: $('send'),
  };

  const OP_LABELS = {
    chat: '聊天 / 文本补全',
    image_generation: '图片生成',
    image_edit: '图片编辑',
    video_generation: '视频生成',
  };
  const KNOWN_CAPABILITIES = new Set([
    'completion', 'tools', 'vision', 'image_generation', 'image_edit',
    'video_generation',
  ]);

  const state = {
    models: new Map(),
    attachments: [],
    controller: null,
    startedAt: 0,
    firstTokenAt: 0,
    timer: null,
  };

  function authHeaders(includeJson = false) {
    const headers = {};
    const key = els.apiKey.value.trim();
    if (key) headers.Authorization = `Bearer ${key}`;
    if (includeJson) headers['Content-Type'] = 'application/json';
    return headers;
  }

  function setConnection(text, kind = '') {
    els.connectionStatus.textContent = text;
    els.connectionStatus.className = `status-dot ${kind}`.trim();
  }

  function formatError(raw, fallback) {
    if (!raw) return fallback;
    if (typeof raw === 'string') return raw;
    if (typeof raw.detail === 'string') return raw.detail;
    if (typeof raw.error === 'string') return raw.error;
    if (raw.error && typeof raw.error.message === 'string') return raw.error.message;
    try { return JSON.stringify(raw, null, 2); } catch (_) { return fallback; }
  }

  async function responseError(response) {
    const text = await response.text();
    try {
      return formatError(JSON.parse(text), `${response.status} ${response.statusText}`);
    } catch (_) {
      return text || `${response.status} ${response.statusText}`;
    }
  }

  function selectedModel() {
    return state.models.get(els.model.value) || null;
  }

  function legacyOperations(modelInfo) {
    const capabilities = new Set(modelInfo.capabilities || []);
    const operations = [];
    if (['completion', 'tools', 'vision'].some((cap) => capabilities.has(cap))) {
      operations.push({
        id: 'chat',
        endpoint: '/v1/chat/completions',
        stream: true,
        accepts_images: capabilities.has('vision'),
        tool_calling: capabilities.has('tools'),
      });
    }
    if (capabilities.has('image_generation')) {
      operations.push({
        id: 'image_generation', endpoint: '/v1/images/generations',
        stream: false, accepts_images: false,
      });
    }
    if (capabilities.has('image_edit')) {
      operations.push({
        id: 'image_edit', endpoint: '/v1/images/edits', stream: false,
        accepts_images: true, requires_images: true, multiple_images: true,
      });
    }
    if (capabilities.has('video_generation')) {
      operations.push({
        id: 'video_generation', endpoint: '/v1/videos/generations', stream: false,
        accepts_images: true, requires_images: false, multiple_images: true,
      });
    }
    return operations;
  }

  function modelOperations(modelInfo) {
    if (!modelInfo) return [];
    return Array.isArray(modelInfo.operations) && modelInfo.operations.length
      ? modelInfo.operations
      : legacyOperations(modelInfo);
  }

  function selectedOperation() {
    const info = selectedModel();
    return modelOperations(info).find((op) => op.id === els.operation.value) || null;
  }

  function renderCapabilities(info) {
    els.capabilityList.replaceChildren();
    for (const capability of (info && info.capabilities) || []) {
      const chip = document.createElement('span');
      chip.className = 'capability-chip';
      if (!KNOWN_CAPABILITIES.has(capability)) chip.classList.add('unknown');
      chip.textContent = capability;
      els.capabilityList.append(chip);
    }
  }

  function syncModel() {
    const info = selectedModel();
    const previous = els.operation.value;
    const operations = modelOperations(info);
    renderCapabilities(info);
    els.operation.replaceChildren();
    if (!operations.length) {
      els.operation.append(new Option('当前 Playground 尚未适配这些能力', ''));
      els.operation.disabled = true;
    } else {
      for (const op of operations) {
        const label = OP_LABELS[op.id] || op.id;
        els.operation.append(new Option(label, op.id));
      }
      els.operation.disabled = false;
      if (operations.some((op) => op.id === previous)) els.operation.value = previous;
    }
    els.modelChip.textContent = info ? info.id : '未选择模型';
    const estimatedVram = info && Number(info.estimated_vram_gb);
    if (Number.isFinite(estimatedVram) && estimatedVram > 0) {
      els.modelVram.textContent = `预计显存 ${estimatedVram.toFixed(2)} GiB`;
      els.modelVram.title = '来自 model_profiles.estimated_vram_gb';
      els.modelVram.hidden = false;
    } else {
      els.modelVram.textContent = '';
      els.modelVram.hidden = true;
    }
    syncOperation();
  }

  function syncOperation() {
    const op = selectedOperation();
    const isChat = op && op.id === 'chat';
    const isImageEdit = op && op.id === 'image_edit';
    const isVideo = op && op.id === 'video_generation';
    const isMedia = op && !isChat;

    els.uploadSection.hidden = !(op && op.accepts_images);
    els.uploadRequirement.textContent = op && op.requires_images ? '至少需要 1 张' : '可选';
    els.chatParams.hidden = !isChat;
    els.mediaParams.hidden = !isMedia;
    els.denoiseField.hidden = !isImageEdit;
    els.videoParams.hidden = !isVideo;
    els.ttftMetric.hidden = !isChat;
    els.resultTitle.textContent = isChat ? '流式输出' : '生成结果';

    if (!op) {
      const info = selectedModel();
      const caps = info && info.capabilities ? info.capabilities.join(', ') : '无';
      els.operationHint.textContent = `已声明能力：${caps}。当前没有可执行的已适配操作。`;
      els.prompt.placeholder = '该模型没有可执行的 Playground 操作';
      els.send.textContent = '不可执行';
    } else {
      const imageHint = op.accepts_images
        ? (op.requires_images ? '，需要图片输入' : '，可附带图片输入')
        : '';
      const limits = op.limits || {};
      const limitHints = [];
      if (limits.max_batch_size) limitHints.push(`最多 ${limits.max_batch_size} 个结果`);
      if (limits.max_reference_images) limitHints.push(`最多 ${limits.max_reference_images} 张参考图`);
      if (limits.max_num_frames) limitHints.push(`最多 ${limits.max_num_frames} 帧`);
      els.operationHint.textContent = [
        `${op.endpoint}${op.stream ? ' · 流式' : ''}${imageHint}`,
        ...limitHints,
      ].join(' · ');
      els.prompt.placeholder = isChat
        ? (op.accepts_images ? '输入问题；也可以粘贴或上传图片…' : '输入要发送给模型的内容…')
        : (isVideo ? '描述要生成的视频…' : (isImageEdit ? '描述希望怎样编辑图片…' : '描述要生成的图片…'));
      els.send.textContent = isChat ? '发送请求 →' : (isVideo ? '生成视频 →' : (isImageEdit ? '编辑图片 →' : '生成图片 →'));
    }
    applyOperationDefaults(op);
    syncSendState();
  }

  function applyOperationDefaults(op) {
    if (!op || op.id === 'chat') return;
    const defaults = op.defaults || {};
    els.mediaSize.value = defaults.size == null ? '' : String(defaults.size);
    els.mediaCount.value = defaults.n == null ? '1' : String(defaults.n);
    els.mediaSeed.value = defaults.seed == null ? '' : String(defaults.seed);
    els.mediaSteps.value = defaults.steps == null ? '' : String(defaults.steps);
    els.mediaCfg.value = defaults.cfg == null ? '' : String(defaults.cfg);
    els.mediaDenoise.value = defaults.denoise == null ? '' : String(defaults.denoise);
    els.videoFrames.value = defaults.num_frames == null ? '' : String(defaults.num_frames);
    els.videoFps.value = defaults.fps == null ? '' : String(defaults.fps);
    els.videoPrefetch.value = defaults.prefetch_count == null
      ? '' : String(defaults.prefetch_count);
  }

  function syncSendState() {
    const op = selectedOperation();
    const missingRequiredImage = Boolean(op && op.requires_images && !state.attachments.length);
    els.send.disabled = Boolean(state.controller) || !selectedModel() || !op
      || !els.prompt.value.trim() || missingRequiredImage;
  }

  function showError(message) {
    els.errorBox.textContent = message;
    els.errorBox.classList.add('visible');
    els.emptyState.hidden = true;
  }

  function showRequestError(op, message) {
    // Media requests only contain a temporary progress label until a complete
    // response arrives.  Do not leave that label looking like active work once
    // the request has already failed.  Keep partial streamed chat output when
    // there is any, because it can still be useful while debugging.
    if (op.id !== 'chat' || !els.answerText.textContent.trim()) {
      els.answerText.textContent = '';
      els.answer.classList.remove('active');
    }
    els.answerText.classList.remove('cursor');
    showError(message);
  }

  function resetOutput() {
    els.errorBox.textContent = '';
    els.errorBox.classList.remove('visible');
    els.reasoningText.textContent = '';
    els.reasoning.classList.remove('visible');
    els.reasoning.open = false;
    els.answerText.textContent = '';
    els.answer.classList.remove('active');
    els.answerText.classList.remove('cursor');
    els.mediaGallery.replaceChildren();
    els.mediaGallery.classList.remove('visible');
    els.emptyState.hidden = false;
    els.ttftMetric.textContent = '首字 —';
    els.timeMetric.textContent = '耗时 —';
  }

  function beginRequest(op) {
    resetOutput();
    els.emptyState.hidden = true;
    els.answer.classList.add('active');
    if (op.id === 'chat') {
      els.answerText.classList.add('cursor');
    } else {
      els.answerText.textContent = op.id === 'video_generation' ? '正在生成视频…' : '正在生成图片…';
    }
    els.send.hidden = true;
    els.stop.hidden = false;
  }

  function finishRequest() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null;
    els.answerText.classList.remove('cursor');
    state.controller = null;
    els.stop.hidden = true;
    els.send.hidden = false;
    syncSendState();
  }

  function textFromContent(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content.map((part) => {
      if (typeof part === 'string') return part;
      if (!part || typeof part !== 'object') return '';
      return part.text || part.content || '';
    }).join('');
  }

  function noteFirstToken() {
    if (state.firstTokenAt) return;
    state.firstTokenAt = performance.now();
    els.ttftMetric.textContent = `首字 ${((state.firstTokenAt - state.startedAt) / 1000).toFixed(2)}s`;
  }

  function consumeChunk(chunk) {
    if (chunk && chunk.error) throw new Error(formatError(chunk.error, '流式响应错误'));
    const choice = chunk && Array.isArray(chunk.choices) ? chunk.choices[0] : null;
    if (!choice) return;
    const delta = choice.delta || {};
    const thought = textFromContent(
      delta.reasoning_content ?? delta.reasoning ?? delta.thinking ?? ''
    );
    const content = textFromContent(delta.content ?? '');
    if (thought) {
      noteFirstToken();
      els.reasoning.classList.add('visible');
      els.reasoningText.textContent += thought;
    }
    if (content) {
      noteFirstToken();
      els.answerText.textContent += content;
    }
    if (delta.tool_calls) {
      noteFirstToken();
      els.answerText.textContent += `\n\n[tool_calls]\n${JSON.stringify(delta.tool_calls, null, 2)}`;
    }
  }

  function parseSseLine(line) {
    const clean = line.endsWith('\r') ? line.slice(0, -1) : line;
    if (!clean.startsWith('data:')) return false;
    const data = clean.slice(5).trimStart();
    if (!data) return false;
    if (data === '[DONE]') return true;
    consumeChunk(JSON.parse(data));
    return false;
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(reader.error || new Error('无法读取图片'));
      reader.readAsDataURL(file);
    });
  }

  async function addFiles(files) {
    const images = Array.from(files || []).filter((file) => file && file.type.startsWith('image/'));
    if (!images.length) return;
    const op = selectedOperation();
    const configuredLimit = op && op.limits && op.limits.max_reference_images;
    const maxImages = configuredLimit || 12;
    for (const file of images) {
      if (state.attachments.length >= maxImages) {
        showError(`当前模型一次最多添加 ${maxImages} 张参考图片。`);
        break;
      }
      const dataUrl = await readFileAsDataUrl(file);
      state.attachments.push({
        file,
        dataUrl,
        name: file.name || `pasted-${state.attachments.length + 1}.png`,
      });
    }
    renderAttachments();
    syncSendState();
  }

  function renderAttachments() {
    els.attachmentList.replaceChildren();
    state.attachments.forEach((attachment, index) => {
      const wrap = document.createElement('div');
      wrap.className = 'attachment';
      wrap.title = attachment.name;
      const image = document.createElement('img');
      image.src = attachment.dataUrl;
      image.alt = attachment.name;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '×';
      remove.setAttribute('aria-label', `移除 ${attachment.name}`);
      remove.addEventListener('click', () => {
        state.attachments.splice(index, 1);
        renderAttachments();
        syncSendState();
      });
      wrap.append(image, remove);
      els.attachmentList.append(wrap);
    });
  }

  function clearAttachments() {
    state.attachments = [];
    els.fileInput.value = '';
    renderAttachments();
  }

  async function loadModelList() {
    if (state.controller) return;
    els.loadModels.disabled = true;
    els.model.disabled = true;
    setConnection('连接中…');
    try {
      const response = await fetch('/v1/models', {
        headers: authHeaders(false),
        cache: 'no-store',
      });
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      const models = Array.isArray(data.data)
        ? data.data.filter((item) => item && item.id)
        : [];
      state.models.clear();
      els.model.replaceChildren();
      if (!models.length) {
        els.model.append(new Option('当前接口没有可用模型', ''));
        setConnection('无模型', 'error');
      } else {
        for (const info of models) {
          state.models.set(info.id, info);
          els.model.append(new Option(info.id, info.id));
        }
        els.model.disabled = false;
        setConnection(`${models.length} 个模型`, 'ok');
      }
      syncModel();
    } catch (error) {
      state.models.clear();
      els.model.replaceChildren(new Option('模型加载失败', ''));
      setConnection('连接失败', 'error');
      showError(error instanceof Error ? error.message : String(error));
      syncModel();
    } finally {
      els.loadModels.disabled = false;
    }
  }

  async function runChat(op) {
    const messages = [];
    if (els.systemPrompt.value.trim()) {
      messages.push({role: 'system', content: els.systemPrompt.value.trim()});
    }
    let userContent = els.prompt.value.trim();
    if (op.accepts_images && state.attachments.length) {
      userContent = [{type: 'text', text: userContent}];
      for (const attachment of state.attachments) {
        userContent.push({type: 'image_url', image_url: {url: attachment.dataUrl}});
      }
    }
    messages.push({role: 'user', content: userContent});
    const body = {model: els.model.value, messages, stream: true};
    const temp = Number(els.temperature.value);
    const limit = Number.parseInt(els.maxTokens.value, 10);
    if (Number.isFinite(temp)) body.temperature = temp;
    if (Number.isFinite(limit) && limit > 0) body.max_tokens = limit;

    const response = await fetch(op.endpoint, {
      method: 'POST',
      headers: authHeaders(true),
      body: JSON.stringify(body),
      signal: state.controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response));
    if (!response.body) throw new Error('浏览器没有提供可读取的响应流');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let doneEvent = false;
    while (!doneEvent) {
      const result = await reader.read();
      buffer += decoder.decode(result.value || new Uint8Array(), {stream: !result.done});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (parseSseLine(line)) {
          doneEvent = true;
          break;
        }
      }
      if (result.done) break;
    }
    if (buffer.trim() && !doneEvent) parseSseLine(buffer);
    if (!els.answerText.textContent && !els.reasoningText.textContent) {
      els.answerText.textContent = '（模型返回了空内容）';
    }
  }

  function optionalNumber(input, integer = false) {
    if (!input.value.trim()) return null;
    const value = integer ? Number.parseInt(input.value, 10) : Number(input.value);
    return Number.isFinite(value) ? value : null;
  }

  function mediaPayload(op) {
    const payload = {
      model: els.model.value,
      prompt: els.prompt.value.trim(),
      response_format: 'b64_json',
      n: optionalNumber(els.mediaCount, true) || 1,
    };
    if (els.mediaSize.value.trim()) payload.size = els.mediaSize.value.trim();
    const optional = [
      ['seed', optionalNumber(els.mediaSeed, true)],
      ['steps', optionalNumber(els.mediaSteps, true)],
      ['cfg', optionalNumber(els.mediaCfg, false)],
    ];
    if (op.id === 'image_edit') {
      optional.push(['denoise', optionalNumber(els.mediaDenoise, false)]);
    }
    if (op.id === 'video_generation') {
      optional.push(
        ['num_frames', optionalNumber(els.videoFrames, true)],
        ['fps', optionalNumber(els.videoFps, false)],
        ['prefetch_count', optionalNumber(els.videoPrefetch, true)],
      );
    }
    for (const [key, value] of optional) {
      if (value !== null) payload[key] = value;
    }
    return payload;
  }

  function renderMedia(items, op) {
    els.answer.classList.remove('active');
    els.answerText.textContent = '';
    els.mediaGallery.replaceChildren();
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index] || {};
      const mime = item.mime_type || (op.id === 'video_generation' ? 'video/mp4' : 'image/png');
      const src = item.url || (item.b64_json ? `data:${mime};base64,${item.b64_json}` : '');
      if (!src) continue;
      const card = document.createElement('article');
      card.className = 'media-card';
      let media;
      if (mime.startsWith('video/')) {
        media = document.createElement('video');
        media.controls = true;
        media.preload = 'metadata';
      } else {
        media = document.createElement('img');
        media.alt = item.filename || `生成结果 ${index + 1}`;
      }
      media.src = src;
      const meta = document.createElement('div');
      meta.className = 'media-meta';
      const name = document.createElement('span');
      name.textContent = item.filename || `${OP_LABELS[op.id] || '结果'} ${index + 1}`;
      const download = document.createElement('a');
      download.href = src;
      download.download = item.filename || `result-${index + 1}`;
      download.textContent = '下载';
      meta.append(name, download);
      card.append(media, meta);
      els.mediaGallery.append(card);
    }
    if (!els.mediaGallery.children.length) throw new Error('媒体接口没有返回可显示的结果');
    els.mediaGallery.classList.add('visible');
  }

  async function runMedia(op) {
    const payload = mediaPayload(op);
    let response;
    if (op.id === 'image_generation') {
      response = await fetch(op.endpoint, {
        method: 'POST',
        headers: authHeaders(true),
        body: JSON.stringify(payload),
        signal: state.controller.signal,
      });
    } else {
      const form = new FormData();
      for (const [key, value] of Object.entries(payload)) form.append(key, String(value));
      for (const attachment of state.attachments) {
        form.append('image[]', attachment.file, attachment.name);
      }
      response = await fetch(op.endpoint, {
        method: 'POST',
        headers: authHeaders(false),
        body: form,
        signal: state.controller.signal,
      });
    }
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    renderMedia(Array.isArray(data.data) ? data.data : [], op);
  }

  async function runRequest() {
    const op = selectedOperation();
    if (state.controller || !op || !els.prompt.value.trim()) return;
    if (op.requires_images && !state.attachments.length) {
      showError('当前能力至少需要一张参考图片。');
      return;
    }

    state.controller = new AbortController();
    state.startedAt = performance.now();
    state.firstTokenAt = 0;
    beginRequest(op);
    syncSendState();
    state.timer = window.setInterval(() => {
      els.timeMetric.textContent = `耗时 ${((performance.now() - state.startedAt) / 1000).toFixed(1)}s`;
    }, 100);
    try {
      if (op.id === 'chat') await runChat(op);
      else await runMedia(op);
    } catch (error) {
      if (error && error.name === 'AbortError') showRequestError(op, '请求已停止。');
      else showRequestError(op, error instanceof Error ? error.message : String(error));
    } finally {
      els.timeMetric.textContent = `耗时 ${((performance.now() - state.startedAt) / 1000).toFixed(2)}s`;
      finishRequest();
    }
  }

  els.toggleKey.addEventListener('click', () => {
    const showing = els.apiKey.type === 'text';
    els.apiKey.type = showing ? 'password' : 'text';
    els.toggleKey.setAttribute('aria-label', showing ? '显示 API key' : '隐藏 API key');
  });
  els.apiKey.addEventListener('input', () => setConnection('凭据已变化'));
  els.loadModels.addEventListener('click', loadModelList);
  els.model.addEventListener('change', syncModel);
  els.operation.addEventListener('change', syncOperation);
  els.prompt.addEventListener('input', syncSendState);
  els.prompt.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      runRequest();
    }
  });

  els.fileInput.addEventListener('change', () => addFiles(els.fileInput.files));
  els.dropZone.addEventListener('click', () => els.fileInput.click());
  for (const eventName of ['dragenter', 'dragover']) {
    els.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropZone.classList.add('dragover');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    els.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropZone.classList.remove('dragover');
    });
  }
  els.dropZone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));
  document.addEventListener('paste', (event) => {
    const op = selectedOperation();
    if (!op || !op.accepts_images || !event.clipboardData) return;
    const files = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
      .map((item) => item.getAsFile())
      .filter(Boolean);
    if (files.length) {
      event.preventDefault();
      addFiles(files);
    }
  });

  els.send.addEventListener('click', runRequest);
  els.stop.addEventListener('click', () => state.controller && state.controller.abort());
  els.clear.addEventListener('click', () => {
    if (state.controller) state.controller.abort();
    els.prompt.value = '';
    clearAttachments();
    resetOutput();
    syncSendState();
    els.prompt.focus();
  });

  syncModel();
})();
