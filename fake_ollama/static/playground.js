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
    uploadLabel: $('uploadLabel'),
    uploadRequirement: $('uploadRequirement'),
    fileInput: $('fileInput'),
    dropZone: $('dropZone'),
    dropText: $('dropText'),
    dropHint: $('dropHint'),
    attachmentList: $('attachmentList'),
    composerAttachmentSummary: $('composerAttachmentSummary'),
    cameraSection: $('cameraSection'),
    cameraStatus: $('cameraStatus'),
    cameraPreview: $('cameraPreview'),
    cameraPlaceholder: $('cameraPlaceholder'),
    cameraFacing: $('cameraFacing'),
    cameraStart: $('cameraStart'),
    cameraStop: $('cameraStop'),
    cameraStats: $('cameraStats'),
    cameraLiveText: $('cameraLiveText'),
    mediaQueueSection: $('mediaQueueSection'),
    generationCount: $('generationCount'),
    decisionSection: $('decisionSection'),
    decisionDialect: $('decisionDialect'),
    decisionDialectHint: $('decisionDialectHint'),
    decisionStateFormat: $('decisionStateFormat'),
    decisionStateLabel: $('decisionStateLabel'),
    decisionCapabilityHint: $('decisionCapabilityHint'),
    decisionQuestionList: $('decisionQuestionList'),
    decisionAddQuestion: $('decisionAddQuestion'),
    decisionThinkField: $('decisionThinkField'),
    decisionThinkMode: $('decisionThinkMode'),
    decisionRequestPreview: $('decisionRequestPreview'),
    chatParams: $('chatParams'),
    mediaParams: $('mediaParams'),
    operationPresetField: $('operationPresetField'),
    operationPreset: $('operationPreset'),
    operationPresetHint: $('operationPresetHint'),
    operationParameterList: $('operationParameterList'),
    externalPlanner: $('externalPlanner'),
    externalPlannerStatus: $('externalPlannerStatus'),
    externalPlannerProtocol: $('externalPlannerProtocol'),
    externalPlannerCapability: $('externalPlannerCapability'),
    externalPlannerUrl: $('externalPlannerUrl'),
    externalPlannerToken: $('externalPlannerToken'),
    toggleExternalPlannerToken: $('toggleExternalPlannerToken'),
    externalPlannerModel: $('externalPlannerModel'),
    externalPlannerModelList: $('externalPlannerModelList'),
    detectExternalPlannerModels: $('detectExternalPlannerModels'),
    systemPrompt: $('systemPrompt'),
    temperature: $('temperature'),
    maxTokens: $('maxTokens'),
    resultTitle: $('resultTitle'),
    modelChip: $('modelChip'),
    modelVram: $('modelVram'),
    modelMemory: $('modelMemory'),
    contextChip: $('contextChip'),
    historyModeChip: $('historyModeChip'),
    contextMetric: $('contextMetric'),
    ttftMetric: $('ttftMetric'),
    timeMetric: $('timeMetric'),
    emptyState: $('emptyState'),
    emptyHint: $('emptyHint'),
    contextNotice: $('contextNotice'),
    errorBox: $('errorBox'),
    conversation: $('conversation'),
    reasoning: $('reasoning'),
    reasoningText: $('reasoningText'),
    answer: $('answer'),
    answerText: $('answerText'),
    prompt: $('prompt'),
    shortcut: $('shortcut'),
    clear: $('clear'),
    stop: $('stop'),
    send: $('send'),
  };

  const OP_LABELS = {
    ocr_recognize: 'OCR 识别 / 可选复核',
    ocr_auto: 'OCR → 复核 → 低分重识别',
    ocr_compare: 'OCR 多模型对比',
    ocr_score: '候选复核',
    chat: '聊天 / 文本补全',
    decision: '结构化决策 / Decision',
    image_generation: '图片生成',
    image_edit: '图片编辑',
    video_generation: '视频生成',
    video_analysis: '视频分析',
    h3_context_ir: 'H3 Prompt 自动增强',
  };
  const KNOWN_CAPABILITIES = new Set([
    'completion', 'tools', 'vision', 'image_generation', 'image_edit',
    'video_generation', 'video_understanding', 'h3_context_ir', 'decision', 'ocr',
  ]);
  const CONTEXT_THRESHOLD_RATIO = 0.9;
  const DISCOVERY_SCHEMA_VERSION = 1;
  const IMAGE_TOKEN_ESTIMATE = 1024;
  const VIDEO_TOKEN_ESTIMATE = 8192;
  const LIVE_CAMERA_HISTORY_LIMIT = 200;
  const LIVE_CAMERA_VIDEO_BITS_PER_SECOND = 2_500_000;
  const MAX_MEDIA_QUEUE_RUNS = 20;
  const textEncoder = new TextEncoder();

  const state = {
    models: new Map(),
    interactionHistories: new Map(),
    decisionForms: new Map(),
    activeDecisionModelId: null,
    loadedCredential: null,
    ocrCatalog: null,
    ocrRuntime: null,
    ocrReleasing: false,
    attachments: [],
    controller: null,
    requestDiscarded: false,
    startedAt: 0,
    firstTokenAt: 0,
    timer: null,
    parameterInputs: new Map(),
    activePlannerChoice: null,
    activePlannerHasImages: false,
    externalPlannerDetecting: false,
    camera: {
      starting: false,
      active: false,
      processing: false,
      stream: null,
      recorder: null,
      stopTimer: null,
      uiTimer: null,
      captureTask: null,
      pending: null,
      sequence: 0,
      analyzed: 0,
      dropped: 0,
      errors: 0,
      consecutiveErrors: 0,
      lastError: '',
      startedAt: 0,
      settings: null,
    },
  };

  function authHeaders(includeJson = false) {
    const headers = {};
    const key = els.apiKey.value.trim();
    if (key) headers.Authorization = `Bearer ${key}`;
    if (includeJson) headers['Content-Type'] = 'application/json';
    return headers;
  }

  function plannerRequestHeaders(plan, includeJson = false) {
    const headers = authHeaders(includeJson);
    if (plan && plan.externalPlannerToken) {
      headers['X-Playground-Upstream-Key'] = plan.externalPlannerToken;
    }
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

  function operationUsesMediaQueue(op = selectedOperation()) {
    return Boolean(op && [
      'image_generation', 'image_edit', 'video_generation',
    ].includes(op.id));
  }

  function mediaQueueRunCount() {
    const value = Number.parseInt(els.generationCount.value, 10);
    if (!Number.isInteger(value) || value < 1 || value > MAX_MEDIA_QUEUE_RUNS) {
      throw new Error(`生成份数必须是 1 到 ${MAX_MEDIA_QUEUE_RUNS} 的整数。`);
    }
    return value;
  }

  function legacyOperations(modelInfo) {
    const capabilities = new Set(modelInfo.capabilities || []);
    const operations = [];
    if (['completion', 'tools', 'vision'].some((cap) => capabilities.has(cap))) {
      operations.push({
        id: 'chat',
        endpoint: '/v1/chat/completions',
        stream: true,
        history_mode: 'conversation',
        accepts_images: capabilities.has('vision'),
        tool_calling: capabilities.has('tools'),
      });
    }
    if (capabilities.has('image_generation')) {
      operations.push({
        id: 'image_generation', endpoint: '/v1/images/generations',
        stream: false, history_mode: 'single_turn', accepts_images: false,
      });
    }
    if (capabilities.has('image_edit')) {
      operations.push({
        id: 'image_edit', endpoint: '/v1/images/edits', stream: false,
        history_mode: 'single_turn',
        accepts_images: true, requires_images: true, multiple_images: true,
      });
    }
    if (capabilities.has('video_generation')) {
      operations.push({
        id: 'video_generation', endpoint: '/v1/videos/generations', stream: false,
        history_mode: 'single_turn',
        accepts_images: true, requires_images: false, multiple_images: true,
      });
    }
    if (capabilities.has('video_understanding')) {
      operations.push({
        id: 'video_analysis', endpoint: '/v1/chat/completions', stream: true,
        history_mode: 'single_turn', accepts_videos: true, requires_videos: true,
        multiple_videos: false,
        limits: {max_videos: 1, max_video_bytes: 64 * 1024 * 1024},
        live_camera: {
          supported: true, capture_mode: 'windowed_media_recorder',
          max_pending_segments: 1,
        },
      });
    }
    return operations;
  }

  function operationUsesChatEndpoint(op) {
    return Boolean(op && ['chat', 'video_analysis'].includes(op.id));
  }

  function operationSupportsLiveCamera(op) {
    return Boolean(
      op && op.id === 'video_analysis'
      && (!op.live_camera || op.live_camera.supported !== false)
    );
  }

  function modelOperations(modelInfo) {
    if (!modelInfo) return [];
    return Array.isArray(modelInfo.operations) && modelInfo.operations.length
      ? modelInfo.operations
      : legacyOperations(modelInfo);
  }

  function selectedOperation() {
    const info = selectedModel();
    const op = modelOperations(info).find((item) => item.id === els.operation.value) || null;
    if (!op || op.id !== 'decision') return op;
    const form = state.decisionForms.get(els.model.value);
    const dialect = form ? form.dialect : op.dialect;
    const wire = decisionDialects(op).find((item) => item.id === dialect);
    const format = op.image_format || (op.dialect === 'openai_decisions' ? 'openai_input_image' : 'typesafe_assets');
    const compatibleFormat = dialect === 'systemone'
      ? ['typesafe_assets', 'startlux_images'].includes(format) : format === 'openai_input_image';
    return {...op, image_format: format, endpoint: wire ? wire.endpoint : op.endpoint,
      accepts_images: Boolean(op.accepts_images && dialect === op.image_dialect && compatibleFormat)};
  }

  function operationHistoryMode(op) {
    if (op && ['conversation', 'single_turn'].includes(op.history_mode)) {
      return op.history_mode;
    }
    return op && op.id === 'chat' ? 'conversation' : 'single_turn';
  }

  function operationUsesHistory(op) {
    return operationHistoryMode(op) === 'conversation';
  }

  function interactionKey(modelId, operationId) {
    return JSON.stringify([modelId, operationId]);
  }

  function interactionFor(modelId, operationId, create = true) {
    if (!modelId || !operationId) return null;
    const key = interactionKey(modelId, operationId);
    let interaction = state.interactionHistories.get(key);
    if (!interaction && create) {
      interaction = {turns: [], lastNotice: ''};
      state.interactionHistories.set(key, interaction);
    }
    return interaction || null;
  }

  function modelContextLength(info = selectedModel()) {
    const value = info && Number(info.context_length);
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : null;
  }

  function requestedMaxTokens() {
    const value = Number.parseInt(els.maxTokens.value, 10);
    return Number.isFinite(value) && value > 0 ? value : 2048;
  }

  function outputTokenReserve(info = selectedModel()) {
    const requested = requestedMaxTokens();
    const configured = info && Number(info.max_output_tokens);
    return Number.isFinite(configured) && configured > 0
      ? Math.floor(configured)
      : requested;
  }

  function formatTokenCount(value) {
    const amount = Math.max(0, Math.round(Number(value) || 0));
    if (amount >= 1000000) return `${(amount / 1000000).toFixed(1)}M`;
    if (amount >= 1000) return `${(amount / 1000).toFixed(amount >= 10000 ? 0 : 1)}K`;
    return String(amount);
  }

  function estimateTextTokens(value) {
    const text = String(value || '');
    if (!text) return 0;
    const cjk = (text.match(/[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]/g) || []).length;
    const nonCjk = Math.max(0, text.length - cjk);
    return Math.max(
      1,
      Math.ceil(textEncoder.encode(text).length / 4),
      cjk + Math.ceil(nonCjk / 4),
    );
  }

  function estimateContentTokens(content) {
    if (typeof content === 'string') return estimateTextTokens(content);
    if (!Array.isArray(content)) return estimateTextTokens(JSON.stringify(content || ''));
    let total = 0;
    for (const part of content) {
      if (typeof part === 'string') total += estimateTextTokens(part);
      else if (part && part.type === 'image_url') total += IMAGE_TOKEN_ESTIMATE;
      else if (part && part.type === 'video_url') total += VIDEO_TOKEN_ESTIMATE;
      else if (part && typeof part === 'object') {
        total += estimateTextTokens(part.text ?? part.content ?? '');
      }
    }
    return total;
  }

  function estimateMessageTokens(message) {
    return 5 + estimateTextTokens(message.role || '') + estimateContentTokens(message.content);
  }

  function estimateMessagesTokens(messages) {
    return 3 + messages.reduce((total, message) => total + estimateMessageTokens(message), 0);
  }

  function imageUrlsFromContent(content) {
    if (!Array.isArray(content)) return [];
    return content.flatMap((part) => {
      if (!part || part.type !== 'image_url') return [];
      const imageUrl = part.image_url;
      const url = typeof imageUrl === 'string' ? imageUrl : imageUrl && imageUrl.url;
      return url ? [url] : [];
    });
  }

  function videoUrlsFromContent(content) {
    if (!Array.isArray(content)) return [];
    return content.flatMap((part) => {
      if (!part || part.type !== 'video_url') return [];
      const videoUrl = part.video_url;
      const url = typeof videoUrl === 'string' ? videoUrl : videoUrl && videoUrl.url;
      return url ? [url] : [];
    });
  }

  function userMessageForRequest(
    op,
    prompt = els.prompt.value.trim(),
    attachments = state.attachments,
  ) {
    let content = prompt;
    const accepted = attachments.filter((attachment) => (
      (attachment.kind === 'image' && op.accepts_images)
      || (attachment.kind === 'video' && op.accepts_videos)
    ));
    if (accepted.length) {
      content = [{type: 'text', text: content}];
      for (const attachment of accepted) {
        if (attachment.kind === 'video') {
          content.push({type: 'video_url', video_url: {url: attachment.dataUrl}});
        } else {
          content.push({type: 'image_url', image_url: {url: attachment.dataUrl}});
        }
      }
    }
    return {role: 'user', content};
  }

  function replaceContentText(content, text) {
    if (typeof content === 'string') return text;
    if (!Array.isArray(content)) return text;
    let replaced = false;
    const result = content.map((part) => {
      if (!replaced && part && part.type === 'text') {
        replaced = true;
        return {...part, text};
      }
      return part;
    });
    if (!replaced) result.unshift({type: 'text', text});
    return result;
  }

  function truncateTextToTokenBudget(text, tokenBudget) {
    const original = String(text || '');
    if (estimateTextTokens(original) <= tokenBudget) return original;
    const marker = '\n\n…[为适配模型上下文，已截断中间内容]…\n\n';
    if (tokenBudget <= estimateTextTokens(marker) + 2) return '';
    let low = 0;
    let high = original.length;
    let best = '';
    while (low <= high) {
      const keep = Math.floor((low + high) / 2);
      const head = Math.ceil(keep * 0.55);
      const tail = keep - head;
      const candidate = `${original.slice(0, head)}${marker}${tail ? original.slice(-tail) : ''}`;
      if (estimateTextTokens(candidate) <= tokenBudget) {
        best = candidate;
        low = keep + 1;
      } else {
        high = keep - 1;
      }
    }
    return best;
  }

  function historyMessages(turns) {
    const messages = [];
    for (const turn of turns) messages.push(turn.user, turn.assistant);
    return messages;
  }

  function baseSystemMessages() {
    const prompt = els.systemPrompt.value.trim();
    return prompt ? [{role: 'system', content: prompt}] : [];
  }

  function appendMessageElement(parent, message, role, options = {}) {
    const article = document.createElement('article');
    article.className = `message ${role}${options.pending ? ' pending' : ''}`;
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = role === 'user' ? 'User' : 'Assistant';
    const content = document.createElement('pre');
    content.className = 'message-content';
    content.textContent = textFromContent(message.content) || '（无文本内容）';
    article.append(label, content);
    const imageUrls = imageUrlsFromContent(message.content);
    if (imageUrls.length) {
      const images = document.createElement('div');
      images.className = 'message-images';
      for (const url of imageUrls) {
        const image = document.createElement('img');
        image.src = url;
        image.alt = '对话参考图片';
        images.append(image);
      }
      article.append(images);
    }
    const videoUrls = videoUrlsFromContent(message.content);
    if (videoUrls.length) {
      const videos = document.createElement('div');
      videos.className = 'message-images';
      for (const url of videoUrls) {
        const video = document.createElement('video');
        video.src = url;
        video.controls = true;
        video.preload = 'metadata';
        videos.append(video);
      }
      article.append(videos);
    }
    if (role === 'assistant' && options.reasoning) {
      const details = document.createElement('details');
      details.className = 'message-reasoning';
      const summary = document.createElement('summary');
      summary.textContent = '模型思考过程';
      const reasoning = document.createElement('pre');
      reasoning.textContent = options.reasoning;
      details.append(summary, reasoning);
      article.append(details);
    }
    if (options.parameters && Object.keys(options.parameters).length) {
      const details = document.createElement('details');
      details.className = 'message-parameters';
      const summary = document.createElement('summary');
      summary.textContent = '请求参数';
      const parameters = document.createElement('pre');
      parameters.textContent = JSON.stringify(options.parameters, null, 2);
      details.append(summary, parameters);
      article.append(details);
    }
    parent.append(article);
  }

  function showContextNotice(message = '') {
    els.contextNotice.textContent = message;
    els.contextNotice.classList.toggle('visible', Boolean(message));
  }

  function renderInteractionHistory(pendingUser = null, pendingParameters = null) {
    els.conversation.replaceChildren();
    const op = selectedOperation();
    const interaction = interactionFor(els.model.value, op && op.id, false);
    for (const turn of (interaction && interaction.turns) || []) {
      const wrap = document.createElement('section');
      wrap.className = 'conversation-turn';
      appendMessageElement(wrap, turn.user, 'user', {parameters: turn.parameters});
      if (turn.assistant) {
        appendMessageElement(wrap, turn.assistant, 'assistant', {reasoning: turn.reasoning});
      } else if (turn.decision) {
        appendDecisionResultElement(wrap, turn.decision);
      } else if (turn.ocr) {
        appendOcrResult(wrap, turn.ocr);
      } else if (turn.media) {
        appendMediaResultElement(wrap, turn.media, op);
      }
      els.conversation.append(wrap);
    }
    if (pendingUser) {
      const wrap = document.createElement('section');
      wrap.className = 'conversation-turn pending-turn';
      appendMessageElement(wrap, pendingUser, 'user', {
        pending: true,
        parameters: pendingParameters,
      });
      els.conversation.append(wrap);
    }
    const notice = op && op.id === 'chat'
      ? ((interaction && interaction.lastNotice) || '')
      : plannerProviderWarning(op);
    showContextNotice(notice);
    els.emptyState.hidden = Boolean(els.conversation.children.length || pendingUser);
  }

  function updateContextPreview() {
    const op = selectedOperation();
    const info = selectedModel();
    const contextLength = modelContextLength(info);
    if (!op || !operationUsesChatEndpoint(op) || !contextLength) {
      els.contextMetric.hidden = true;
      els.contextMetric.className = 'metric';
      return;
    }
    const interaction = interactionFor(els.model.value, op.id, false);
    const priorTurns = operationUsesHistory(op)
      ? ((interaction && interaction.turns) || [])
      : [];
    const messages = [
      ...baseSystemMessages(),
      ...historyMessages(priorTurns),
    ];
    if (els.prompt.value.trim()) messages.push(userMessageForRequest(op));
    const inputTokens = estimateMessagesTokens(messages);
    const outputTokens = outputTokenReserve(info);
    const total = inputTokens + outputTokens;
    const threshold = Math.floor(contextLength * CONTEXT_THRESHOLD_RATIO);
    els.contextMetric.hidden = false;
    els.contextMetric.textContent = `预算 ~${formatTokenCount(total)} / ${formatTokenCount(contextLength)}`;
    els.contextMetric.title = `输入估算 ${inputTokens} + 输出预留 ${outputTokens}；发送阈值为上下文的 ${Math.round(CONTEXT_THRESHOLD_RATIO * 100)}%`;
    els.contextMetric.className = `metric${total > contextLength ? ' danger' : (total > threshold ? ' warning' : '')}`;
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

  function renderModelResourceChips(info, {planner = false, showContext = false} = {}) {
    const estimatedVram = info && Number(info.estimated_vram_gb);
    const backendKind = info && info.backend_kind;
    if (Number.isFinite(estimatedVram) && estimatedVram > 0) {
      els.modelVram.textContent = `预计显存 ${estimatedVram.toFixed(2)} GiB`;
      els.modelVram.title = planner
        ? '当前 Planner 对应 model_profiles.estimated_vram_gb'
        : '来自 model_profiles.estimated_vram_gb';
      els.modelVram.hidden = false;
    } else if (planner && backendKind === 'remote') {
      els.modelVram.textContent = 'Planner 远端 API';
      els.modelVram.title = '该 Planner 在远端运行，不占用本机模型显存';
      els.modelVram.hidden = false;
    } else if (planner) {
      els.modelVram.textContent = '预计显存 未配置';
      els.modelVram.title = '当前 Planner 没有 estimated_vram_gb 配置';
      els.modelVram.hidden = false;
    } else {
      els.modelVram.textContent = '';
      els.modelVram.hidden = true;
    }

    const estimatedMemory = info && Number(info.estimated_memory_gb);
    if (Number.isFinite(estimatedMemory) && estimatedMemory > 0) {
      els.modelMemory.textContent = `预计内存 ${estimatedMemory.toFixed(2)} GiB`;
      els.modelMemory.title = planner
        ? '当前 Planner 对应 model_profiles.estimated_memory_gb'
        : '来自 model_profiles.estimated_memory_gb';
      els.modelMemory.hidden = false;
    } else if (planner && backendKind !== 'remote') {
      els.modelMemory.textContent = '预计内存 未配置';
      els.modelMemory.title = '当前 Planner 没有 estimated_memory_gb 配置';
      els.modelMemory.hidden = false;
    } else {
      els.modelMemory.textContent = '';
      els.modelMemory.hidden = true;
    }

    const contextLength = modelContextLength(info);
    if (contextLength && showContext) {
      els.contextChip.textContent = `${planner ? 'Planner ' : ''}上下文 ${formatTokenCount(contextLength)}`;
      els.contextChip.title = `${planner ? '当前 Planner ' : '模型 '}context_length=${contextLength}`;
      els.contextChip.hidden = false;
    } else {
      els.contextChip.textContent = '';
      els.contextChip.hidden = true;
    }
  }

  function plannerParameterName(op) {
    if (!op) return '';
    if (op.id === 'h3_context_ir') return 'provider';
    return op.context_ir_profile ? 'context_ir_provider' : '';
  }

  function plannerSelectionValue(op = selectedOperation()) {
    const parameterName = plannerParameterName(op);
    const entry = parameterName ? state.parameterInputs.get(parameterName) : null;
    return entry
      ? String(readParameterValue(entry.spec, entry.input, false) || 'auto')
      : '';
  }

  function plannerIsBypassed(op = selectedOperation()) {
    if (!op || !op.context_ir_profile) return false;
    const entry = state.parameterInputs.get('prompt_mode');
    return Boolean(
      entry && String(readParameterValue(entry.spec, entry.input, false)) === 'raw'
    );
  }

  function externalPlannerSelected(op = selectedOperation()) {
    return Boolean(
      op && op.external_planner_api && !plannerIsBypassed(op)
      && plannerSelectionValue(op) === 'external'
    );
  }

  function setExternalPlannerStatus(text, kind = '') {
    els.externalPlannerStatus.textContent = text;
    els.externalPlannerStatus.className = `external-planner-status ${kind}`.trim();
  }

  function externalPlannerConnection(op = selectedOperation(), validate = true, requireModel = true) {
    if (!externalPlannerSelected(op)) return null;
    const protocol = els.externalPlannerProtocol.value;
    const baseUrl = els.externalPlannerUrl.value.trim();
    const token = els.externalPlannerToken.value.trim();
    const model = els.externalPlannerModel.value.trim();
    if (validate) {
      if (!baseUrl) throw new Error('请输入第三方 API URL。');
      let parsed;
      try {
        parsed = new URL(baseUrl);
      } catch (_error) {
        throw new Error('第三方 API URL 必须是完整的 http(s) 地址。');
      }
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        throw new Error('第三方 API URL 必须使用 http 或 https。');
      }
      if (!token) throw new Error('请输入第三方 API token。');
      if (requireModel && !model) throw new Error('请先识别或输入第三方模型 ID。');
    }
    return {
      protocol,
      baseUrl,
      token,
      model,
      modalities: els.externalPlannerCapability.value === 'vision'
        ? ['text', 'image']
        : ['text'],
    };
  }

  function externalPlannerPayload(op = selectedOperation()) {
    const connection = externalPlannerConnection(op, true, true);
    if (!connection) return {};
    return {
      external_api_protocol: connection.protocol,
      external_api_base_url: connection.baseUrl,
      external_api_model: connection.model,
      external_api_modalities: connection.modalities.join(','),
    };
  }

  function syncExternalPlannerUi(op = selectedOperation()) {
    const visible = externalPlannerSelected(op);
    els.externalPlanner.hidden = !visible;
    if (!visible) return;
    const busy = Boolean(state.controller) || state.externalPlannerDetecting;
    els.externalPlannerProtocol.disabled = busy;
    els.externalPlannerCapability.disabled = busy;
    els.externalPlannerUrl.disabled = busy;
    els.externalPlannerToken.disabled = busy;
    els.toggleExternalPlannerToken.disabled = busy;
    els.externalPlannerModel.disabled = busy;
    els.detectExternalPlannerModels.disabled = busy
      || !els.externalPlannerUrl.value.trim()
      || !els.externalPlannerToken.value.trim();
  }

  function invalidateExternalPlannerDetection() {
    els.externalPlannerModelList.replaceChildren();
    if (!state.externalPlannerDetecting) setExternalPlannerStatus('连接参数已变化');
    syncExternalPlannerUi();
    syncSendState();
  }

  async function detectExternalPlannerModels() {
    const op = selectedOperation();
    let connection;
    try {
      connection = externalPlannerConnection(op, true, false);
    } catch (error) {
      setExternalPlannerStatus(error instanceof Error ? error.message : String(error), 'error');
      return;
    }
    if (!connection || !op.external_planner_api) return;
    state.externalPlannerDetecting = true;
    setExternalPlannerStatus('正在识别…');
    syncExternalPlannerUi(op);
    try {
      const headers = authHeaders(true);
      headers['X-Playground-Upstream-Key'] = connection.token;
      const response = await fetch(op.external_planner_api.models_endpoint, {
        method: 'POST',
        headers,
        cache: 'no-store',
        body: JSON.stringify({
          profile: op.context_ir_profile || els.model.value,
          protocol: connection.protocol,
          base_url: connection.baseUrl,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      const models = Array.isArray(data.models)
        ? data.models.filter((item) => typeof item === 'string' && item.trim())
        : [];
      els.externalPlannerModelList.replaceChildren();
      for (const model of models) {
        els.externalPlannerModelList.append(new Option(model, model));
      }
      if (data.base_url) els.externalPlannerUrl.value = data.base_url;
      if (!els.externalPlannerModel.value.trim() && models.length) {
        els.externalPlannerModel.value = models[0];
      }
      setExternalPlannerStatus(
        models.length ? `已识别 ${models.length} 个模型` : '连接成功；请手动输入模型 ID',
        'ok',
      );
    } catch (error) {
      setExternalPlannerStatus(
        `识别失败：${error instanceof Error ? error.message : String(error)}`,
        'error',
      );
    } finally {
      state.externalPlannerDetecting = false;
      syncExternalPlannerUi(op);
      updatePlannerProviderUi();
      syncSendState();
    }
  }

  function effectivePlannerChoice(op = selectedOperation()) {
    if (!op) return null;
    const parameterName = plannerParameterName(op);
    if (!parameterName || plannerIsBypassed(op)) return null;
    if (state.controller && state.activePlannerChoice) {
      return state.activePlannerChoice;
    }
    const entry = state.parameterInputs.get(parameterName);
    if (!entry) return null;
    const choices = Array.isArray(entry.spec.choices) ? entry.spec.choices : [];
    const selectedValue = String(readParameterValue(entry.spec, entry.input, false) || 'auto');
    let choice = choices.find((item) => String(item.value) === selectedValue) || null;
    if (choice && selectedValue === 'external') {
      const connection = externalPlannerConnection(op, false, false);
      choice = {
        ...choice,
        protocol: connection.protocol,
        model: connection.model || null,
        modalities: connection.modalities,
      };
    }
    if (selectedValue === 'auto') {
      const hasImages = state.attachments.some((attachment) => attachment.kind === 'image');
      const defaults = op.planner_defaults || {};
      const effectiveValue = hasImages ? defaults.image : defaults.text;
      choice = choices.find((item) => String(item.value) === String(effectiveValue)) || choice;
    }
    return choice;
  }

  function plannerProviderWarning(op = selectedOperation()) {
    const hasImages = state.controller
      ? state.activePlannerHasImages
      : state.attachments.some((attachment) => attachment.kind === 'image');
    const choice = effectivePlannerChoice(op);
    const modalities = new Set((choice && choice.modalities) || []);
    if (hasImages && choice && !modalities.has('image')) {
      return '提醒：所选 Planner 是纯文字模型。参考图仍用于 H3 的 <Picture N> 对齐，但不会发送给 Planner，因此它无法理解图片内容。';
    }
    return '';
  }

  function updatePlannerProviderUi() {
    const op = selectedOperation();
    const parameterName = plannerParameterName(op);
    const entry = parameterName ? state.parameterInputs.get(parameterName) : null;
    const bypassed = plannerIsBypassed(op);
    if (entry) entry.field.hidden = bypassed;
    if (!op || !parameterName || bypassed) {
      els.externalPlanner.hidden = true;
      showContextNotice('');
      return;
    }
    syncExternalPlannerUi(op);
    const choice = effectivePlannerChoice(op);
    if (choice && op.id === 'h3_context_ir') {
      renderModelResourceChips(choice, {planner: true, showContext: true});
    }
    const warning = plannerProviderWarning(op);
    if (entry) {
      let warningElement = entry.field.querySelector('.parameter-warning');
      if (warning && !warningElement) {
        warningElement = document.createElement('div');
        warningElement.className = 'help parameter-warning';
        entry.field.append(warningElement);
      }
      if (warningElement) {
        warningElement.textContent = warning;
        warningElement.hidden = !warning;
      }
    }
    showContextNotice(warning);
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
    els.modelChip.textContent = info ? (info.label || info.id) : '未选择模型';
    const hasChat = operations.some((operation) => operation.id === 'chat');
    renderModelResourceChips(info, {showContext: hasChat});
    syncOperation();
  }

  function syncOperation() {
    const op = selectedOperation();
    const isChat = op && op.id === 'chat';
    const isChatRequest = operationUsesChatEndpoint(op);
    const isImageEdit = op && op.id === 'image_edit';
    const isVideoGeneration = op && op.id === 'video_generation';
    const isVideoAnalysis = op && op.id === 'video_analysis';
    const isContextIR = op && op.id === 'h3_context_ir';
    const isDecision = op && op.id === 'decision';
    const isOcr = Boolean(op && op.ocr_mode);
    const isMedia = op && !isChatRequest && !isDecision && !isOcr;
    const usesMediaQueue = operationUsesMediaQueue(op);
    const acceptsUploads = op && (op.accepts_images || op.accepts_videos);

    const preserveDecisionImages = isDecision && state.activeDecisionModelId === els.model.value;
    state.activeDecisionModelId = isDecision ? els.model.value : null;
    if (op) {
      const compatible = state.attachments.filter((attachment) => (
        (attachment.kind === 'image' && (op.accepts_images || preserveDecisionImages))
        || (attachment.kind === 'video' && op.accepts_videos)
      ));
      if (compatible.length !== state.attachments.length) {
        state.attachments = compatible;
        els.fileInput.value = '';
        renderAttachments();
      }
    }
    syncDecisionForm(op);
    const hasDecisionImages = isDecision && state.attachments.some((attachment) => attachment.kind === 'image');
    els.uploadSection.hidden = !acceptsUploads && !hasDecisionImages;
    els.dropZone.disabled = Boolean(isDecision && !acceptsUploads);
    els.fileInput.disabled = Boolean(isDecision && !acceptsUploads);
    els.cameraSection.hidden = !operationSupportsLiveCamera(op);
    els.mediaQueueSection.hidden = !usesMediaQueue;
    els.uploadLabel.textContent = isVideoAnalysis ? '待分析视频' : (op && op.accepts_videos ? 'Reference images / videos' : '参考图片');
    els.uploadRequirement.textContent = isVideoAnalysis
      ? '文件模式需要 1 个；摄像头模式无需上传'
      : (op && (op.requires_images || op.requires_videos) ? '至少需要 1 个文件' : '可选');
    els.fileInput.accept = isDecision && op.image_format === 'typesafe_assets'
      ? 'image/png,image/jpeg'
      : [op && op.accepts_images ? 'image/*' : '', op && op.accepts_videos ? 'video/*' : ''].filter(Boolean).join(',');
    els.fileInput.multiple = Boolean(op && (
      (op.accepts_images && op.multiple_images !== false)
      || (op.accepts_videos && op.multiple_videos === true)
    ));
    els.dropText.textContent = isVideoAnalysis
      ? '拖放或选择本地视频'
      : (op && op.accepts_videos ? 'Drop or select reference images / videos' : '粘贴、拖放或选择图片');
    els.dropHint.textContent = isVideoAnalysis
      ? '上传单个视频，或使用下方手机 / PC 摄像头持续分析'
      : (op && op.accepts_videos ? 'Ref2VA: up to 9 images and 3 videos at 24 FPS; visual references only' : '支持多张参考图；点击这里选择文件');
    if (isDecision && op.accepts_images) {
      els.dropHint.textContent = op.image_format === 'startlux_images'
        ? '图片按后端声明写入顶层 images 数组，原始 State 保持不变。'
        : (op.image_format === 'typesafe_assets' ? 'PNG / JPEG；自动写入 assets，并在 State.images 中引用。' : '图片通过原生 input_image 内容发送。');
    }
    if (hasDecisionImages && !acceptsUploads) {
      els.dropHint.textContent = '当前格式不支持这些图片。请切回后端原生格式或移除图片；不会自动转换或丢弃。';
    }
    els.chatParams.hidden = !isChatRequest;
    els.chatParams.closest('.advanced').hidden = Boolean(isDecision || isOcr);
    els.mediaParams.hidden = !(isMedia || isVideoAnalysis);
    els.ttftMetric.hidden = !isChatRequest;
    els.resultTitle.textContent = isDecision ? '决策结果' : isChat
      ? '连续对话'
      : (isVideoAnalysis ? '视频分析记录' : (isContextIR ? 'H3 Prompt 增强记录' : (op ? '连续调试' : '交互记录')));
    els.clear.textContent = '清空记录';
    els.shortcut.textContent = isDecision ? 'Enter 运行决策 · Ctrl / ⌘ + Enter 换行' : isVideoAnalysis
      ? 'Enter 分析 · Ctrl / ⌘ + Enter 换行 · 可拖放视频'
      : (isChat
        ? 'Enter 发送 · Ctrl / ⌘ + Enter 换行 · 可直接粘贴图片'
        : 'Enter 执行 · Ctrl / ⌘ + Enter 换行 · 可直接粘贴图片');

    if (!op) {
      const info = selectedModel();
      const caps = info && info.capabilities ? info.capabilities.join(', ') : '无';
      els.historyModeChip.textContent = '';
      els.historyModeChip.hidden = true;
      els.emptyHint.textContent = '交互记录只保留在当前页面内。';
      els.operationHint.textContent = `已声明能力：${caps}。当前没有可执行的已适配操作。`;
      els.prompt.placeholder = '该模型没有可执行的 Playground 操作';
      els.send.textContent = '不可执行';
    } else {
      const usesHistory = operationUsesHistory(op);
      els.historyModeChip.textContent = usesHistory ? '携带历史' : '单轮请求';
      els.historyModeChip.title = usesHistory
        ? '每次请求会携带本页中已完成的较早对话'
        : '接口仅接收本次输入；较早记录只在本页显示';
      els.historyModeChip.hidden = false;
      els.emptyHint.textContent = usesHistory
        ? '后续请求会携带本页对话，并按模型 context 自动裁剪。'
        : '接口仅接收本次输入；此前的输入与结果仍会保留在本页。';
      const imageHint = op.accepts_images
        ? (op.requires_images ? '，需要图片输入' : '，可附带图片输入')
        : '';
      const videoHint = op.accepts_videos
        ? (op.requires_videos ? '，需要视频输入' : '，可附带视频输入')
        : '';
      const limits = op.limits || {};
      const limitHints = [usesHistory ? '携带页面内历史' : '仅发送本次输入'];
      if (op.configured === false) limitHints.push('工作流尚未配置');
      if (usesMediaQueue) limitHints.push(`可自动排队 1–${MAX_MEDIA_QUEUE_RUNS} 份`);
      else if (limits.max_batch_size) limitHints.push(`最多 ${limits.max_batch_size} 个结果`);
      if (limits.max_reference_images) limitHints.push(`最多 ${limits.max_reference_images} 张参考图`);
      if (limits.max_num_frames) limitHints.push(`最多 ${limits.max_num_frames} 帧`);
      if (limits.max_video_bytes) {
        limitHints.push(`视频不超过 ${(limits.max_video_bytes / 1024 / 1024).toFixed(0)} MiB`);
      }
      const contextLength = isChatRequest ? modelContextLength() : null;
      if (contextLength) limitHints.push(`上下文 ${contextLength.toLocaleString()} tokens`);
      els.operationHint.textContent = [
        `${op.endpoint}${op.stream ? ' · 流式' : ''}${imageHint}${videoHint}`,
        ...limitHints,
      ].join(' · ');
      els.prompt.placeholder = isVideoAnalysis
        ? '输入希望模型持续关注和分析的问题…'
        : (isChat
          ? (op.accepts_images ? '输入问题；也可以粘贴或上传图片…' : '输入要发送给模型的内容…')
          : (isContextIR ? '用自然语言描述要生成的 H3 视频…' : (isVideoGeneration ? '描述要生成的视频…' : (isImageEdit ? '描述希望怎样编辑图片…' : '描述要生成的图片…'))));
      if (isDecision) els.prompt.placeholder = '输入本次决策的状态 / 背景，或选择 JSON 后输入对象 / 数组…';
      els.send.textContent = isDecision ? '运行决策 →' : isVideoAnalysis
        ? '分析视频 →'
        : (isChat ? '发送请求 →' : (isContextIR ? '增强 H3 Prompt →' : (isVideoGeneration ? '生成视频 →' : (isImageEdit ? '编辑图片 →' : '生成图片 →'))));
    }
    renderOperationParameters(op);
    syncOcrForm(op);
    resetOutput();
    renderInteractionHistory();
    updatePlannerProviderUi();
    updateContextPreview();
    syncSendState();
    syncCameraControls();
  }

  function operationParameters(op) {
    if (!op || op.id === 'chat') return [];
    if (Array.isArray(op.parameters)) {
      return operationUsesMediaQueue(op)
        ? op.parameters.filter((parameter) => parameter.name !== 'n')
        : op.parameters;
    }

    // Compatibility with servers that predate the workflow-derived schema.
    const defaults = op.defaults || {};
    const legacy = [
      ['size', '尺寸', 'resolution'],
      ['n', '数量', 'integer'],
      ['seed', 'Seed', 'integer'],
      ['steps', 'Steps', 'integer'],
      ['cfg', 'CFG', 'number'],
      ['denoise', 'Denoise', 'number'],
      ['num_frames', '帧数', 'integer'],
      ['fps', 'FPS', 'number'],
      ['prefetch_count', 'Prefetch count', 'integer'],
      ['enable_tile', '分块解码', 'boolean'],
      ['enable_streaming', '模型流式加载', 'boolean'],
    ];
    const parameters = legacy
      .filter(([name]) => Object.prototype.hasOwnProperty.call(defaults, name))
      .map(([name, label, type]) => ({name, label, type, default: defaults[name]}));
    return operationUsesMediaQueue(op)
      ? parameters.filter((parameter) => parameter.name !== 'n')
      : parameters;
  }

  function setParameterValue(entry, value) {
    if (entry.spec.type === 'boolean') {
      entry.input.checked = Boolean(value);
      const status = entry.field.querySelector('.boolean-status');
      if (status) status.textContent = entry.input.checked ? '开启' : '关闭';
    } else {
      entry.input.value = value == null ? '' : String(value);
    }
  }

  function presetMatches(preset) {
    return Object.entries(preset.values || {}).every(([name, expected]) => {
      const entry = state.parameterInputs.get(name);
      if (!entry) return true;
      const actual = readParameterValue(entry.spec, entry.input, false);
      return entry.spec.type === 'number'
        ? Number(actual) === Number(expected)
        : actual === expected;
    });
  }

  function syncPresetSelection() {
    const op = selectedOperation();
    if (!op || !Array.isArray(op.presets) || !op.presets.length) return;
    const match = op.presets.find(presetMatches);
    els.operationPreset.value = match ? match.id : '';
    els.operationPresetHint.textContent = match
      ? (match.description || '')
      : '参数已调整为自定义值。';
  }

  function applyOperationPreset(op, presetId) {
    const preset = (op.presets || []).find((item) => item.id === presetId);
    if (!preset) {
      els.operationPresetHint.textContent = '参数已调整为自定义值。';
      return;
    }
    for (const [name, value] of Object.entries(preset.values || {})) {
      const entry = state.parameterInputs.get(name);
      if (entry) setParameterValue(entry, value);
    }
    els.operationPresetHint.textContent = preset.description || '';
  }

  function renderOperationPresets(op) {
    const presets = op && Array.isArray(op.presets) ? op.presets : [];
    els.operationPreset.replaceChildren();
    els.operationPresetHint.textContent = '';
    els.operationPresetField.hidden = !presets.length;
    if (!presets.length) return;

    els.operationPreset.append(new Option('自定义', ''));
    for (const preset of presets) {
      const suffix = preset.recommended ? '（推荐）' : '';
      els.operationPreset.append(new Option(`${preset.label}${suffix}`, preset.id));
    }
    const selected = op.default_preset || '';
    els.operationPreset.value = selected;
    if (selected) applyOperationPreset(op, selected);
    else syncPresetSelection();
  }

  function renderOperationParameters(op) {
    state.parameterInputs.clear();
    els.operationParameterList.replaceChildren();
    if (!op || op.id === 'chat') {
      els.operationPresetField.hidden = true;
      return;
    }

    const parameters = operationParameters(op);
    if (!parameters.length) {
      const empty = document.createElement('div');
      empty.className = 'help empty-parameters';
      empty.textContent = '该工作流没有可调整的公开参数。';
      els.operationParameterList.append(empty);
      renderOperationPresets(op);
      return;
    }

    for (const spec of parameters) {
      const field = document.createElement('div');
      field.className = [
        'field operation-parameter',
        spec.advanced ? 'advanced-parameter' : '',
        spec.wide ? 'wide-parameter' : '',
      ].filter(Boolean).join(' ');
      const id = `operation-param-${String(spec.name).replace(/[^a-zA-Z0-9_-]/g, '-')}`;
      const label = document.createElement('label');
      label.htmlFor = id;
      label.textContent = spec.label || spec.name;
      let input;

      if (spec.type === 'select') {
        input = document.createElement('select');
        const groups = new Map();
        for (const choice of spec.choices || []) {
          const option = new Option(
            choice.label == null ? choice.value : choice.label,
            choice.value,
          );
          if (choice.group) {
            let group = groups.get(choice.group);
            if (!group) {
              group = document.createElement('optgroup');
              group.label = choice.group;
              groups.set(choice.group, group);
              input.append(group);
            }
            group.append(option);
          } else {
            input.append(option);
          }
        }
      } else {
        input = document.createElement('input');
        input.className = 'control';
        if (spec.type === 'boolean') {
          input.type = 'checkbox';
          input.className = 'boolean-input';
          const toggle = document.createElement('label');
          toggle.className = 'boolean-control';
          toggle.htmlFor = id;
          const status = document.createElement('span');
          status.className = 'boolean-status';
          toggle.append(input, status);
          field.append(label, toggle);
        } else {
          input.type = ['integer', 'number'].includes(spec.type) ? 'number' : 'text';
          if (spec.type === 'resolution') input.placeholder = '例如 1024x1024';
          for (const attribute of ['min', 'max', 'step']) {
            if (spec[attribute] != null) input.setAttribute(attribute, String(spec[attribute]));
          }
        }
      }

      input.id = id;
      input.dataset.parameter = spec.name;
      if (spec.type !== 'boolean') field.append(label, input);
      if (spec.description) {
        const help = document.createElement('div');
        help.className = 'help';
        help.textContent = spec.description;
        field.append(help);
      }
      const entry = {spec, input, field};
      state.parameterInputs.set(spec.name, entry);
      setParameterValue(entry, spec.default);
      input.addEventListener('input', () => {
        if (spec.type === 'boolean') setParameterValue(entry, input.checked);
        syncPresetSelection();
        updatePlannerProviderUi();
      });
      input.addEventListener('change', () => {
        if (['video_mode', 'keyframes'].includes(spec.name)) {
          const mode = state.parameterInputs.get('video_mode');
          const guides = state.parameterInputs.get('keyframes');
          const promptMode = state.parameterInputs.get('prompt_mode');
          if (promptMode && ((mode && mode.input.value === 'ref2va')
              || (guides && !['', '[]'].includes(guides.input.value.trim())))) {
            setParameterValue(promptMode, 'raw');
          }
        }
        syncPresetSelection();
        updatePlannerProviderUi();
      });
      els.operationParameterList.append(field);
    }
    renderOperationPresets(op);
  }

  function syncSendState() {
    const op = selectedOperation();
    updatePlannerProviderUi();
    const hasImage = state.attachments.some((attachment) => attachment.kind === 'image');
    const hasVideo = state.attachments.some((attachment) => attachment.kind === 'video');
    const missingRequiredMedia = Boolean(op && (
      (op.requires_images && !hasImage) || (op.requires_videos && !hasVideo)
    ));
    const credentialReady = state.loadedCredential !== null
      && state.loadedCredential === els.apiKey.value.trim();
    const cameraBusy = state.camera.starting || state.camera.active || state.camera.processing;
    const externalConnection = externalPlannerConnection(op, false, false);
    const externalPlannerReady = !externalConnection || Boolean(
      externalConnection.baseUrl && externalConnection.token && externalConnection.model
    );
    els.send.disabled = Boolean(state.controller) || cameraBusy || !selectedModel() || !op
      || !credentialReady || op.configured === false
      || (op.id !== 'decision' && !op.ocr_mode && !els.prompt.value.trim()) || missingRequiredMedia || !externalPlannerReady
      || (op.id === 'decision' && hasImage && !op.accepts_images);
    els.generationCount.disabled = Boolean(state.controller) || !operationUsesMediaQueue(op);
    for (const input of els.decisionSection.querySelectorAll('input, select, textarea, button')) {
      input.disabled = Boolean(state.controller) || (input.classList.contains('decision-remove') && decisionForm(op)?.questions.length <= 1);
    }
    for (const input of $('ocrSection').querySelectorAll('input, select, textarea, button')) {
      input.disabled = Boolean(state.controller) || input.dataset.unavailable === 'true';
    }
    if (op && op.ocr_mode) {
      els.send.disabled ||= state.ocrReleasing;
      $('ocrRelease').disabled = Boolean(state.controller) || state.ocrReleasing || !credentialReady
        || !state.ocrRuntime?.worker_running || state.ocrRuntime?.busy;
      $('ocrRefreshRuntime').disabled = Boolean(state.controller) || state.ocrReleasing || !credentialReady;
      const requiresVerifier = ['auto', 'score'].includes(op.ocr_mode);
      const verifierReady = (state.ocrCatalog?.verifiers || []).some(v => v.id === $('ocrVerifier').value && v.available);
      $('ocrThreshold').disabled = Boolean(state.controller) || !verifierReady;
      $('ocrVerifier').options[0].textContent = requiresVerifier ? '请选择复核模型（必选）' : '不复核';
      $('ocrVerifierHint').textContent = requiresVerifier
        ? '此模式需要复核模型；请选择后再运行。'
        : '可选；不复核时仅返回 OCR 识别结果。';
      els.send.disabled ||= (op.ocr_mode === 'compare' && !$('ocrEngines').querySelector('input:checked'))
        || (op.ocr_mode === 'auto' && !$('ocrFallback').value)
        || (requiresVerifier && !verifierReady)
        || (Boolean($('ocrVerifier').value) && !verifierReady)
        || (op.ocr_mode === 'score' && ![...$('ocrCandidates').querySelectorAll('textarea')].some(e => e.value.trim()));
    }
    updateDecisionPreview();
    syncCameraControls();
  }

  function showError(message) {
    els.errorBox.textContent = message;
    els.errorBox.classList.add('visible');
    els.emptyState.hidden = true;
  }

  function showRequestError(op, message) {
    // Remove the temporary pending turn and progress label after a failure.
    // Partial streamed chat output remains useful for debugging, so preserve it.
    const keepPartialChat = operationUsesChatEndpoint(op)
      && Boolean(els.answerText.textContent.trim() || els.reasoningText.textContent.trim());
    if (!keepPartialChat) {
      els.answerText.textContent = '';
      els.answer.classList.remove('active');
      els.reasoningText.textContent = '';
      els.reasoning.classList.remove('visible');
    }
    els.answerText.classList.remove('cursor');
    renderInteractionHistory();
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
    els.emptyState.hidden = false;
    els.ttftMetric.textContent = '首字 —';
    els.timeMetric.textContent = '耗时 —';
  }

  function beginRequest(op, plan = null) {
    resetOutput();
    renderInteractionHistory(
      plan && (plan.displayUserMessage || plan.userMessage),
      plan && plan.parameters,
    );
    els.answer.classList.add('active');
    if (operationUsesChatEndpoint(op)) {
      els.answerText.classList.add('cursor');
    } else {
      els.answerText.textContent = op.id === 'decision' ? '正在评估决策问题…' : op.id === 'h3_context_ir'
        ? '正在规划并校验 H3 Prompt…'
        : (op.id === 'video_generation'
          ? `正在生成第 1/${(plan && plan.runCount) || 1} 份视频…`
          : (op.id === 'image_edit'
            ? `正在生成第 1/${(plan && plan.runCount) || 1} 份编辑结果…`
            : `正在生成第 1/${(plan && plan.runCount) || 1} 份图片…`));
    }
    if (op.ocr_mode) els.answerText.textContent = '正在识别 / 复核…';
    if (op.ocr_mode) els.apiKey.disabled = true;
    els.loadModels.disabled = true;
    els.model.disabled = true;
    els.operation.disabled = true;
    els.send.hidden = true;
    els.stop.hidden = false;
    syncExternalPlannerUi(op);
  }

  function finishRequest() {
    els.apiKey.disabled = false;
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null;
    els.answerText.classList.remove('cursor');
    state.controller = null;
    state.activePlannerChoice = null;
    state.activePlannerHasImages = false;
    const cameraBusy = state.camera.starting || state.camera.active || state.camera.processing;
    els.loadModels.disabled = cameraBusy;
    els.model.disabled = cameraBusy || state.models.size === 0;
    els.operation.disabled = cameraBusy || modelOperations(selectedModel()).length === 0;
    els.stop.hidden = true;
    els.send.hidden = false;
    updateContextPreview();
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

  function formatCameraTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return hours
      ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
      : `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function setCameraStatus(text, kind = '') {
    els.cameraStatus.textContent = text;
    els.cameraStatus.className = `camera-state ${kind}`.trim();
  }

  function cameraWindowSeconds() {
    const entry = state.parameterInputs.get('segment_seconds');
    const value = entry ? Number(readParameterValue(entry.spec, entry.input, false)) : 8;
    return Number.isFinite(value) ? Math.min(60, Math.max(2, value)) : 8;
  }

  function updateCameraStats() {
    const camera = state.camera;
    const windowSeconds = camera.settings
      ? camera.settings.windowSeconds
      : cameraWindowSeconds();
    const elapsed = camera.startedAt
      ? formatCameraTime((performance.now() - camera.startedAt) / 1000)
      : '00:00';
    const queued = camera.pending ? ' · 待处理 1' : '';
    const windowLabel = Number.isInteger(windowSeconds)
      ? String(windowSeconds)
      : Number(windowSeconds).toFixed(1);
    els.cameraStats.textContent = [
      `窗口 ${windowLabel}s`,
      `已录制 ${camera.sequence}`,
      `已分析 ${camera.analyzed}`,
      `丢弃 ${camera.dropped}`,
      `错误 ${camera.errors}`,
      `运行 ${elapsed}${queued}`,
    ].join(' · ');
  }

  function updateCameraStateStatus() {
    const camera = state.camera;
    if (camera.starting) {
      setCameraStatus('等待权限', 'busy');
    } else if (camera.active && camera.processing) {
      setCameraStatus('录制中 · GPU 分析中', 'busy');
    } else if (camera.active && camera.lastError) {
      setCameraStatus('录制中 · 上段失败', 'error');
    } else if (camera.active) {
      setCameraStatus('录制中', 'live');
    } else if (camera.processing) {
      setCameraStatus('摄像头已停 · 分析收尾', 'busy');
    } else if (camera.lastError) {
      setCameraStatus('已停止：发生错误', 'error');
    } else if (camera.startedAt) {
      setCameraStatus('已停止');
    } else {
      setCameraStatus('未启动');
    }
    updateCameraStats();
  }

  function syncCameraControls() {
    const camera = state.camera;
    const op = selectedOperation();
    const supported = operationSupportsLiveCamera(op);
    const credentialReady = state.loadedCredential !== null
      && state.loadedCredential === els.apiKey.value.trim();
    const cameraBusy = camera.starting || camera.active || camera.processing;
    const requestBusy = Boolean(state.controller);
    const operationReady = supported && op.configured !== false;

    els.cameraStart.hidden = camera.active;
    els.cameraStart.textContent = camera.starting
      ? '等待摄像头权限…'
      : ((!camera.active && camera.processing) ? '等待当前分析…' : '开始实时分析');
    els.cameraStart.disabled = !operationReady || !credentialReady
      || !els.prompt.value.trim() || cameraBusy || requestBusy;
    els.cameraStop.hidden = !camera.active;
    els.cameraFacing.disabled = cameraBusy;

    els.apiKey.disabled = cameraBusy;
    els.toggleKey.disabled = cameraBusy;
    els.prompt.disabled = cameraBusy;
    els.systemPrompt.disabled = cameraBusy;
    els.temperature.disabled = cameraBusy;
    els.maxTokens.disabled = cameraBusy;
    const decisionUploadsBlocked = op && op.id === 'decision' && (requestBusy || !op.accepts_images);
    els.fileInput.disabled = cameraBusy || decisionUploadsBlocked;
    els.dropZone.disabled = cameraBusy || decisionUploadsBlocked;
    els.clear.disabled = cameraBusy;
    for (const entry of state.parameterInputs.values()) {
      entry.input.disabled = cameraBusy;
    }

    if (cameraBusy) {
      els.loadModels.disabled = true;
      els.model.disabled = true;
      els.operation.disabled = true;
    } else if (!requestBusy) {
      els.loadModels.disabled = false;
      els.model.disabled = state.models.size === 0;
      els.operation.disabled = modelOperations(selectedModel()).length === 0;
    }
    updateCameraStateStatus();
  }

  function preferredCameraMimeType() {
    if (!window.MediaRecorder || typeof window.MediaRecorder.isTypeSupported !== 'function') {
      return '';
    }
    const candidates = [
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4;codecs=avc1.42E01E',
      'video/mp4',
    ];
    return candidates.find((value) => window.MediaRecorder.isTypeSupported(value)) || '';
  }

  function recordCameraWindow(stream, durationMs) {
    const preferredMime = preferredCameraMimeType();
    const options = {videoBitsPerSecond: LIVE_CAMERA_VIDEO_BITS_PER_SECOND};
    if (preferredMime) options.mimeType = preferredMime;
    let recorder;
    try {
      recorder = new window.MediaRecorder(stream, options);
    } catch (_) {
      recorder = new window.MediaRecorder(stream);
    }
    state.camera.recorder = recorder;

    return new Promise((resolve, reject) => {
      const chunks = [];
      let settled = false;
      const clearRecorderTimer = () => {
        if (state.camera.stopTimer) window.clearTimeout(state.camera.stopTimer);
        state.camera.stopTimer = null;
      };
      recorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size) chunks.push(event.data);
      });
      recorder.addEventListener('error', (event) => {
        if (settled) return;
        settled = true;
        clearRecorderTimer();
        if (state.camera.recorder === recorder) state.camera.recorder = null;
        reject(event.error || new Error('摄像头录制失败'));
      });
      recorder.addEventListener('stop', () => {
        if (settled) return;
        settled = true;
        clearRecorderTimer();
        if (state.camera.recorder === recorder) state.camera.recorder = null;
        const baseMime = String(recorder.mimeType || preferredMime || 'video/webm')
          .split(';')[0];
        resolve(new Blob(chunks, {type: baseMime}));
      });
      try {
        recorder.start(1000);
      } catch (error) {
        settled = true;
        if (state.camera.recorder === recorder) state.camera.recorder = null;
        reject(error);
        return;
      }
      state.camera.stopTimer = window.setTimeout(() => {
        if (recorder.state !== 'inactive') recorder.stop();
      }, durationMs);
    });
  }

  function stopCameraHardware() {
    const camera = state.camera;
    if (camera.stopTimer) window.clearTimeout(camera.stopTimer);
    camera.stopTimer = null;
    if (camera.recorder && camera.recorder.state !== 'inactive') {
      try { camera.recorder.stop(); } catch (_) { /* recorder already stopping */ }
    }
    if (camera.stream) {
      for (const track of camera.stream.getTracks()) track.stop();
    }
    camera.stream = null;
    els.cameraPreview.srcObject = null;
    els.cameraPreview.hidden = true;
    els.cameraPlaceholder.hidden = false;
  }

  function stopCameraAnalysis(message = '已停止', isError = false) {
    const camera = state.camera;
    camera.starting = false;
    camera.active = false;
    camera.pending = null;
    if (isError) camera.lastError = message;
    else camera.lastError = '';
    stopCameraHardware();
    if (!camera.processing && camera.uiTimer) {
      window.clearInterval(camera.uiTimer);
      camera.uiTimer = null;
    }
    updateCameraStateStatus();
    syncSendState();
  }

  function cameraRequestSettings(op) {
    const parameters = {};
    for (const [name, entry] of state.parameterInputs.entries()) {
      const value = readParameterValue(entry.spec, entry.input);
      if (value !== null) parameters[name] = value;
    }
    const modelTokenLimit = outputTokenReserve(selectedModel()) || 96;
    const requestedTokens = requestedMaxTokens();
    const temperature = Number(els.temperature.value);
    return {
      modelId: els.model.value,
      operationId: op.id,
      endpoint: op.endpoint,
      headers: authHeaders(true),
      prompt: els.prompt.value.trim(),
      systemMessages: baseSystemMessages(),
      parameters,
      windowSeconds: cameraWindowSeconds(),
      maxTokens: Math.max(16, Math.min(128, requestedTokens, modelTokenLimit)),
      temperature: Number.isFinite(temperature) ? temperature : 0,
      maxVideoBytes: Number(op.limits && op.limits.max_video_bytes) || (64 * 1024 * 1024),
    };
  }

  function consumeCameraSseData(data, result, onUpdate) {
    if (data === '[DONE]') return true;
    const chunk = JSON.parse(data);
    if (chunk && chunk.error) {
      throw new Error(formatError(chunk.error, '实时摄像头分析失败'));
    }
    const choice = chunk && Array.isArray(chunk.choices) ? chunk.choices[0] : null;
    if (!choice) return false;
    const delta = choice.delta || {};
    const content = textFromContent(delta.content ?? '');
    const reasoning = textFromContent(
      delta.reasoning_content ?? delta.reasoning ?? delta.thinking ?? ''
    );
    if (content) result.content += content;
    if (reasoning) result.reasoning += reasoning;
    if (content || reasoning) onUpdate(result);
    return false;
  }

  async function readCameraSse(response, onUpdate) {
    if (!response.ok) throw new Error(await responseError(response));
    if (!response.body) throw new Error('浏览器没有提供可读取的响应流');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const result = {content: '', reasoning: ''};
    let buffer = '';
    let doneEvent = false;
    while (!doneEvent) {
      const item = await reader.read();
      buffer += decoder.decode(item.value || new Uint8Array(), {stream: !item.done});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const clean = line.endsWith('\r') ? line.slice(0, -1) : line;
        if (!clean.startsWith('data:')) continue;
        const data = clean.slice(5).trimStart();
        if (!data) continue;
        if (consumeCameraSseData(data, result, onUpdate)) {
          doneEvent = true;
          break;
        }
      }
      if (item.done) break;
    }
    if (buffer.trim() && !doneEvent) {
      const clean = buffer.endsWith('\r') ? buffer.slice(0, -1) : buffer;
      if (clean.startsWith('data:')) {
        const data = clean.slice(5).trimStart();
        if (data) consumeCameraSseData(data, result, onUpdate);
      }
    }
    return result;
  }

  function appendCameraTurn(segment, answer, reasoning, parameters) {
    const settings = state.camera.settings;
    if (!settings) return;
    const interaction = interactionFor(settings.modelId, settings.operationId);
    interaction.turns.push({
      user: {
        role: 'user',
        content: `[实时摄像头 ${segment.label} · 第 ${segment.sequence} 段]\n${settings.prompt}`,
      },
      assistant: {role: 'assistant', content: answer},
      reasoning,
      parameters: {
        ...parameters,
        live_camera: true,
        capture_sequence: segment.sequence,
        capture_range: segment.label,
        captured_bytes: segment.blob.size,
      },
    });
    if (interaction.turns.length > LIVE_CAMERA_HISTORY_LIMIT) {
      interaction.turns.splice(0, interaction.turns.length - LIVE_CAMERA_HISTORY_LIMIT);
    }
    if (els.model.value === settings.modelId && selectedOperation()?.id === settings.operationId) {
      renderInteractionHistory();
    }
  }

  async function analyzeCameraSegment(segment) {
    const settings = state.camera.settings;
    if (!settings) throw new Error('实时摄像头会话配置已丢失');
    if (segment.blob.size > settings.maxVideoBytes) {
      throw new Error(`摄像头片段超过 ${(settings.maxVideoBytes / 1024 / 1024).toFixed(0)} MiB`);
    }
    const dataUrl = await readFileAsDataUrl(segment.blob);
    const prompt = [
      `这是持续摄像头会话的第 ${segment.sequence} 个片段，源时间 ${segment.label}。`,
      '只描述这个片段中实际可见的对象、动作、变化和异常；不确定时明确说明。',
      `用户持续关注的问题：${settings.prompt}`,
    ].join('\n');
    const parameters = {
      ...settings.parameters,
      segment_seconds: Math.min(60, Math.max(2, segment.durationSeconds + 1)),
      max_segments: 1,
      include_summary: false,
    };
    const body = {
      model: settings.modelId,
      messages: [
        ...settings.systemMessages,
        {
          role: 'user',
          content: [
            {type: 'text', text: prompt},
            {type: 'video_url', video_url: {url: dataUrl}},
          ],
        },
      ],
      stream: true,
      max_tokens: settings.maxTokens,
      temperature: settings.temperature,
      video_duration_seconds: segment.durationSeconds,
      ...parameters,
    };
    els.cameraLiveText.hidden = false;
    els.cameraLiveText.textContent = `[${segment.label}] 正在分析第 ${segment.sequence} 段…`;
    const response = await fetch(settings.endpoint, {
      method: 'POST',
      headers: settings.headers,
      body: JSON.stringify(body),
    });
    const result = await readCameraSse(response, (partial) => {
      const text = partial.content || '（正在生成）';
      els.cameraLiveText.textContent = `[${segment.label}]\n${text}`;
    });
    const answer = result.content.trim() || '（模型返回了空内容）';
    els.cameraLiveText.textContent = `[${segment.label}]\n${answer}`;
    appendCameraTurn(segment, answer, result.reasoning, parameters);
  }

  async function drainCameraSegments() {
    const camera = state.camera;
    if (camera.processing) return;
    camera.processing = true;
    syncSendState();
    try {
      while (camera.pending) {
        const segment = camera.pending;
        camera.pending = null;
        updateCameraStateStatus();
        try {
          await analyzeCameraSegment(segment);
          camera.analyzed += 1;
          camera.consecutiveErrors = 0;
          camera.lastError = '';
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          camera.errors += 1;
          camera.consecutiveErrors += 1;
          camera.lastError = message;
          els.cameraLiveText.hidden = false;
          els.cameraLiveText.textContent = `[${segment.label}] 分析失败：${message}`;
          if (camera.consecutiveErrors >= 3) {
            stopCameraAnalysis(`连续 ${camera.consecutiveErrors} 个片段失败：${message}`, true);
            break;
          }
        }
        updateCameraStateStatus();
      }
    } finally {
      camera.processing = false;
      if (!camera.active && camera.uiTimer) {
        window.clearInterval(camera.uiTimer);
        camera.uiTimer = null;
      }
      updateCameraStateStatus();
      syncSendState();
    }
  }

  function enqueueCameraSegment(segment) {
    const camera = state.camera;
    if (camera.pending) {
      camera.pending = segment;
      camera.dropped += 1;
    } else {
      camera.pending = segment;
    }
    updateCameraStateStatus();
    void drainCameraSegments();
  }

  async function cameraCaptureLoop() {
    const camera = state.camera;
    while (camera.active && camera.stream) {
      const startSeconds = (performance.now() - camera.startedAt) / 1000;
      const blob = await recordCameraWindow(
        camera.stream,
        Math.round(camera.settings.windowSeconds * 1000),
      );
      const endSeconds = (performance.now() - camera.startedAt) / 1000;
      if (!camera.active) break;
      if (!blob.size) throw new Error('摄像头没有产生可分析的视频数据');
      camera.sequence += 1;
      enqueueCameraSegment({
        sequence: camera.sequence,
        startSeconds,
        endSeconds,
        durationSeconds: Math.max(0.1, endSeconds - startSeconds),
        label: `${formatCameraTime(startSeconds)}–${formatCameraTime(endSeconds)}`,
        blob,
      });
    }
  }

  function cameraAccessError(error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return '当前浏览器没有开放 getUserMedia；手机通过 HTTP 访问时通常会被安全策略阻止。';
    }
    if (error && ['NotAllowedError', 'SecurityError'].includes(error.name)) {
      return `摄像头权限被拒绝：${message}。手机 HTTP 页面可能需要浏览器测试开关或后续 HTTPS 支持。`;
    }
    return `无法启动摄像头：${message}`;
  }

  async function startCameraAnalysis() {
    const camera = state.camera;
    const op = selectedOperation();
    if (camera.starting || camera.active || camera.processing || state.controller) return;
    try {
      if (!operationSupportsLiveCamera(op)) throw new Error('当前能力不支持实时摄像头');
      if (!els.prompt.value.trim()) throw new Error('请先填写希望模型持续关注的问题');
      if (
        state.loadedCredential === null
        || state.loadedCredential !== els.apiKey.value.trim()
      ) {
        throw new Error('请先用当前 API key 连接并获取模型列表');
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('当前浏览器没有开放 getUserMedia');
      }
      if (!window.MediaRecorder) throw new Error('当前浏览器不支持 MediaRecorder');

      camera.starting = true;
      camera.lastError = '';
      syncSendState();
      const facingMode = els.cameraFacing.value;
      const videoConstraints = {
        width: {ideal: 1280, max: 1920},
        height: {ideal: 720, max: 1080},
        frameRate: {ideal: 15, max: 30},
      };
      if (facingMode) videoConstraints.facingMode = {ideal: facingMode};
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
      camera.settings = cameraRequestSettings(op);
      camera.stream = stream;
      camera.starting = false;
      camera.active = true;
      camera.pending = null;
      camera.sequence = 0;
      camera.analyzed = 0;
      camera.dropped = 0;
      camera.errors = 0;
      camera.consecutiveErrors = 0;
      camera.lastError = '';
      camera.startedAt = performance.now();
      els.cameraPreview.srcObject = stream;
      els.cameraPreview.hidden = false;
      els.cameraPlaceholder.hidden = true;
      try { await els.cameraPreview.play(); } catch (_) { /* autoplay may already be active */ }
      for (const track of stream.getVideoTracks()) {
        track.addEventListener('ended', () => {
          if (camera.active) stopCameraAnalysis('摄像头已断开', true);
        }, {once: true});
      }
      resetOutput();
      renderInteractionHistory();
      els.resultTitle.textContent = '实时摄像头分析记录';
      if (!window.isSecureContext) {
        els.cameraLiveText.hidden = false;
        els.cameraLiveText.textContent = '当前页面不是安全上下文；若浏览器已允许摄像头，将继续运行。';
      }
      if (camera.uiTimer) window.clearInterval(camera.uiTimer);
      camera.uiTimer = window.setInterval(updateCameraStats, 500);
      syncSendState();
      camera.captureTask = cameraCaptureLoop().catch((error) => {
        if (!camera.active) return;
        const message = cameraAccessError(error);
        els.cameraLiveText.hidden = false;
        els.cameraLiveText.textContent = message;
        stopCameraAnalysis(message, true);
      });
    } catch (error) {
      camera.starting = false;
      const message = cameraAccessError(error);
      camera.lastError = message;
      stopCameraHardware();
      els.cameraLiveText.hidden = false;
      els.cameraLiveText.textContent = message;
      updateCameraStateStatus();
      syncSendState();
    }
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(reader.error || new Error('无法读取附件'));
      reader.readAsDataURL(file);
    });
  }

  function fileKind(file) {
    if (!file) return '';
    const type = String(file.type || '').toLowerCase();
    if (type.startsWith('image/')) return 'image';
    if (type.startsWith('video/')) return 'video';
    const name = String(file.name || '').toLowerCase();
    if (/\.(png|jpe?g|gif|webp|bmp)$/.test(name)) return 'image';
    if (/\.(mp4|mov|mkv|webm|avi|mpe?g)$/.test(name)) return 'video';
    return '';
  }

  async function addFiles(files) {
    const op = selectedOperation();
    if (!op) return;
    const candidates = Array.from(files || [])
      .map((file) => ({file, kind: fileKind(file)}))
      .filter(({kind}) => (
        (kind === 'image' && op.accepts_images)
        || (kind === 'video' && op.accepts_videos)
      ));
    if (!candidates.length) {
      showError(op.accepts_videos ? '请选择受支持的视频文件。' : '请选择受支持的图片文件。');
      return;
    }
    const configuredLimit = op && op.limits && op.limits.max_reference_images;
    const maxImages = (configuredLimit || 12) + ((op.limits && op.limits.max_keyframes) || 0);
    const maxVideos = (op.limits && op.limits.max_videos) || 1;
    const maxVideoBytes = op.limits && Number(op.limits.max_video_bytes);
    for (const {file, kind} of candidates) {
      const currentKindCount = state.attachments.filter(
        (attachment) => attachment.kind === kind,
      ).length;
      if (kind === 'image' && currentKindCount >= maxImages) {
        showError(`当前模型一次最多添加 ${maxImages} 张参考图片。`);
        break;
      }
      if (kind === 'video' && currentKindCount >= maxVideos) {
        showError(`当前模型一次最多添加 ${maxVideos} 个视频。`);
        break;
      }
      if (kind === 'video' && Number.isFinite(maxVideoBytes) && file.size > maxVideoBytes) {
        showError(`视频不能超过 ${(maxVideoBytes / 1024 / 1024).toFixed(0)} MiB。`);
        continue;
      }
      const dataUrl = await readFileAsDataUrl(file);
      state.attachments.push({
        file,
        dataUrl,
        previewUrl: kind === 'video' ? URL.createObjectURL(file) : dataUrl,
        kind,
        name: file.name || `pasted-${state.attachments.length + 1}.${kind === 'video' ? 'mp4' : 'png'}`,
      });
    }
    renderAttachments();
    updateContextPreview();
    syncSendState();
  }

  function renderAttachments() {
    const imageCount = state.attachments.filter((attachment) => attachment.kind === 'image').length;
    const videoCount = state.attachments.filter((attachment) => attachment.kind === 'video').length;
    const summary = [];
    if (imageCount) summary.push(`${imageCount} 张图片`);
    if (videoCount) summary.push(`${videoCount} 个视频`);
    els.composerAttachmentSummary.textContent = summary.length ? `已附带 ${summary.join('、')}` : '';
    els.composerAttachmentSummary.hidden = !summary.length;
    els.attachmentList.replaceChildren();
    state.attachments.forEach((attachment, index) => {
      const wrap = document.createElement('div');
      wrap.className = 'attachment';
      wrap.title = attachment.name;
      let preview;
      if (attachment.kind === 'video') {
        preview = document.createElement('video');
        preview.src = attachment.previewUrl || attachment.dataUrl;
        preview.controls = true;
        preview.muted = true;
        preview.preload = 'metadata';
      } else {
        preview = document.createElement('img');
        preview.src = attachment.previewUrl || attachment.dataUrl;
        preview.alt = attachment.name;
      }
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '×';
      remove.setAttribute('aria-label', `移除 ${attachment.name}`);
      remove.addEventListener('click', () => {
        state.attachments.splice(index, 1);
        renderAttachments();
        updateContextPreview();
        syncSendState();
      });
      wrap.append(preview, remove);
      els.attachmentList.append(wrap);
    });
  }

  function clearAttachments() {
    state.attachments = [];
    els.fileInput.value = '';
    renderAttachments();
    updateContextPreview();
  }

  function clearComposerInput() {
    els.prompt.value = '';
    clearAttachments();
    syncSendState();
  }

  async function loadModelList() {
    if (state.controller) return;
    const credential = els.apiKey.value.trim();
    els.loadModels.disabled = true;
    els.model.disabled = true;
    setConnection('连接中…');
    try {
      const response = await fetch('/playground/api/models', {
        headers: authHeaders(false),
        cache: 'no-store',
      });
      if (!response.ok) throw new Error(await responseError(response));
      const data = await response.json();
      if (credential !== els.apiKey.value.trim()) return;
      if (data.schema_version !== DISCOVERY_SCHEMA_VERSION) {
        throw new Error(`不支持的模型发现格式：${data.schema_version ?? 'missing'}`);
      }
      const models = Array.isArray(data.models)
        ? data.models.filter((item) => item && item.id)
        : [];
      if (state.loadedCredential !== null && state.loadedCredential !== credential) {
        state.interactionHistories.clear();
        state.decisionForms.clear();
      }
      state.loadedCredential = credential;
      state.ocrCatalog = data.ocr || null;
      populateOcrEngines();
      $('authenticationHint').textContent = data.authentication === 'local_anonymous'
        ? '本机免密：当前本机监听与接口配置允许不填 key。此按钮仅获取列表，执行时才加载模型。'
        : data.authentication === 'trusted_proxy_anonymous'
          ? '当前通过服务端明确配置的可信代理免密访问。此按钮仅获取列表。'
          : '使用接口 API key 连接。此按钮仅获取列表，执行时才加载模型。';
      state.models.clear();
      els.model.replaceChildren();
      if (!models.length) {
        els.model.append(new Option('当前接口没有可用模型', ''));
        setConnection('无模型', 'error');
      } else {
        for (const info of models) {
          state.models.set(info.id, info);
          els.model.append(new Option(info.label || info.id, info.id));
        }
        els.model.disabled = false;
        setConnection(`${models.length} 个模型${data.authentication === 'local_anonymous' ? ' · 本机免密' : ''}`, 'ok');
      }
      syncModel();
      if (data.ocr_error) showError(data.ocr_error);
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

  function prepareChatRequest(op) {
    const modelId = els.model.value;
    const info = selectedModel();
    const interaction = interactionFor(modelId, op.id);
    const usesHistory = operationUsesHistory(op);
    const systemMessages = baseSystemMessages();
    const userMessage = userMessageForRequest(op);
    const contextLength = modelContextLength(info);
    const outputReserve = outputTokenReserve(info);
    let turns = usesHistory ? interaction.turns.slice() : [];
    let messages = [...systemMessages, ...historyMessages(turns), userMessage];
    let inputTokens = estimateMessagesTokens(messages);
    let droppedTurns = 0;
    let truncatedTokens = 0;
    let threshold = null;

    if (contextLength) {
      threshold = Math.floor(contextLength * CONTEXT_THRESHOLD_RATIO);
      const inputBudget = threshold - outputReserve;
      if (inputBudget < 32) {
        throw new Error(
          `输出预算 ${outputReserve} 已占满模型上下文阈值 ${threshold}，请调低 Max tokens。`,
        );
      }

      while (turns.length && inputTokens > inputBudget) {
        turns.shift();
        droppedTurns += 1;
        messages = [...systemMessages, ...historyMessages(turns), userMessage];
        inputTokens = estimateMessagesTokens(messages);
      }

      if (inputTokens > inputBudget) {
        const originalText = textFromContent(userMessage.content);
        const emptyUser = {
          ...userMessage,
          content: replaceContentText(userMessage.content, ''),
        };
        const fixedMessages = [...systemMessages, ...historyMessages(turns), emptyUser];
        const availableTextTokens = inputBudget - estimateMessagesTokens(fixedMessages);
        const shortened = truncateTextToTokenBudget(originalText, availableTextTokens);
        if (!shortened) {
          throw new Error(
            '当前输入和 System prompt 即使移除全部历史仍超过上下文，请缩短输入或调低 Max tokens。',
          );
        }
        truncatedTokens = Math.max(
          0,
          estimateTextTokens(originalText) - estimateTextTokens(shortened),
        );
        userMessage.content = replaceContentText(userMessage.content, shortened);
        messages = [...systemMessages, ...historyMessages(turns), userMessage];
        inputTokens = estimateMessagesTokens(messages);
        if (inputTokens > inputBudget) {
          throw new Error('当前输入无法安全压缩到模型上下文范围内，请手动缩短后重试。');
        }
      }
    }

    if (droppedTurns) interaction.turns.splice(0, droppedTurns);
    const actions = [];
    if (droppedTurns) actions.push(`已丢弃最早 ${droppedTurns} 轮对话`);
    if (truncatedTokens) actions.push(`已截短当前输入约 ${truncatedTokens} tokens`);
    interaction.lastNotice = actions.length
      ? `${actions.join('；')}。上下文按 ${Math.round(CONTEXT_THRESHOLD_RATIO * 100)}% 阈值保留安全余量。`
      : '';

    const parameters = {};
    if (op.id === 'video_analysis') {
      for (const [name, entry] of state.parameterInputs.entries()) {
        const value = readParameterValue(entry.spec, entry.input);
        if (value !== null) parameters[name] = value;
      }
    }
    const displayAttachments = state.attachments.map((attachment) => ({
      ...attachment,
      dataUrl: attachment.previewUrl || attachment.dataUrl,
    }));
    const displayUserMessage = op.id === 'video_analysis'
      ? userMessageForRequest(op, textFromContent(userMessage.content), displayAttachments)
      : userMessage;

    return {
      modelId,
      operationId: op.id,
      messages,
      userMessage,
      displayUserMessage,
      parameters,
      inputTokens,
      outputReserve,
      contextLength,
      threshold,
    };
  }

  async function runChat(op, plan) {
    const body = {model: plan.modelId, messages: plan.messages, stream: true};
    const temp = Number(els.temperature.value);
    if (Number.isFinite(temp)) body.temperature = temp;
    if (plan.outputReserve > 0) body.max_tokens = plan.outputReserve;
    Object.assign(body, plan.parameters || {});

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
    const assistantContent = els.answerText.textContent || '（模型返回了空内容）';
    const reasoning = els.reasoningText.textContent;
    const interaction = interactionFor(plan.modelId, plan.operationId);
    interaction.turns.push({
      user: plan.displayUserMessage || plan.userMessage,
      assistant: {role: 'assistant', content: assistantContent},
      reasoning,
      parameters: plan.parameters,
    });
    els.answerText.textContent = '';
    els.answer.classList.remove('active');
    els.reasoningText.textContent = '';
    els.reasoning.classList.remove('visible');
    els.reasoning.open = false;
    renderInteractionHistory();
    els.prompt.focus();
  }

  function readParameterValue(spec, input, validate = true) {
    if (spec.type === 'boolean') return input.checked;
    const raw = input.value.trim();
    if (!raw) return null;
    if (spec.type === 'resolution') {
      const match = raw.match(/^(\d+)\s*x\s*(\d+)$/i);
      if (!match) {
        if (!validate) return raw;
        throw new Error(`${spec.label || spec.name} 必须使用 WIDTHxHEIGHT 格式。`);
      }
      const width = Number(match[1]);
      const height = Number(match[2]);
      const constraints = [
        ['宽度', width, spec.min_width, spec.max_width, spec.width_modulo],
        ['高度', height, spec.min_height, spec.max_height, spec.height_modulo],
      ];
      if (validate) {
        for (const [label, value, minimum, maximum, modulo] of constraints) {
          if (minimum != null && value < Number(minimum)) {
            throw new Error(`${label}不能小于 ${minimum}。`);
          }
          if (maximum != null && value > Number(maximum)) {
            throw new Error(`${label}不能大于 ${maximum}。`);
          }
          if (modulo != null && value % Number(modulo) !== 0) {
            throw new Error(`${label}必须是 ${modulo} 的倍数。`);
          }
        }
      }
      return `${width}x${height}`;
    }
    if (!['integer', 'number'].includes(spec.type)) return raw;

    const value = Number(raw);
    const invalidInteger = spec.type === 'integer' && !Number.isInteger(value);
    if (!Number.isFinite(value) || invalidInteger) {
      if (!validate) return raw;
      throw new Error(`${spec.label || spec.name} 必须是${spec.type === 'integer' ? '整数' : '数字'}。`);
    }
    if (validate && spec.min != null && value < Number(spec.min)) {
      throw new Error(`${spec.label || spec.name} 不能小于 ${spec.min}。`);
    }
    if (validate && spec.max != null && value > Number(spec.max)) {
      throw new Error(`${spec.label || spec.name} 不能大于 ${spec.max}。`);
    }
    if (validate && spec.modulo != null
        && (value - Number(spec.offset || 0)) % Number(spec.modulo) !== 0) {
      throw new Error(
        `${spec.label || spec.name} 必须满足 (值 - ${spec.offset || 0}) % ${spec.modulo} = 0。`,
      );
    }
    return value;
  }

  function mediaPayload(op) {
    const payload = {
      model: els.model.value,
      prompt: els.prompt.value.trim(),
      response_format: 'b64_json',
    };
    for (const [name, entry] of state.parameterInputs.entries()) {
      const value = readParameterValue(entry.spec, entry.input);
      if (value !== null) payload[name] = value;
    }
    if (op.id === 'video_generation' && payload.keyframes) {
      try {
        payload.keyframes = JSON.parse(payload.keyframes);
        if (!Array.isArray(payload.keyframes)) throw new Error('Expected an array');
      } catch (error) {
        throw new Error('Keyframes must be a JSON array.');
      }
    }
    Object.assign(payload, externalPlannerPayload(op));
    if (!Object.prototype.hasOwnProperty.call(payload, 'n')) payload.n = 1;
    return payload;
  }

  function prepareMediaRequest(op) {
    const payload = mediaPayload(op);
    const runCount = operationUsesMediaQueue(op) ? mediaQueueRunCount() : 1;
    // Playground queues independent requests so every target, including video
    // workflows whose API max_batch_size is 1, gets the same multi-run behavior.
    payload.n = 1;
    const attachments = state.attachments.slice();
    const parameters = Object.fromEntries(
      Object.entries(payload).filter(([key]) => !['model', 'prompt', 'response_format'].includes(key)),
    );
    delete parameters.n;
    if (operationUsesMediaQueue(op)) parameters.generation_count = runCount;
    return {
      modelId: els.model.value,
      operationId: op.id,
      generationBatchId: (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function')
        ? globalThis.crypto.randomUUID()
        : `media-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      payload,
      attachments,
      parameters,
      runCount,
      externalPlannerToken: externalPlannerSelected(op)
        ? els.externalPlannerToken.value.trim()
        : '',
      userMessage: userMessageForRequest(op, payload.prompt, attachments),
    };
  }

  function normalizeMediaItems(items, op, runNumber = 1, runCount = 1) {
    const normalized = [];
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index] || {};
      const mime = item.mime_type || (op.id === 'video_generation' ? 'video/mp4' : 'image/png');
      const src = item.url || (item.b64_json ? `data:${mime};base64,${item.b64_json}` : '');
      if (!src) continue;
      normalized.push({
        src,
        mime,
        filename: item.filename || `${OP_LABELS[op.id] || '结果'} ${index + 1}`,
        revisedPrompt: item.revised_prompt || '',
        runNumber,
        runCount,
      });
    }
    if (!normalized.length) throw new Error('媒体接口没有返回可显示的结果');
    return normalized;
  }

  function appendMediaResultElement(parent, items, op) {
    const article = document.createElement('article');
    article.className = 'message assistant media-message';
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = 'Assistant';
    const content = document.createElement('pre');
    content.className = 'message-content';
    content.textContent = `已返回 ${items.length} 个${op && op.id === 'video_generation' ? '视频' : '图片'}结果`;
    const revisedPrompts = [...new Set(items.map((item) => item.revisedPrompt).filter(Boolean))];
    if (revisedPrompts.length) {
      content.textContent += `\n\nRevised prompt:\n${revisedPrompts.join('\n')}`;
    }
    const gallery = document.createElement('div');
    gallery.className = 'media-gallery visible';
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      const card = document.createElement('article');
      card.className = 'media-card';
      let media;
      if (item.mime.startsWith('video/')) {
        media = document.createElement('video');
        media.controls = true;
        media.preload = 'metadata';
      } else {
        media = document.createElement('img');
        media.alt = item.filename;
      }
      media.src = item.src;
      const meta = document.createElement('div');
      meta.className = 'media-meta';
      const name = document.createElement('span');
      name.textContent = item.runCount > 1
        ? `第 ${item.runNumber}/${item.runCount} 份 · ${item.filename}`
        : item.filename;
      const download = document.createElement('a');
      download.href = item.src;
      download.download = item.filename || `result-${index + 1}`;
      download.textContent = '下载';
      meta.append(name, download);
      card.append(media, meta);
      gallery.append(card);
    }
    article.append(label, content, gallery);
    parent.append(article);
  }

  function mediaPayloadForRun(plan, runIndex, reusablePrompt = '') {
    const payload = {...plan.payload, n: 1};
    payload.generation_batch_id = plan.generationBatchId;
    payload.generation_run_index = runIndex + 1;
    payload.generation_run_count = plan.runCount;
    const baseSeed = Number(payload.seed);
    if (Number.isSafeInteger(baseSeed)) payload.seed = baseSeed + runIndex;
    if (reusablePrompt && payload.prompt_mode === 'auto') {
      payload.prompt = reusablePrompt;
      payload.prompt_mode = 'raw';
    }
    return payload;
  }

  async function requestMediaRun(op, plan, payload) {
    let response;
    if (op.id === 'image_generation') {
      response = await fetch(op.endpoint, {
        method: 'POST',
        headers: plannerRequestHeaders(plan, true),
        body: JSON.stringify(payload),
        signal: state.controller.signal,
      });
    } else {
      const form = new FormData();
      for (const [key, value] of Object.entries(payload)) form.append(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
      for (const attachment of plan.attachments) {
        form.append(attachment.kind === 'video' ? 'video[]' : 'image[]', attachment.file, attachment.name);
      }
      response = await fetch(op.endpoint, {
        method: 'POST',
        headers: plannerRequestHeaders(plan, false),
        body: form,
        signal: state.controller.signal,
      });
    }
    if (!response.ok) throw new Error(await responseError(response));
    return response.json();
  }

  async function runMedia(op, plan) {
    const interaction = interactionFor(plan.modelId, plan.operationId);
    let turn = null;
    let reusablePrompt = '';
    const noun = op.id === 'video_generation'
      ? '视频'
      : (op.id === 'image_edit' ? '编辑结果' : '图片');
    for (let runIndex = 0; runIndex < plan.runCount; runIndex += 1) {
      els.answerText.textContent = runIndex
        ? `已完成 ${runIndex}/${plan.runCount} 份；正在自动排队生成第 ${runIndex + 1} 份${noun}…`
        : `正在生成第 1/${plan.runCount} 份${noun}…`;
      try {
        const payload = mediaPayloadForRun(plan, runIndex, reusablePrompt);
        const data = await requestMediaRun(op, plan, payload);
        const media = normalizeMediaItems(
          Array.isArray(data.data) ? data.data : [],
          op,
          runIndex + 1,
          plan.runCount,
        );
        if (!reusablePrompt) {
          reusablePrompt = media.find((item) => item.revisedPrompt)?.revisedPrompt || '';
        }
        if (!turn) {
          turn = {user: plan.userMessage, parameters: plan.parameters, media: []};
          interaction.turns.push(turn);
        }
        turn.media.push(...media);
        renderInteractionHistory();
      } catch (error) {
        if (error && error.name === 'AbortError') throw error;
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `第 ${runIndex + 1}/${plan.runCount} 份生成失败，队列已停止；`
          + `已完成 ${runIndex} 份：${message}`
        );
      }
    }
    els.answerText.textContent = '';
    els.answer.classList.remove('active');
    renderInteractionHistory();
    els.prompt.focus();
  }

  async function runContextIR(op, plan) {
    const form = new FormData();
    for (const [key, value] of Object.entries(plan.payload)) {
      form.append(key, String(value));
    }
    for (const attachment of plan.attachments) {
      form.append('image[]', attachment.file, attachment.name);
    }
    const response = await fetch(op.endpoint, {
      method: 'POST',
      headers: plannerRequestHeaders(plan, false),
      body: form,
      signal: state.controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    const enhancedPrompt = data && data.content && data.content.prompt;
    if (!enhancedPrompt) {
      throw new Error('H3 Context-IR 接口没有返回 content.prompt');
    }
    const diagnostics = {
      provider: data.provider,
      mode: data.mode,
      duration_seconds: data.duration_seconds,
      fallback: data.fallback,
      attempts: data.attempts,
      warnings: data.warnings,
      usage: data.usage,
      ir: data.ir,
    };
    const interaction = interactionFor(plan.modelId, plan.operationId);
    interaction.turns.push({
      user: plan.userMessage,
      assistant: {role: 'assistant', content: enhancedPrompt},
      reasoning: JSON.stringify(diagnostics, null, 2),
      parameters: plan.parameters,
    });
    els.answerText.textContent = '';
    els.answer.classList.remove('active');
    renderInteractionHistory();
    els.prompt.focus();
  }

  const DECISION_QUESTION_LABELS = {
    predicate: '布尔概率 · P(true)',
    choice: '选项概率 · Choice',
    score: '有序等级 · Ordinal score',
  };

  function decisionForm(op = selectedOperation()) {
    if (!op || op.id !== 'decision') return null;
    const key = els.model.value;
    let form = state.decisionForms.get(key);
    if (!form) {
      const types = op.question_types || [];
      form = {
        dialect: op.dialect || 'systemone', stateFormat: 'text', thinkMode: '',
        questions: [{name: 'question_1', type: types[0] || 'predicate', instructions: '', criteria: ''}],
      };
      state.decisionForms.set(key, form);
    }
    return form;
  }

  function decisionDialects(op) {
    return Array.isArray(op.dialects) ? op.dialects : [];
  }

  function syncDecisionForm(op) {
    const visible = Boolean(op && op.id === 'decision');
    els.decisionSection.hidden = !visible;
    els.decisionStateLabel.hidden = !visible;
    if (!visible) return;
    const form = decisionForm(op);
    const dialects = decisionDialects(op);
    if (!dialects.some((item) => item.id === form.dialect)) form.dialect = dialects[0]?.id || '';
    els.decisionDialect.replaceChildren();
    for (const dialect of dialects) {
      els.decisionDialect.append(new Option(dialect.label || dialect.id, dialect.id));
    }
    els.decisionDialect.value = form.dialect;
    els.decisionStateFormat.value = form.stateFormat;
    const dialect = dialects.find((item) => item.id === form.dialect);
    els.decisionDialectHint.textContent = dialect
      ? `${dialect.endpoint} · 两种格式使用各自的请求与响应结构。`
      : '没有可用的决策 API 格式。';
    const types = op.question_types || [];
    const modalities = op.modalities || ['text'];
    const hints = [
      `问题：${types.map((type) => DECISION_QUESTION_LABELS[type] || type).join(' / ') || '未声明'}`,
      `输入：${modalities.join(' / ')}`,
    ];
    if (modalities.includes('image') && form.dialect !== op.image_dialect) {
      hints.push('图片需要切换到后端原生 API 格式');
    }
    if (op.image_format && modalities.includes('image')) hints.push(`图片格式：${op.image_format}`);
    if (op.runtime) hints.push(`运行方式：${op.runtime}`);
    if (op.version) hints.push(`版本：${op.version}`);
    if (op.status) hints.push(`端点：${op.endpoint_verified ? '已验证' : '尚未验证'}`);
    if (Array.isArray(op.warnings) && op.warnings.length) hints.push(op.warnings.join('；'));
    els.decisionCapabilityHint.textContent = hints.join(' · ');
    const thinkModes = Array.isArray(op.think_modes) ? op.think_modes : [];
    const supportsThink = (op.extensions || []).includes('think_mode') && thinkModes.length > 0;
    els.decisionThinkField.hidden = !supportsThink;
    els.decisionThinkMode.replaceChildren(new Option('后端默认（不发送）', ''));
    if (supportsThink) {
      for (const mode of thinkModes) els.decisionThinkMode.append(new Option(mode, mode));
    }
    if (!supportsThink || (form.thinkMode && !thinkModes.includes(form.thinkMode))) form.thinkMode = '';
    els.decisionThinkMode.value = form.thinkMode;
    renderDecisionQuestions(op);
    updateDecisionPreview();
  }

  function decisionField(labelText, input) {
    const field = document.createElement('div');
    field.className = 'field';
    const label = document.createElement('label');
    label.htmlFor = input.id;
    label.textContent = labelText;
    field.append(label, input);
    return field;
  }

  function renderDecisionQuestions(op = selectedOperation()) {
    const form = decisionForm(op);
    if (!form) return;
    els.decisionQuestionList.replaceChildren();
    const types = op.question_types || [];
    form.questions.forEach((question, index) => {
      const group = document.createElement('fieldset');
      group.className = 'decision-question';
      const legend = document.createElement('legend');
      legend.textContent = `问题 ${index + 1}`;
      group.append(legend);
      const head = document.createElement('div');
      head.className = 'decision-question-head';
      const name = document.createElement('input');
      name.id = `decision-question-${index}-name`;
      name.className = 'control';
      name.value = question.name;
      name.placeholder = '问题名称（可留空）';
      name.setAttribute('aria-label', `问题 ${index + 1} 名称`);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'secondary decision-remove';
      remove.textContent = '移除';
      remove.disabled = form.questions.length <= 1;
      remove.setAttribute('aria-label', `移除问题 ${index + 1}`);
      remove.addEventListener('click', () => {
        form.questions.splice(index, 1);
        renderDecisionQuestions(op);
        updateDecisionPreview();
      });
      head.append(name, remove);
      const type = document.createElement('select');
      type.id = `decision-question-${index}-type`;
      for (const value of types) type.append(new Option(DECISION_QUESTION_LABELS[value] || value, value));
      type.value = question.type;
      // Keep unsupported saved questions visible, rather than changing their meaning silently.
      if (!types.includes(question.type)) {
        type.append(new Option(`${question.type}（当前后端未声明支持）`, question.type));
        type.value = question.type;
      }
      const instructions = document.createElement('textarea');
      instructions.id = `decision-question-${index}-instructions`;
      instructions.rows = 2;
      instructions.value = question.instructions;
      instructions.placeholder = '要根据当前状态判断什么？';
      group.append(head, decisionField('类型', type), decisionField('问题 / Instructions', instructions));
      const criteria = document.createElement('textarea');
      criteria.id = `decision-question-${index}-criteria`;
      criteria.rows = 3;
      criteria.value = question.criteria;
      criteria.placeholder = question.type === 'score' ? '低\n中\n高' : '接受\n拒绝';
      const criteriaField = decisionField(
        question.type === 'score' ? '等级（每行一个，由低到高；索引从 0 开始）' : '选项（每行一个）',
        criteria,
      );
      criteriaField.hidden = question.type === 'predicate';
      group.append(criteriaField);
      for (const [input, key] of [[name, 'name'], [instructions, 'instructions'], [criteria, 'criteria']]) {
        input.addEventListener('input', () => {
          question[key] = input.value;
          updateDecisionPreview();
        });
      }
      type.addEventListener('change', () => {
        question.type = type.value;
        renderDecisionQuestions(op);
        updateDecisionPreview();
      });
      els.decisionQuestionList.append(group);
    });
  }

  function prepareDecisionRequest(op) {
    const form = decisionForm(op);
    const dialect = decisionDialects(op).find((item) => item.id === form.dialect);
    if (!dialect) throw new Error('请选择受支持的决策 API 格式。');
    let context = els.prompt.value.trim();
    if (form.stateFormat === 'json') {
      try { context = JSON.parse(context); } catch (_) { throw new Error('State 不是有效的 JSON。'); }
      if (context === null || typeof context !== 'object') throw new Error('JSON State 必须是对象或数组。');
    }
    if (!form.questions.length) throw new Error('至少添加一个问题。');
    const seen = new Set();
    const questions = form.questions.map((question, index) => {
      const name = question.name.trim();
      if (form.dialect === 'systemone' && seen.has(name)) throw new Error(`SystemOne 问题 ${index + 1} 需要唯一名称。`);
      seen.add(name);
      if (!(op.question_types || []).includes(question.type)) throw new Error(`后端没有声明支持问题类型 ${question.type}。`);
      const instructions = question.instructions.trim();
      const criteria = question.criteria.split('\n').map((line) => line.trim()).filter(Boolean);
      const minimum = question.type === 'choice' && form.dialect === 'systemone' ? 1 : 2;
      if (question.type !== 'predicate' && criteria.length < minimum) throw new Error(`问题 ${index + 1} 至少需要 ${minimum} 个选项或等级。`);
      if (question.type === 'choice' && new Set(criteria).size !== criteria.length) throw new Error(`问题 ${index + 1} 的选项不能重复。`);
      return {name, type: question.type, instructions, criteria};
    });
    const images = state.attachments.filter((attachment) => attachment.kind === 'image');
    if (images.length && !op.accepts_images) throw new Error('当前 API 格式没有声明图片支持。请切回后端原生格式或移除图片；不支持跨格式图片转换。');
    let payload;
    const extraHeaders = {};
    if (form.dialect === 'systemone') {
      payload = {model: els.model.value, state: context, questions: Object.fromEntries(questions.map((question) => {
        const value = {type: question.type === 'predicate' ? 'noul' : question.type, instructions: question.instructions};
        if (question.type === 'choice') value.criteria = Object.fromEntries(question.criteria.map((label) => [label, null]));
        if (question.type === 'score') value.criteria = question.criteria;
        return [question.name, value];
      }))};
      if (images.length && op.image_format === 'startlux_images') {
        payload.images = images.map((attachment) => attachment.dataUrl);
      } else if (images.length && op.image_format === 'typesafe_assets') {
        const assets = {};
        images.forEach((attachment, index) => {
          const match = /^data:image\/(png|jpe?g);base64,(.+)$/i.exec(attachment.dataUrl);
          if (!match) throw new Error('TypeSafe 图片仅支持 PNG 和 JPEG。');
          const reference = `playground_image_${index + 1}`;
          assets[reference] = {type: 'image', format: match[1].toLowerCase(), data_base64: match[2]};
        });
        payload.assets = assets;
        payload.state = {context, images: Object.keys(assets)};
        extraHeaders['TypeSafe-Danger-Zone-Images'] = 'true';
      }
    } else if (form.dialect === 'openai_decisions') {
      const text = typeof context === 'string' ? context : JSON.stringify(context);
      payload = {
        model: els.model.value,
        input: images.length ? [{role: 'user', content: [
          {type: 'input_text', text},
          ...images.map((attachment) => ({type: 'input_image', image_url: attachment.dataUrl})),
        ]}] : text,
        questions: questions.map((question) => {
          const value = {type: question.type, name: question.name, instructions: question.instructions};
          if (question.type === 'choice') value.choices = question.criteria.map((label) => ({value: label}));
          if (question.type === 'score') value.levels = question.criteria.map((label) => ({label}));
          return value;
        }),
      };
    } else {
      throw new Error(`不支持的决策 API 格式：${form.dialect}`);
    }
    if (form.thinkMode) {
      if (!(op.extensions || []).includes('think_mode') || !(op.think_modes || []).includes(form.thinkMode)) {
        throw new Error('当前后端没有声明支持此 Think mode。');
      }
      payload.think_mode = form.thinkMode;
    }
    return {
      modelId: els.model.value, operationId: op.id, dialect: form.dialect,
      endpoint: dialect.endpoint, payload, extraHeaders, questions,
      userMessage: {role: 'user', content: typeof context === 'string' ? context : JSON.stringify(context, null, 2)},
    };
  }

  function updateDecisionPreview() {
    const op = selectedOperation();
    if (!op || op.id !== 'decision') return;
    try {
      const plan = prepareDecisionRequest(op);
      const headerLines = Object.entries(plan.extraHeaders).map(([key, value]) => `${key}: ${value}`).join('\n');
      els.decisionRequestPreview.textContent = `POST ${plan.endpoint}\n${headerLines ? `${headerLines}\n` : ''}${JSON.stringify(plan.payload, null, 2)}`;
    } catch (error) {
      els.decisionRequestPreview.textContent = error instanceof Error ? error.message : String(error);
    }
  }

  function appendDecisionRaw(parent, label, value) {
    const details = document.createElement('details');
    details.className = 'decision-raw';
    const summary = document.createElement('summary');
    summary.textContent = label;
    const pre = document.createElement('pre');
    pre.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    details.append(summary, pre);
    parent.append(details);
  }

  function appendDecisionProbabilities(parent, rows) {
    if (!rows.length) return;
    const table = document.createElement('table');
    table.className = 'decision-probabilities';
    const head = document.createElement('thead');
    const header = document.createElement('tr');
    for (const title of ['选项 / 等级', '后端返回概率']) {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = title;
      header.append(cell);
    }
    head.append(header);
    const body = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      const label = document.createElement('td');
      label.textContent = String(row.label);
      const value = document.createElement('td');
      const cell = document.createElement('div');
      cell.className = 'decision-probability-cell';
      const number = document.createElement('span');
      if (typeof row.probability === 'number' && Number.isFinite(row.probability) && row.probability >= 0 && row.probability <= 1) {
        const bar = document.createElement('progress');
        bar.max = 1;
        bar.value = row.probability;
        bar.setAttribute('aria-label', `${row.label} 概率`);
        number.textContent = `${(row.probability * 100).toFixed(2)}%`;
        number.title = String(row.probability);
        cell.append(bar);
      } else {
        number.textContent = '未提供';
      }
      cell.append(number);
      value.append(cell);
      tr.append(label, value);
      body.append(tr);
    }
    table.append(head, body);
    parent.append(table);
  }

  function appendDecisionResultElement(parent, result) {
    const article = document.createElement('article');
    article.className = 'message assistant decision-result';
    const response = result.response;
    const answers = response && response.answers;
    const entries = result.dialect === 'systemone'
      ? Object.entries(answers && !Array.isArray(answers) ? answers : {})
      : (Array.isArray(answers) ? answers : []).map((answer, index) => [answer.name || result.questions[index]?.name || `问题 ${index + 1}`, answer]);
    entries.forEach(([name, answer], index) => {
      if (!answer || typeof answer !== 'object') return;
      const section = document.createElement('section');
      section.className = 'decision-answer';
      const heading = document.createElement('h3');
      heading.textContent = name || `问题 ${index + 1}`;
      section.append(heading);
      const question = result.dialect === 'systemone'
        ? result.questions.find((item) => item.name === name) : result.questions[index];
      const instructions = document.createElement('p');
      instructions.textContent = question?.instructions || '';
      section.append(instructions);
      let rows = [];
      const summary = document.createElement('p');
      if (['noul', 'predicate'].includes(answer.type)) {
        rows = [{label: 'P(true)', probability: answer.type === 'noul' ? answer.noul : answer.probability}];
      } else if (answer.type === 'choice' || answer.type === 'score') {
        if (Array.isArray(answer.probabilities)) {
          rows = answer.probabilities.map((item) => ({label: item.label ?? item.value, probability: item.probability}));
        } else if (answer.probabilities && typeof answer.probabilities === 'object') {
          rows = Object.entries(answer.probabilities).map(([value, probability]) => ({
            label: answer.legend?.[value] ?? value, probability,
          }));
        }
        if (answer.type === 'score' && typeof answer.score === 'number') {
          summary.textContent = `期望等级分数（从 0 开始）：${answer.score}`;
        } else if (answer.type === 'choice' && answer.choice != null) {
          summary.textContent = `选择：${String(answer.choice)}`;
        } else if (answer.type === 'choice' && answer.choice === null) {
          summary.textContent = '未选择任何列出的选项（none-of-the-above）。';
        }
      } else if (answer.type === 'refusal') {
        summary.textContent = '后端拒绝回答此问题。';
      }
      if (summary.textContent) section.append(summary);
      appendDecisionProbabilities(section, rows);
      if (!rows.length && !summary.textContent) {
        const unknown = document.createElement('p');
        unknown.textContent = '后端未返回可展示的概率或评分；请查看原始响应。';
        section.append(unknown);
      }
      if (answer.none_probability != null) {
        const alarm = document.createElement('p');
        alarm.className = 'decision-result-meta decision-none-alarm';
        alarm.textContent = `后端 none_probability 警报分数：${String(answer.none_probability)}（未校准；独立于列出选项的归一化概率分布）`;
        section.append(alarm);
      }
      if (answer.confidence != null) {
        const confidence = document.createElement('p');
        confidence.className = 'decision-result-meta';
        confidence.textContent = `后端 confidence：${String(answer.confidence)}（按后端语义，不等同于校准概率）`;
        section.append(confidence);
      }
      article.append(section);
    });
    if (result.error) {
      const error = document.createElement('div');
      error.className = 'decision-error';
      error.textContent = result.error;
      article.append(error);
    }
    const meta = document.createElement('div');
    meta.className = 'decision-result-meta';
    const facts = [`POST ${result.endpoint}`, result.elapsedMs != null ? `耗时 ${(result.elapsedMs / 1000).toFixed(2)}s` : '请求中…'];
    if (result.status) facts.push(`HTTP ${result.status}`);
    meta.textContent = facts.join(' · ');
    article.append(meta);
    if (response && response.usage != null) appendDecisionRaw(article, 'Usage（后端返回）', response.usage);
    if (response && response.decision_metadata != null) appendDecisionRaw(article, '来源与概率语义', response.decision_metadata);
    appendDecisionRaw(article, '原始请求（不含认证信息）', result.request);
    if (result.response !== undefined) appendDecisionRaw(article, '原始响应', result.response);
    parent.append(article);
  }

  async function runDecision(op, plan) {
    const result = {
      dialect: plan.dialect, endpoint: plan.endpoint, request: plan.payload,
      questions: plan.questions, elapsedMs: null,
    };
    const interaction = interactionFor(plan.modelId, plan.operationId);
    interaction.turns.push({user: plan.userMessage, decision: result});
    renderInteractionHistory();
    try {
      const response = await fetch(plan.endpoint, {
        method: 'POST', headers: {...authHeaders(true), ...plan.extraHeaders},
        body: JSON.stringify(plan.payload), signal: state.controller.signal,
      });
      result.status = response.status;
      const text = await response.text();
      try { result.response = JSON.parse(text); } catch (_) { result.response = text; }
      if (!response.ok) throw new Error(formatError(result.response, `${response.status} ${response.statusText}`));
      if (!result.response || typeof result.response !== 'object' || !result.response.answers) {
        throw new Error('决策接口没有返回 answers；原始响应已保留。');
      }
    } catch (error) {
      result.error = error && error.name === 'AbortError' ? '请求已停止。' : (error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      result.elapsedMs = performance.now() - state.startedAt;
      els.answerText.textContent = '';
      els.answer.classList.remove('active');
      renderInteractionHistory();
    }
  }

  function populateOcrEngines() {
    $('ocrEngines').replaceChildren();
    for (const engine of state.ocrCatalog?.engines || []) {
      const label = document.createElement('label'), check = document.createElement('input');
      check.type = 'checkbox'; check.value = engine.id;
      check.checked = engine.available && !engine.id.endsWith('-dflash');
      check.dataset.unavailable = String(!engine.available);
      label.append(check, document.createTextNode(engine.label + (engine.available ? '' : '（文件缺失）')));
      $('ocrEngines').append(label);
    }
    const previousVerifier = $('ocrVerifier').value;
    $('ocrVerifier').replaceChildren(new Option('不复核', ''));
    for (const verifier of state.ocrCatalog?.verifiers || []) {
      const option = new Option(verifier.label + (verifier.available ? '' : '（不可用）'), verifier.id);
      option.disabled = !verifier.available;
      $('ocrVerifier').append(option);
    }
    $('ocrVerifier').value = (state.ocrCatalog?.verifiers || []).some(v => v.id === previousVerifier && v.available) ? previousVerifier : '';
    state.ocrRuntime = null;
    refreshOcrRuntime();
  }

  function syncOcrForm(op) {
    const mode = op?.ocr_mode;
    $('ocrSection').hidden = !mode;
    if (!mode) return;
    $('ocrEnginesField').hidden = mode !== 'compare';
    $('ocrFallbackField').hidden = mode !== 'auto';
    $('ocrCandidatesField').hidden = mode !== 'score';
    const previous = $('ocrFallback').value;
    const alternatives = (state.ocrCatalog?.engines || []).filter(e => e.available && e.id !== selectedModel()?.ocr_engine);
    $('ocrFallback').replaceChildren(...alternatives.map(e => new Option(e.label, e.id)));
    $('ocrFallback').value = alternatives.find(e => e.id === previous)?.id
      || alternatives.find(e => e.id === 'hunyuan-ocr-current')?.id || alternatives[0]?.id || '';
    els.resultTitle.textContent = 'OCR 识别与复核记录';
    els.uploadLabel.textContent = '待识别 / 复核区域';
    els.uploadRequirement.textContent = '需要 1 张图片';
    els.fileInput.accept = 'image/png,image/jpeg,image/webp';
    els.dropHint.textContent = '一个文字、公式或表格区域 · PNG / JPEG / WebP · 最大 12 MiB';
    els.prompt.placeholder = '可选备注（仅保留在本页记录，不作为 OCR 指令）';
    els.send.textContent = '运行 ' + OP_LABELS[op.id] + ' →';
    els.operationHint.textContent = mode === 'auto'
      ? '识别 → 所选模型复核 → 低于阈值或识别失败时用备用引擎重试一次'
      : mode === 'score' ? '将候选文本渲染后，与上传图片比较；不执行 OCR 识别'
        : mode === 'compare' ? '依次运行选中引擎，保留全部识别结果与可选复核评分'
          : '运行选中的 OCR 模型，可选复核；不会自动重试';
  }

  function addOcrCandidate() {
    const list = $('ocrCandidates');
    if (list.children.length >= 8) return;
    const row = document.createElement('div'); row.className = 'ocr-candidate';
    const input = document.createElement('textarea'); input.maxLength = 24000; input.rows = 3;
    input.placeholder = '粘贴待复核的识别结果'; input.setAttribute('aria-label', '候选文本');
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'secondary'; remove.textContent = '删除';
    remove.onclick = () => { row.remove(); syncSendState(); };
    row.append(input, remove); list.append(row); syncSendState();
  }

  function prepareOcrRequest(op) {
    const mode = op.ocr_mode, engine = selectedModel()?.ocr_engine;
    const engines = mode === 'compare'
      ? [...$('ocrEngines').querySelectorAll('input:checked')].map(e => e.value)
      : mode === 'auto' ? [engine, $('ocrFallback').value]
        : mode === 'score' ? [] : [engine];
    if (mode !== 'score' && (!engines.length || engines.some(e => !e))) throw new Error('请选择可用 OCR 引擎。');
    const verifier = $('ocrVerifier').value || null;
    if (['auto', 'score'].includes(mode) && !verifier) throw new Error('此模式需要选择复核模型。');
    if (verifier && !state.ocrCatalog?.verifiers.some(v => v.id === verifier && v.available)) throw new Error('所选复核模型不可用。');
    if (mode === 'auto' && engines[0] === engines[1]) throw new Error('自动重试需要两个不同引擎。');
    const attachment = state.attachments.find(e => e.kind === 'image');
    if (!attachment || !/^data:image\/(png|jpeg|webp);base64,/.test(attachment.dataUrl)) throw new Error('请上传 PNG、JPEG 或 WebP 图片。');
    if (attachment.file.size > 12 * 1024 * 1024) throw new Error('图片超过 12 MiB，请先裁剪。');
    const candidates = mode === 'score' ? [...$('ocrCandidates').querySelectorAll('textarea')].map(e => e.value.trim()).filter(Boolean) : [];
    if (mode === 'score' && !candidates.length) throw new Error('请填写至少一个候选文本。');
    const threshold = Number($('ocrThreshold').value), maxTokens = Number($('ocrMaxTokens').value);
    const repeat = Number($('ocrRepeat').value);
    if (!Number.isInteger(repeat) || repeat < 1 || repeat > 20) throw new Error('连续运行次数须为 1–20 的整数。');
    if (verifier && (!$('ocrThreshold').value || !Number.isFinite(threshold) || threshold < 0 || threshold > 1)) throw new Error('复核阈值须在 0–1 之间。');
    if (!Number.isInteger(maxTokens) || maxTokens < 64 || maxTokens > 8192) throw new Error('Max tokens 须为 64–8192 的整数。');
    const payload = {image: attachment.dataUrl, mode, engines, candidates,
      category: $('ocrCategory').value, verifier,
      threshold: verifier ? threshold : 0.9, max_tokens: maxTokens, keep_loaded: $('ocrLifetime').value === 'keep'};
    const note = els.prompt.value.trim();
    return {modelId: els.model.value, operationId: op.id, payload, runCount: repeat,
      userMessage: userMessageForRequest(op, OP_LABELS[op.id] + (note ? `\n备注：${note}` : ''))};
  }

  function appendOcrResult(parent, result) {
    const wrap = document.createElement('div'); wrap.className = 'ocr-results';
    const summary = document.createElement('p'); summary.className = 'ocr-summary';
    const decisions = {accept: '达到设定阈值', review: '需要人工核对', unverified: '未复核'};
    const verifierLabel = result.verifier_label || result.verifier;
    summary.textContent = `${decisions[result.decision] || result.decision} · 总耗时 ${result.elapsed_seconds}s · 重试 ${result.retry_count} 次。`
      + (verifierLabel ? `复核模型：${verifierLabel}；阈值 ${result.threshold}，评分仅针对当前区域。` : '未使用复核模型。');
    summary.textContent += result.keep_loaded ? ' 模型已保留。' : ' 模型已释放。';
    if (result.browser_seconds != null) summary.textContent += ` 本轮往返 ${result.browser_seconds.toFixed(3)}s。`;
    if (result.runtime_events?.length) {
      const loading = result.runtime_events.reduce((sum, e) => sum + (e.load_seconds || 0), 0);
      summary.textContent += ` 模型加载合计 ${loading.toFixed(3)}s，复用 ${result.runtime_events.filter(e => e.reused).length} 次。`;
    }
    if (result.benchmark_summary) summary.textContent += ' ' + result.benchmark_summary;
    wrap.append(summary);
    for (const [index, row] of result.results.entries()) {
      const card = document.createElement('article'); card.className = 'ocr-result' + (index === result.best_index ? ' ocr-best' : '');
      const title = document.createElement('strong'); title.textContent = state.ocrCatalog?.engines.find(e => e.id === row.engine)?.label || row.engine;
      const score = document.createElement('div'); score.className = 'ocr-score';
      score.textContent = row.score == null ? (result.verifier ? `未获得有效评分 · ${row.verdict}` : '未复核') : `复核分 ${row.score.toFixed(4)}${index === result.best_index ? ' · 当前最高分' : ''}`;
      const meta = document.createElement('p'); meta.className = 'help';
      meta.textContent = `${row.model_reused ? '复用已加载模型' : '首次 / 重新加载'} · 加载 ${row.load_seconds ?? '—'}s · 识别阶段 ${row.elapsed_seconds ?? '—'}s · 推理 ${row.inference_seconds ?? '—'}s${row.truncated ? ' · 输出截断，未接受' : ''}`;
      const text = document.createElement('pre'); text.textContent = row.text || '';
      card.append(title, score, meta, text);
      if (row.error || row.verification_error) {
        const error = document.createElement('p'); error.className = 'ocr-failure'; error.textContent = row.error || row.verification_error; card.append(error);
      }
      const copy = document.createElement('button'); copy.type = 'button'; copy.className = 'secondary'; copy.textContent = '复制识别文本';
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(row.text || ''); copy.textContent = '已复制 ✓'; }
        catch (e) { showError('复制失败：' + e.message); }
      };
      card.append(copy); wrap.append(card);
    }
    appendDecisionRaw(wrap, '原始结果与运行信息', result);
    parent.append(wrap);
  }

  async function runOcr(op, plan) {
    const rounds = [];
    $('ocrBenchmark').hidden = false;
    $('ocrBenchmark').textContent = `连续测试 0/${plan.runCount}`;
    try {
      for (let i = 0; i < plan.runCount; i++) {
        if (state.controller.signal.aborted) throw new DOMException('Stopped', 'AbortError');
        els.answer.classList.add('active');
        els.answerText.textContent = `正在识别 / 复核，第 ${i + 1}/${plan.runCount} 轮…`;
        const start = performance.now();
        const response = await fetch(op.endpoint, {method: 'POST', headers: authHeaders(true),
          body: JSON.stringify(plan.payload), signal: state.controller.signal});
        if (!response.ok) throw new Error(await responseError(response));
        const result = await response.json();
        if (state.requestDiscarded) return;
        result.browser_seconds = (performance.now() - start) / 1000;
        result.round = i + 1;
        rounds.push(result);
        const later = rounds.slice(1);
        const avg = later.length ? later.reduce((sum, r) => sum + r.browser_seconds, 0) / later.length : null;
        const failed = result.results.some(r => r.error || r.verification_error);
        const summary = `完成 ${rounds.length}/${plan.runCount} 轮 · 首轮往返 ${rounds[0].browser_seconds.toFixed(3)}s`
          + (avg == null ? '' : ` · 后续平均往返 ${avg.toFixed(3)}s`)
          + (failed ? ' · 出错，已停止后续轮次' : '');
        result.benchmark_summary = summary;
        $('ocrBenchmark').textContent = summary;
        state.ocrRuntime = {worker_running: result.keep_loaded, loaded_models: result.loaded_models || [], busy: false};
        renderOcrRuntime();
        interactionFor(plan.modelId, plan.operationId).turns.push({user: plan.userMessage, ocr: result});
        els.answerText.textContent = ''; els.answer.classList.remove('active');
        renderInteractionHistory();
        if (failed) break;
      }
    } finally {
      // Cancellation may race the server's disconnect cleanup. Explicit release
      // also handles stopping between rounds when there is no pending HTTP call.
      if (state.controller?.signal.aborted) {
        await releaseOcrRuntime(true);
        if (state.requestDiscarded) $('ocrBenchmark').hidden = true;
      }
      await refreshOcrRuntime();
    }
  }

  function renderOcrRuntime() {
    const value = state.ocrRuntime;
    $('ocrRuntimeStatus').textContent = value ? (value.worker_running
      ? `已保留 ${(value.loaded_models || []).length} 个模型：${(value.loaded_models || []).join('、')}${value.busy ? '（运行中）' : ''}`
      : '当前无保留模型') : '模型状态尚未获取';
  }

  async function refreshOcrRuntime() {
    if (!state.ocrCatalog || state.loadedCredential !== els.apiKey.value.trim()) return;
    const credential = els.apiKey.value.trim();
    try {
      const response = await fetch('/playground/api/ocr/models', {headers: authHeaders(), cache: 'no-store'});
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json();
      if (credential !== els.apiKey.value.trim()) return;
      state.ocrRuntime = result;
      renderOcrRuntime(); syncSendState();
    } catch (e) {
      state.ocrRuntime = null;
      $('ocrRuntimeStatus').textContent = '状态获取失败：' + e.message;
      syncSendState();
    }
  }

  async function releaseOcrRuntime(afterCancel = false) {
    state.ocrReleasing = true; syncSendState();
    try {
      for (let attempt = 0; attempt < (afterCancel ? 30 : 1); attempt++) {
        const response = await fetch('/playground/api/ocr/release', {method: 'POST', headers: authHeaders()});
        if (response.status === 409 && afterCancel) { await new Promise(resolve => setTimeout(resolve, 200)); continue; }
        if (!response.ok) throw new Error(await responseError(response));
        state.ocrRuntime = await response.json(); renderOcrRuntime(); return;
      }
      throw new Error('任务清理仍在进行，请稍后刷新状态并手动释放。');
    } catch (e) {
      if (afterCancel) throw new Error('请求已停止，但释放模型失败：' + e.message);
      showError('释放模型失败：' + e.message);
    }
    finally { state.ocrReleasing = false; syncSendState(); }
  }

  $('ocrRelease').addEventListener('click', () => releaseOcrRuntime());
  $('ocrRefreshRuntime').addEventListener('click', refreshOcrRuntime);

  $('ocrAddCandidate').addEventListener('click', addOcrCandidate);
  $('ocrSection').addEventListener('input', syncSendState);
  $('ocrSection').addEventListener('change', syncSendState);
  addOcrCandidate();

  async function runRequest() {
    const op = selectedOperation();
    if (op?.ocr_mode && state.ocrReleasing) return;
    if (state.controller || !op || (op.id !== 'decision' && !op.ocr_mode && !els.prompt.value.trim())) return;
    if (state.loadedCredential !== els.apiKey.value.trim()) return;
    const imageCount = state.attachments.filter(
      (attachment) => attachment.kind === 'image',
    ).length;
    const videoCount = state.attachments.filter(
      (attachment) => attachment.kind === 'video',
    ).length;
    if (op.requires_images && !imageCount) {
      showError('当前能力至少需要一张参考图片。');
      return;
    }
    if (op.requires_videos && !videoCount) {
      showError('当前能力至少需要一个视频。');
      return;
    }
    const maxReferences = op.limits && op.limits.max_reference_images;
    if (maxReferences && imageCount > Number(maxReferences) + Number(op.limits.max_keyframes || 0)) {
      showError(`当前能力最多接收 ${maxReferences} 张参考图片，请先移除多余图片。`);
      return;
    }
    const maxVideos = op.limits && op.limits.max_videos;
    if (maxVideos && videoCount > Number(maxVideos)) {
      showError(`当前能力最多接收 ${maxVideos} 个视频，请先移除多余视频。`);
      return;
    }
    let plan;
    try {
      plan = operationUsesChatEndpoint(op)
        ? prepareChatRequest(op)
        : (op.ocr_mode ? prepareOcrRequest(op) : (op.id === 'decision' ? prepareDecisionRequest(op) : prepareMediaRequest(op)));
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
      return;
    }

    state.controller = new AbortController();
    state.requestDiscarded = false;
    state.activePlannerChoice = effectivePlannerChoice(op);
    state.activePlannerHasImages = imageCount > 0;
    state.startedAt = performance.now();
    state.firstTokenAt = 0;
    beginRequest(op, plan);
    if (op.id !== 'decision' && !op.ocr_mode) clearComposerInput();
    els.prompt.focus();
    syncSendState();
    state.timer = window.setInterval(() => {
      els.timeMetric.textContent = `耗时 ${((performance.now() - state.startedAt) / 1000).toFixed(1)}s`;
    }, 100);
    try {
      if (operationUsesChatEndpoint(op)) await runChat(op, plan);
      else if (op.ocr_mode) await runOcr(op, plan);
      else if (op.id === 'decision') await runDecision(op, plan);
      else if (op.id === 'h3_context_ir') await runContextIR(op, plan);
      else await runMedia(op, plan);
    } catch (error) {
      if (state.requestDiscarded && (!error || error.name === 'AbortError')) {
        // Clear cancels this decision without repopulating the cleared view.
      } else if (error && error.name === 'AbortError') {
        showRequestError(
          op,
          plan && plan.runCount > 1
            ? (op.ocr_mode ? '连续识别已停止；已完成结果已保留。' : '生成队列已停止；已完成结果已保留。')
            : '请求已停止。',
        );
      }
      else showRequestError(op, error instanceof Error ? error.message : String(error));
    } finally {
      els.timeMetric.textContent = `耗时 ${((performance.now() - state.startedAt) / 1000).toFixed(2)}s`;
      finishRequest();
    }
  }

  els.decisionAddQuestion.addEventListener('click', () => {
    const op = selectedOperation();
    const form = decisionForm(op);
    if (!form || state.controller) return;
    let number = form.questions.length + 1;
    while (form.questions.some((question) => question.name === `question_${number}`)) number += 1;
    form.questions.push({name: `question_${number}`, type: (op.question_types || [])[0] || 'predicate', instructions: '', criteria: ''});
    renderDecisionQuestions(op);
    updateDecisionPreview();
  });
  els.decisionDialect.addEventListener('change', () => {
    const form = decisionForm();
    if (!form) return;
    form.dialect = els.decisionDialect.value;
    syncOperation();
  });
  els.decisionStateFormat.addEventListener('change', () => {
    const form = decisionForm();
    if (form) form.stateFormat = els.decisionStateFormat.value;
    updateDecisionPreview();
  });
  els.decisionThinkMode.addEventListener('change', () => {
    const form = decisionForm();
    if (form) form.thinkMode = els.decisionThinkMode.value;
    updateDecisionPreview();
  });

  els.toggleKey.addEventListener('click', () => {
    const showing = els.apiKey.type === 'text';
    els.apiKey.type = showing ? 'password' : 'text';
    els.toggleKey.setAttribute('aria-label', showing ? '显示 API key' : '隐藏 API key');
  });
  els.apiKey.addEventListener('input', () => {
    setConnection('凭据已变化');
    syncSendState();
  });
  els.toggleExternalPlannerToken.addEventListener('click', () => {
    const showing = els.externalPlannerToken.type === 'text';
    els.externalPlannerToken.type = showing ? 'password' : 'text';
    els.toggleExternalPlannerToken.setAttribute(
      'aria-label',
      showing ? '显示第三方 API token' : '隐藏第三方 API token',
    );
  });
  els.externalPlannerProtocol.addEventListener('change', invalidateExternalPlannerDetection);
  els.externalPlannerUrl.addEventListener('input', invalidateExternalPlannerDetection);
  els.externalPlannerToken.addEventListener('input', invalidateExternalPlannerDetection);
  els.externalPlannerCapability.addEventListener('change', () => {
    updatePlannerProviderUi();
    syncSendState();
  });
  els.externalPlannerModel.addEventListener('input', () => {
    updatePlannerProviderUi();
    syncSendState();
  });
  els.detectExternalPlannerModels.addEventListener('click', detectExternalPlannerModels);
  els.loadModels.addEventListener('click', loadModelList);
  els.model.addEventListener('change', syncModel);
  els.operation.addEventListener('change', syncOperation);
  els.operationPreset.addEventListener('change', () => {
    const op = selectedOperation();
    if (op) applyOperationPreset(op, els.operationPreset.value);
  });
  els.prompt.addEventListener('input', () => {
    syncSendState();
    updateContextPreview();
  });
  els.systemPrompt.addEventListener('input', updateContextPreview);
  els.maxTokens.addEventListener('input', updateContextPreview);
  els.generationCount.addEventListener('change', () => {
    const value = Number.parseInt(els.generationCount.value, 10);
    els.generationCount.value = String(
      Number.isInteger(value) ? Math.min(MAX_MEDIA_QUEUE_RUNS, Math.max(1, value)) : 1
    );
  });
  els.prompt.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || event.isComposing) return;
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      const start = els.prompt.selectionStart ?? els.prompt.value.length;
      const end = els.prompt.selectionEnd ?? start;
      els.prompt.setRangeText('\n', start, end, 'end');
      els.prompt.dispatchEvent(new Event('input', {bubbles: true}));
      return;
    }
    if (event.shiftKey || event.altKey) return;
    event.preventDefault();
    runRequest();
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
  els.cameraStart.addEventListener('click', startCameraAnalysis);
  els.cameraStop.addEventListener('click', () => stopCameraAnalysis());
  els.clear.addEventListener('click', () => {
    const op = selectedOperation();
    if (state.controller) {
      if (op && (op.id === 'decision' || op.ocr_mode)) state.requestDiscarded = true;
      state.controller.abort();
    }
    if (op) {
      state.interactionHistories.delete(interactionKey(els.model.value, op.id));
    }
    clearComposerInput();
    resetOutput();
    renderInteractionHistory();
    updateContextPreview();
    syncSendState();
    els.prompt.focus();
  });

  window.addEventListener('pagehide', () => {
    state.camera.active = false;
    state.camera.pending = null;
    stopCameraHardware();
  });

  syncModel();
})();
