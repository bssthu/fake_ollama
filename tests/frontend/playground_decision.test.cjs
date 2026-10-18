/* Offline DOM regression. Run with Node.js and jsdom available on NODE_PATH. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'fake_ollama/static/playground.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'fake_ollama/static/playground.js'), 'utf8');
const dialects = [
  {id: 'systemone', label: 'TypeSafe SystemOne', endpoint: '/v1/systemone'},
  {id: 'openai_decisions', label: 'OpenAI Decisions', endpoint: '/v1/decisions'},
];
function decision(id, options = {}) {
  const op = {
    id: 'decision', endpoint: '/v1/systemone', dialect: 'systemone', dialects,
    question_types: ['predicate', 'choice', 'score'], modalities: ['text'],
    accepts_images: false, image_dialect: 'systemone', history_mode: 'single_turn',
    runtime: 'external', image_format: 'typesafe_assets', extensions: [], think_modes: [], parameters: [], ...options,
  };
  return {id, capabilities: ['decision'], operations: [op]};
}
const models = [
  decision('fixture-decision'),
  decision('instant-named-startlux', {question_types: ['choice']}),
  decision('fixture-startlux', {runtime: 'startlux', image_format: 'startlux_images', modalities: ['text', 'image'], accepts_images: true,
    extensions: ['think_mode'], think_modes: ['never', 'auto', 'always']}),
  decision('fixture-typesafe-image', {modalities: ['text', 'image'], accepts_images: true}),
  decision('fixture-external-startlux', {image_format: 'startlux_images', modalities: ['text', 'image'], accepts_images: true}),
  decision('fixture-openai-image', {dialect: 'openai_decisions', image_dialect: 'openai_decisions', image_format: 'openai_input_image',
    modalities: ['text', 'image'], accepts_images: true}),
  {id: 'chat-fixture', capabilities: ['completion'], operations: [
    {id: 'chat', endpoint: '/v1/chat/completions', stream: true, history_mode: 'conversation'},
  ]},
];
const dom = new JSDOM(html, {url: 'http://localhost:21431/playground/', runScripts: 'outside-only'});
const {window} = dom;
window.TextEncoder = TextEncoder;
window.AbortController = AbortController;
const requests = [];
let reply = {answers: {question_1: {type: 'noul', noul: 0.75}}};
let httpStatus = 200;
let waitForAbort = false;
window.fetch = async (url, options) => {
  requests.push({url, options, body: options.body ? JSON.parse(options.body) : null});
  if (url === '/playground/api/models') return {ok: true, json: async () => ({schema_version: 1, models})};
  if (waitForAbort) {
    return new Promise((resolve, reject) => options.signal.addEventListener('abort', () => {
      const error = new Error('aborted'); error.name = 'AbortError'; reject(error);
    }, {once: true}));
  }
  return {ok: httpStatus >= 200 && httpStatus < 300, status: httpStatus, statusText: 'Fixture', text: async () => JSON.stringify(reply)};
};
window.eval(script);
const $ = id => window.document.getElementById(id);
function input(id, value, event = 'input') {
  $(id).value = value;
  $(id).dispatchEvent(new window.Event(event, {bubbles: true}));
}
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
async function settle() { for (let n = 0; n < 8; n++) await tick(); }
async function send() { $('send').click(); await settle(); }
function type(index, value) { input(`decision-question-${index}-type`, value, 'change'); }
function question(index, instructions, criteria = '') {
  input(`decision-question-${index}-instructions`, instructions);
  input(`decision-question-${index}-criteria`, criteria);
}
function lastPost() { return requests.filter(item => item.options.method === 'POST').at(-1); }
async function upload(name = 'fixture.png', mime = 'image/png') {
  const file = new window.File(['fixture-image'], name, {type: mime});
  Object.defineProperty($('fileInput'), 'files', {value: [file], configurable: true});
  $('fileInput').dispatchEvent(new window.Event('change'));
  await settle();
}
(async () => {
  assert.equal($('decisionSection').hidden, true);
  input('apiKey', 'fixture-secret');
  $('loadModels').click(); await settle();
  assert.equal(requests[0].options.headers.Authorization, 'Bearer fixture-secret');
  assert.equal($('decisionSection').hidden, false);
  assert.equal($('decisionThinkField').hidden, true);
  assert.equal($('uploadSection').hidden, true);
  assert.equal($('chatParams').hidden, true);
  assert.equal($('mediaParams').hidden, true);
  assert.equal($('ttftMetric').hidden, true);
  input('prompt', '{"risk":"low"}');
  input('decisionStateFormat', 'json', 'change');
  question(0, 'Is this acceptable?');
  $('decisionAddQuestion').click(); type(1, 'choice'); question(1, 'Choose a route', 'accept\nreject');
  $('decisionAddQuestion').click(); type(2, 'score'); question(2, 'Rate the risk', 'low\nmedium\nhigh');
  reply = {answers: {
    question_1: {type: 'noul', noul: 0.75},
    question_2: {type: 'choice', choice: 'accept', probabilities: {accept: 0.8, reject: 0.2}},
    question_3: {type: 'score', score: 0.6, probabilities: {'0': 0.5, '1': 0.4, '2': 0.1}, legend: {'0': 'low', '1': 'medium', '2': 'high'}},
  }, usage: {input_tokens: 12}, decision_metadata: {source: 'fixture', confidence_semantics: 'uncalibrated'}};
  await send();
  let post = lastPost();
  assert.equal(post.url, '/v1/systemone');
  assert.deepEqual(post.body.state, {risk: 'low'});
  assert.equal(post.body.questions.question_1.type, 'noul');
  assert.deepEqual(post.body.questions.question_2.criteria, {accept: null, reject: null});
  assert.deepEqual(post.body.questions.question_3.criteria, ['low', 'medium', 'high']);
  assert(!('input' in post.body)); assert(!('think_mode' in post.body));
  assert.equal(window.document.querySelectorAll('.decision-answer').length, 3);
  assert.equal(window.document.querySelectorAll('.decision-probabilities progress').length, 6);
  assert($('conversation').textContent.includes('75.00%'));
  assert($('conversation').textContent.includes('期望等级分数（从 0 开始）：0.6'));
  assert($('conversation').textContent.includes('原始响应'));
  assert(!$('conversation').textContent.includes('fixture-secret'));
  assert.equal($('prompt').value, '{"risk":"low"}'); // retryable state is kept
  assert.equal($('send').disabled, false);

  input('decisionDialect', 'openai_decisions', 'change');
  assert($('operationHint').textContent.includes('/v1/decisions'));
  reply = {answers: [
    {type: 'predicate', name: 'question_1', probability: 0.3},
    {type: 'choice', name: 'question_2', choice: 'reject', probabilities: [{value: 'accept', probability: 0.2}, {value: 'reject', probability: 0.8}]},
    {type: 'score', name: 'question_3', score: 1.2, probabilities: [{value: 0, label: 'low', probability: 0.2}, {value: 1, label: 'medium', probability: 0.4}, {value: 2, label: 'high', probability: 0.4}]},
  ]};
  await send(); post = lastPost();
  assert.equal(post.url, '/v1/decisions');
  assert.equal(post.body.input, '{"risk":"low"}');
  assert(Array.isArray(post.body.questions));
  assert.equal(post.body.questions[0].type, 'predicate');
  assert.deepEqual(post.body.questions[1].choices, [{value: 'accept'}, {value: 'reject'}]);
  assert.deepEqual(post.body.questions[2].levels, [{label: 'low'}, {label: 'medium'}, {label: 'high'}]);
  assert(!('state' in post.body)); assert(!('messages' in post.body));
  assert.equal(window.document.querySelectorAll('.decision-result').length, 2);
  $('loadModels').click(); await settle();
  assert.equal($('decisionDialect').value, 'openai_decisions');
  assert.equal($('decision-question-1-instructions').value, 'Choose a route');
  assert.equal(window.document.querySelectorAll('.decision-result').length, 2);

  const postCount = requests.filter(item => item.options.method === 'POST').length;
  input('prompt', '{bad'); await send();
  assert.equal(requests.filter(item => item.options.method === 'POST').length, postCount);
  assert($('errorBox').textContent.includes('有效的 JSON'));
  input('prompt', '{}');
  input('decisionDialect', 'systemone', 'change');
  input('decision-question-1-name', 'question_1'); await send();
  assert($('errorBox').textContent.includes('唯一'));
  input('decision-question-1-name', 'question_2');

  input('model', 'instant-named-startlux', 'change');
  assert.equal($('decisionThinkField').hidden, true); // never infer features from model names
  assert.equal($('uploadSection').hidden, true);
  assert.equal($('decision-question-0-type').options.length, 1);
  question(0, 'Choose', 'one\ntwo'); input('prompt', 'Context');
  reply = {answers: {question_1: {type: 'choice', choice: 'one'}}};
  await send();
  assert.equal(window.document.querySelectorAll('.decision-probabilities progress').length, 0); // no invented probabilities
  reply = {answers: {question_1: {type: 'choice', choice: null, none_probability: 0.72, probabilities: {one: 0.5, two: 0.5}}}};
  await send();
  assert($('conversation').textContent.includes('未选择任何列出的选项'));
  assert($('conversation').textContent.includes('警报分数：0.72'));
  assert($('conversation').textContent.includes('独立于列出选项'));
  assert.equal(window.document.querySelectorAll('.decision-probabilities progress').length, 2); // alarm is not a third distribution category

  input('model', 'fixture-startlux', 'change');
  assert.equal($('decisionThinkField').hidden, false);
  assert.equal($('uploadSection').hidden, false);
  input('decisionThinkMode', 'always', 'change');
  question(0, 'Inspect the image'); await upload();
  reply = {answers: {question_1: {type: 'noul', noul: 0.9}}};
  await send(); post = lastPost();
  assert.equal(post.body.think_mode, 'always');
  assert(!('extensions' in post.body));
  assert(!('TypeSafe-Danger-Zone-Images' in post.options.headers));
  assert.equal(post.body.images.length, 1);
  assert(post.body.images[0].startsWith('data:image/png;base64,'));
  assert.equal(post.body.state, 'Context');
  assert(!('assets' in post.body));
  input('decisionDialect', 'openai_decisions', 'change');
  assert.equal($('uploadSection').hidden, false); // retain uploads so users can remove them explicitly
  assert.equal($('dropZone').disabled, true);
  assert.equal($('send').disabled, true);
  assert.equal($('attachmentList').children.length, 1);
  assert($('dropHint').textContent.includes('不会自动转换或丢弃'));
  input('decisionDialect', 'systemone', 'change');
  assert.equal($('send').disabled, false);
  assert.equal($('attachmentList').children.length, 1);
  input('decisionDialect', 'openai_decisions', 'change');
  $('attachmentList').querySelector('button').click();
  assert.equal($('send').disabled, false);
  assert($('decisionCapabilityHint').textContent.includes('后端原生'));

  input('model', 'fixture-typesafe-image', 'change');
  question(0, 'Inspect native TypeSafe assets'); await upload();
  reply = {answers: {question_1: {type: 'noul', noul: 0.9}}};
  await send(); post = lastPost();
  assert.equal(post.options.headers['TypeSafe-Danger-Zone-Images'], 'true');
  assert.equal(post.body.assets.playground_image_1.type, 'image');
  assert.equal(post.body.assets.playground_image_1.format, 'png');
  assert.deepEqual(post.body.state, {context: 'Context', images: ['playground_image_1']});
  assert(!('images' in post.body));
  $('attachmentList').querySelector('button').click();

  input('model', 'fixture-external-startlux', 'change');
  assert.equal($('uploadSection').hidden, false);
  question(0, 'Inspect external StartLux image'); await upload();
  await send(); post = lastPost();
  assert.equal(post.body.images.length, 1);
  assert(post.body.images[0].startsWith('data:image/png;base64,'));
  assert.equal(post.body.state, 'Context');
  assert(!('assets' in post.body));
  assert(!('TypeSafe-Danger-Zone-Images' in post.options.headers));
  $('attachmentList').querySelector('button').click();

  input('model', 'fixture-openai-image', 'change');
  assert.equal($('decisionDialect').value, 'openai_decisions');
  assert.equal($('decisionThinkField').hidden, true);
  question(0, 'Inspect'); await upload();
  reply = {answers: [{type: 'predicate', probability: 0.5}]};
  await send(); post = lastPost();
  assert.equal(post.body.input[0].content[0].type, 'input_text');
  assert.equal(post.body.input[0].content[1].type, 'input_image');
  assert(post.body.input[0].content[1].image_url.startsWith('data:image/png;base64,'));
  assert(!('assets' in post.body));

  httpStatus = 503; reply = {error: {message: '<img src=x onerror="bad()"> unavailable'}};
  await send();
  assert($('errorBox').textContent.includes('<img src=x'));
  assert.equal(window.document.querySelectorAll('.decision-result img').length, 0);
  assert($('conversation').textContent.includes('HTTP 503'));
  assert.equal($('send').disabled, false);
  httpStatus = 200;
  waitForAbort = true;
  $('send').click(); await tick();
  assert.equal($('decisionThinkMode').disabled, true);
  assert.equal($('stop').hidden, false);
  const beforeRepeated = requests.length;
  $('send').click(); await tick(); assert.equal(requests.length, beforeRepeated);
  $('stop').click(); await settle();
  assert($('conversation').textContent.includes('请求已停止'));
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  $('send').click(); await tick();
  $('clear').click(); await settle();
  assert.equal(window.document.querySelectorAll('.decision-result').length, 0);
  assert.equal($('errorBox').textContent, '');
  assert.equal($('send').disabled, false);
  waitForAbort = false;

  input('model', 'chat-fixture', 'change');
  assert.equal($('decisionSection').hidden, true);
  assert.equal($('decisionStateLabel').hidden, true);
  assert.equal($('chatParams').hidden, false);
  assert.equal($('mediaParams').hidden, true);
  assert.equal($('ttftMetric').hidden, false);
  input('model', 'fixture-decision', 'change');
  assert.equal(window.document.querySelectorAll('.decision-result').length, 2);
  $('clear').click(); await settle();
  assert.equal(window.document.querySelectorAll('.decision-result').length, 0);
  assert.equal($('prompt').value, '');
  assert.equal($('decision-question-0-instructions').value, 'Is this acceptable?');

  input('apiKey', 'new-fixture-key');
  assert.equal($('send').disabled, true);
  $('loadModels').click(); await settle();
  assert.equal($('decision-question-0-instructions').value, ''); // old user's draft is dropped
  reply = {answers: {question_1: {type: 'noul', noul: 0.5}}};
  await send();
  assert.equal(lastPost().body.state, ''); // empty state/instructions are valid native schemas
  assert.equal(lastPost().body.questions.question_1.instructions, '');
  assert.equal(window.localStorage.length, 0);
  assert.equal(window.sessionStorage.length, 0);
  if (process.env.PLAYGROUND_REQUEST_FIXTURES) {
    fs.writeFileSync(process.env.PLAYGROUND_REQUEST_FIXTURES, JSON.stringify(
      requests.filter(item => item.options.method === 'POST').map(item => ({
        endpoint: item.url, body: item.body,
        capabilities: models.find(model => model.id === item.body.model).operations[0],
      })),
    ));
  }
  console.log('Playground decision DOM: codecs, results, gating, images, errors, cancellation, retries, reload, clear, and chat regression passed.');
})().finally(() => window.close()).catch(error => { console.error(error); process.exitCode = 1; });
