const { test } = require('node:test');
const assert = require('node:assert/strict');
const { timestamp, subtitles } = require('../../fake_ollama/static/speech.js');

test('subtitle clock survives hour boundaries and millisecond rounding', () => {
  assert.equal(timestamp(3599.9996), '01:00:00,000');
  assert.equal(timestamp(3.5), '00:00:03,500');
  assert.equal(timestamp(-1), '00:00:00,000');
});

test('bilingual export preserves mixed source and visibly marks failed translation', () => {
  const result = subtitles([
    { start: 0, end: 2.1, text: 'Hello 你好 bonjour', translation: '你好，你好，你好' },
    { start: 2.1, end: 3, text: 'Still visible', translation: '', state: 'translation_error' },
  ]);
  assert.match(result, /Hello 你好 bonjour\n你好，你好，你好/);
  assert.match(result, /00:00:02,100 --> 00:00:03,000\nStill visible\n\[译文未完成\]/);
});
