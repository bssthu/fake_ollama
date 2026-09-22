"""Execute management UI helpers offline with ordinary text and fixture data."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.e2e
def test_management_ui_auth_headers_and_text_rendering(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the UI helper regression")
    script = tmp_path / "management-ui.cjs"
    script.write_text(r'''
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = process.argv[2];
function body(name) {
  return fs.readFileSync(root + '/' + name, 'utf8').split('<script>')[1].split('</script>')[0];
}
function fn(source, name) {
  const start = source.indexOf('function ' + name + '(');
  assert(start >= 0);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
for (const page of ['admin.html', 'dashboard.html']) {
  const source = body(page);
  new vm.Script(source);  // syntax-check the complete shipped script too
  let captured;
  const context = vm.createContext({Headers, window: {fetch: (url, options) => { captured = {url, options}; }}});
  vm.runInContext("const managementToken = 'fixture-token';\n" + fn(source, 'managementFetch'), context);
  vm.runInContext("managementFetch('/fixture', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})", context);
  assert.equal(captured.options.headers.get('X-Management-Token'), 'fixture-token');
  assert.equal(captured.options.headers.get('Content-Type'), 'application/json');
  assert.equal(captured.options.method, 'POST');
  assert.equal(captured.options.cache, 'no-store');
}
const source = body('dashboard.html');
const elements = {slotsTable: {}, slotsStatus: {}};
const context = vm.createContext({
  $: id => elements[id],
  latest: {target_telemetry: [{target_id: 'fixture', slots: {slots: [
    {id: 'A & B', task_id: 7, n_past: 4, n_predict: 12, n_ctx: 2048, state: 1},
  ]}}]},
});
vm.runInContext(fn(source, 'escapeHtml') + '\n' + fn(source, 'renderUpstreamSlots') + '\nrenderUpstreamSlots();', context);
assert(elements.slotsTable.innerHTML.includes('A &amp; B'));
assert(elements.slotsTable.innerHTML.includes('>7</td>'));
assert(elements.slotsTable.innerHTML.includes('>12</td>'));
assert.equal(elements.slotsStatus.textContent, '1/1 processing');
console.log('management UI helpers passed');
''', encoding="utf-8")
    static_dir = Path(__file__).resolve().parents[2] / "fake_ollama" / "static"
    result = subprocess.run([node, str(script), str(static_dir)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
