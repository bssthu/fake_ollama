"""Process-level CLI smoke tests."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

# These tests spawn ``sys.executable`` to import ``fake_ollama.__main__``,
# whose top-level imports include ``uvicorn``. ``uvicorn`` is in
# the ``test`` optional dependency group, so a configured dev env has it
# and this skip will be a no-op; the guard only triggers when pytest is
# invoked from a stripped-down interpreter (e.g. bare system Python),
# in which case the failure would be an environment issue rather than a
# regression in the logging code under test.
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.e2e


def test_cli_logging_writes_default_style_log_file(tmp_path):
    log_file = tmp_path / "logs" / "fake_ollama.log"
    script = (
        "import logging, sys\n"
        "from fake_ollama.__main__ import _configure_logging\n"
        "_configure_logging('info', log_file=sys.argv[1])\n"
        "logging.getLogger('fake_ollama').info('file log smoke')\n"
        "for handler in logging.getLogger().handlers:\n"
        "    handler.flush()\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(log_file)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "file log smoke" in completed.stderr
    assert log_file.exists()
    assert "INFO fake_ollama: file log smoke" in log_file.read_text(encoding="utf-8")


def test_cli_help_mentions_separate_request_data_log():
    completed = subprocess.run(
        [sys.executable, "-m", "fake_ollama", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--request-data-log-file" in completed.stdout
    assert "--no-request-data-log" in completed.stdout
    assert "--privacy-mode" in completed.stdout
    assert "--playground-host" in completed.stdout
    assert "--playground-port" in completed.stdout


_CLI_REQUEST_SMOKE = r'''
import json
import os
import sys

import httpx
from fastapi.testclient import TestClient
from fake_ollama import __main__ as cli
from fake_ollama.anthropic_client import AnthropicClient

os.chdir(sys.argv[1])
sys.argv = ["fake-ollama", "--config", "config.json", "--log-level", "debug", *sys.argv[2:]]

def upstream(request):
    payload = json.loads(request.content)
    prompt = payload["messages"][0]["content"]
    if prompt == "private-error-input":
        return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "private-error-output"}})
    if prompt == "private-exception-input":
        raise ValueError("private-exception-output")
    message = {
        "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "private-output"}],
        "model": payload["model"], "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    if payload.get("stream"):
        events = [
            {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "private-stream-output"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        content = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)
    return httpx.Response(200, json=message)

class Server:
    def __init__(self, config, callback):
        self.config = config

    def run(self):
        app = self.config.app
        app.state.clients = {
            up.name: AnthropicClient(up.base_url, up.auth_token, client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
            for up in app.state.settings.anthropic_upstreams
        }
        with TestClient(app, base_url="http://127.0.0.1:21435", raise_server_exceptions=False) as client:
            for prompt, stream, status, expected in [
                ("private-input", False, 200, "private-output"),
                ("private-stream-input", True, 200, "private-stream-output"),
                ("private-error-input", False, 400, "private-error-output"),
                ("private-exception-input", False, 500, "Internal Server Error"),
            ]:
                response = client.post("/v1/messages", json={
                    "model": "claude-3-5-sonnet-20241022@default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16, "stream": stream,
                })
                assert response.status_code == status, response.text
                assert expected in response.text, response.text

cli._ShutdownAwareServer = Server
cli.main()
'''


@pytest.mark.parametrize("mode", ["default", "private", "private-custom-first", "private-custom-last", "legacy-disabled"])
def test_cli_privacy_controls_files_without_changing_api_responses(settings, tmp_path, mode):
    config = settings.model_dump(mode="json")
    config.update(admin_enabled=False, dashboard_enabled=False, ollama_interfaces=[])
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    args = []
    snapshots = {}
    if "custom" in mode:
        for name in ("app.log", "requests.jsonl"):
            path = tmp_path / "custom" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"existing history\n")
            snapshots[path] = (path.read_bytes(), path.stat().st_mtime_ns)
        args = ["--log-file", "custom/app.log", "--request-data-log-file", "custom/requests.jsonl"]
    if mode.startswith("private"):
        args.insert(0 if mode.endswith("first") else len(args), "--privacy-mode")
    elif mode == "legacy-disabled":
        args = ["--no-log-file", "--no-request-data-log"]

    completed = subprocess.run(
        [sys.executable, "-c", _CLI_REQUEST_SMOKE, str(tmp_path), *args],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "access listener=api" in completed.stderr
    assert ("privacy mode enabled" in completed.stderr) == mode.startswith("private")
    app_log = tmp_path / "logs" / "fake_ollama.log"
    request_log = tmp_path / "logs" / "fake_ollama.requests.jsonl"
    if mode == "default":
        content = request_log.read_text(encoding="utf-8")
        for value in ("private-input", "private-output", "private-stream-input", "private-stream-output", "private-error-output"):
            assert value in content
        content = app_log.read_text(encoding="utf-8")
        assert "private-error-output" in content
        assert "private-exception-output" in content
    else:
        assert not app_log.exists()
        assert not request_log.exists()
        assert set(tmp_path.rglob("*.log")) | set(tmp_path.rglob("*.jsonl")) == set(snapshots)
        for path, snapshot in snapshots.items():
            assert (path.read_bytes(), path.stat().st_mtime_ns) == snapshot


@pytest.mark.parametrize("private", [False, True])
async def test_managed_worker_file_capture_inherits_privacy(tmp_path, monkeypatch, private):
    from fake_ollama.privacy import PRIVACY_MODE_ENV
    from fake_ollama.process_utils import create_managed_subprocess_exec, terminate_process_tree

    monkeypatch.setenv(PRIVACY_MODE_ENV, "1" if private else "0")
    capture = tmp_path / "capture.log"
    nested_capture = tmp_path / "nested.log"
    script = (
        "import sys\n"
        "from fake_ollama.process_utils import _resolve_stderr\n"
        "print('private-worker-output', file=sys.stderr, flush=True)\n"
        "target, owned = _resolve_stderr(sys.argv[1])\n"
        "if owned is not None:\n"
        "    owned.write('private-nested-output')\n"
        "    owned.close()\n"
    )
    # Even an explicit child environment cannot undo the parent's privacy mode.
    process = await create_managed_subprocess_exec(
        [sys.executable, "-c", script, str(nested_capture)],
        stderr=capture, env={**os.environ, PRIVACY_MODE_ENV: "0"},
    )
    try:
        assert await asyncio.wait_for(process.wait(), 15) == 0
    finally:
        assert await terminate_process_tree(process, timeout=1)
    if private:
        assert not capture.exists()
        assert not nested_capture.exists()
    else:
        assert "private-worker-output" in capture.read_text(encoding="utf-8")
        assert "private-nested-output" in nested_capture.read_text(encoding="utf-8")
