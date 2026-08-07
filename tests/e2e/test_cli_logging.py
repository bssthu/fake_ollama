"""Process-level CLI smoke tests."""

from __future__ import annotations

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
    assert "--playground-host" in completed.stdout
    assert "--playground-port" in completed.stdout
