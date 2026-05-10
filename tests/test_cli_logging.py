from __future__ import annotations

import subprocess
import sys


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
