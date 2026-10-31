"""Privacy must close existing request logs and prevent later re-enabling."""

from fake_ollama.privacy import PRIVACY_MODE_ENV, configure_privacy_mode
from fake_ollama.request_data_log import (
    configure_request_data_logging,
    log_data_event,
    request_data_log_path,
    request_data_logging_enabled,
)


def test_privacy_closes_existing_request_log_and_blocks_reconfiguration(tmp_path, monkeypatch):
    monkeypatch.delenv(PRIVACY_MODE_ENV, raising=False)
    existing = tmp_path / "existing.jsonl"
    ignored = tmp_path / "ignored" / "requests.jsonl"
    try:
        configure_request_data_logging(str(existing))
        log_data_event("before_privacy", body="old content")
        before = existing.read_bytes()

        configure_privacy_mode(True)
        log_data_event("after_privacy", body="private content")
        configure_request_data_logging(str(ignored))
        assert not request_data_logging_enabled()
        assert request_data_log_path() is None
        assert existing.read_bytes() == before
        assert not ignored.parent.exists()

        configure_privacy_mode(False)
        configure_request_data_logging(str(existing))
        log_data_event("normal_again", body="new content")
        assert "new content" in existing.read_text(encoding="utf-8")
    finally:
        configure_request_data_logging(None)
