"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from fake_ollama.config import Settings, get_settings, load_settings

# Load .env early so both unit fixtures and integration skip-checks see it.
load_dotenv()


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://upstream.test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("FAKE_OLLAMA_MODELS", "claude-3-5-sonnet-20241022,llama-test")
    monkeypatch.setenv("FAKE_OLLAMA_DEFAULT_MAX_TOKENS", "1024")
    get_settings.cache_clear()
    return load_settings()


@pytest.fixture(autouse=True)
def _isolate_config_file(request, monkeypatch: pytest.MonkeyPatch, tmp_path_factory):
    """Stop tests from picking up a developer's real ./config.json or .env."""
    if "integration" in request.keywords:
        return
    p = tmp_path_factory.mktemp("noconfig") / "no-such-config.json"
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(p))
    # Strip any FAKE_OLLAMA_* / ANTHROPIC_* env vars loaded from .env so that
    # each test starts from a clean slate and only the values it sets via
    # monkeypatch are visible.
    for key in list(os.environ):
        if key.startswith("FAKE_OLLAMA_") or key.startswith("ANTHROPIC_"):
            if key == "FAKE_OLLAMA_CONFIG":
                continue
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    yield
    get_settings.cache_clear()


def _live_env_present() -> bool:
    return bool(os.getenv("ANTHROPIC_BASE_URL")) and bool(os.getenv("ANTHROPIC_AUTH_TOKEN"))


def pytest_collection_modifyitems(config, items):  # pragma: no cover - pytest hook
    if _live_env_present():
        return
    skip = pytest.mark.skip(
        reason="ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN not set; integration tests skipped"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
