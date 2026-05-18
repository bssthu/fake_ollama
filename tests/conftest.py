"""Shared pytest fixtures."""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv

from fake_ollama.config import Settings, get_settings, load_settings

# Load .env early so integration skip-checks see FAKE_OLLAMA_TEST_*.
load_dotenv()


def _write_default_config(path) -> None:
    """Write a minimal config.json used by the generic ``settings`` fixture."""
    data = {
        "upstreams": [
            {
                "name": "default",
                "base_url": "http://upstream.test",
                "auth_token": "test-token",
                "models": ["claude-3-5-sonnet-20241022", "llama-test"],
            }
        ],
        "internal_exposed_models": [
            "claude-3-5-sonnet-20241022@default",
            "llama-test@default",
        ],
        "external_exposed_models": [
            "claude-3-5-sonnet-20241022@default",
            "llama-test@default",
        ],
        "default_max_tokens": 1024,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Settings:
    config_path = tmp_path / "config.json"
    _write_default_config(config_path)
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(config_path))
    get_settings.cache_clear()
    return load_settings()


@pytest.fixture(autouse=True)
def _isolate_config_file(request, monkeypatch: pytest.MonkeyPatch, tmp_path_factory):
    """Stop tests from picking up a developer's real ./config.json."""
    if "integration" in request.keywords:
        return
    p = tmp_path_factory.mktemp("noconfig") / "no-such-config.json"
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(p))


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    yield
    get_settings.cache_clear()


def _live_env_present() -> bool:
    return bool(os.getenv("FAKE_OLLAMA_TEST_BASE_URL")) and bool(
        os.getenv("FAKE_OLLAMA_TEST_AUTH_TOKEN")
    )


def pytest_collection_modifyitems(config, items):  # pragma: no cover - pytest hook
    if _live_env_present():
        return
    skip = pytest.mark.skip(
        reason="FAKE_OLLAMA_TEST_BASE_URL/FAKE_OLLAMA_TEST_AUTH_TOKEN not set; integration tests skipped"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
