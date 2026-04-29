"""Live integration test against the configured upstream.

These tests are skipped automatically unless ``ANTHROPIC_BASE_URL`` and
``ANTHROPIC_AUTH_TOKEN`` are set in the environment (or in ``.env``).
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from fake_ollama.config import Settings, get_settings
from fake_ollama.server import create_app


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    # Keep whatever the user has in their environment / .env file.
    if not os.getenv("ANTHROPIC_BASE_URL") or not os.getenv("ANTHROPIC_AUTH_TOKEN"):
        pytest.skip("upstream credentials not configured")
    get_settings.cache_clear()
    return get_settings()


@pytest.mark.integration
def test_live_chat_non_streaming(live_settings: Settings):
    app = create_app(live_settings)
    with TestClient(app) as client:
        model = live_settings.models[0]
        resp = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Reply with the single word: pong"}
                ],
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["done"] is True
        assert body["message"]["role"] == "assistant"
        assert isinstance(body["message"]["content"], str)
        assert body["message"]["content"].strip() != ""


@pytest.mark.integration
def test_live_chat_streaming(live_settings: Settings):
    app = create_app(live_settings)
    with TestClient(app) as client:
        model = live_settings.models[0]
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Count 1 to 3, comma-separated."}
                ],
                "stream": True,
                "options": {"num_predict": 256, "temperature": 0},
            },
        ) as resp:
            assert resp.status_code == 200, resp.read().decode()
            chunks = [json.loads(line) for line in resp.iter_lines() if line]

    text = "".join(c["message"]["content"] for c in chunks if not c["done"])
    final = [c for c in chunks if c["done"]]
    assert text.strip() != ""
    assert len(final) == 1
    assert final[0]["done"] is True
