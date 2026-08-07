"""Live integration tests against the configured upstream.

These tests are skipped automatically unless ``FAKE_OLLAMA_TEST_BASE_URL``
and ``FAKE_OLLAMA_TEST_AUTH_TOKEN`` are set in the environment (or in
``.env``). The two variables are used only as a presence signal: the
actual upstream and credentials come from the active ``config.json``.
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
    # Live tests run whenever upstream creds are configured (env vars
    # or .env). They DO surface real regressions in the Anthropic <->
    # Ollama translation layer, so we intentionally do NOT hide them
    # behind an extra opt-in flag.
    if not os.getenv("FAKE_OLLAMA_TEST_BASE_URL") or not os.getenv(
        "FAKE_OLLAMA_TEST_AUTH_TOKEN"
    ):
        pytest.skip("upstream credentials not configured")
    get_settings.cache_clear()
    settings = get_settings()
    # Only skip on genuine upstream-side outages: connect errors or
    # 5xx. 4xx is kept as a real failure so routing / auth regressions
    # still surface. 401/403 on the unauthenticated probe just means
    # the endpoint is alive and gating auth — that's healthy.
    import httpx

    base_url = settings.anthropic_upstreams[0].base_url
    try:
        with httpx.Client(timeout=5.0, trust_env=settings.use_system_proxy) as cli:
            probe = cli.get(base_url + "/v1/models")
        if probe.status_code >= 500:
            pytest.skip(
                f"upstream {base_url} returned {probe.status_code}; "
                "skipping live integration tests (upstream-side outage)"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"upstream {base_url} unreachable: {exc}")
    return settings


@pytest.mark.integration
def test_live_chat_non_streaming(live_settings: Settings):
    app = create_app(live_settings)
    with TestClient(app) as client:
        model = live_settings.ollama_interfaces[0].public_ids()[0]
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
        # 5xx == upstream-side outage; skip so flaky provider does not
        # mask itself as a regression in our translation layer. Any
        # other status still fails the test as before.
        if 500 <= resp.status_code < 600:
            pytest.skip(f"upstream returned {resp.status_code}: {resp.text[:200]}")
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
        model = live_settings.ollama_interfaces[0].public_ids()[0]
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
            if 500 <= resp.status_code < 600:
                pytest.skip(
                    f"upstream returned {resp.status_code}: "
                    f"{resp.read().decode()[:200]}"
                )
            assert resp.status_code == 200, resp.read().decode()
            chunks = [json.loads(line) for line in resp.iter_lines() if line]

    # Mid-stream upstream failure: HTTP 200 was already sent, but the
    # SSE body contains an error chunk instead of content. Skip rather
    # than fail so flaky providers don't masquerade as regressions.
    for c in chunks:
        err = c.get("error") if isinstance(c, dict) else None
        if err:
            pytest.skip(f"upstream errored mid-stream: {err}")

    text = "".join(c["message"]["content"] for c in chunks if not c["done"])
    final = [c for c in chunks if c["done"]]
    assert text.strip() != ""
    assert len(final) == 1
    assert final[0]["done"] is True
