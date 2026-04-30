"""Tests for the JSON-based config loader and multi-upstream routing."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.config import LEGACY_UPSTREAM_NAME, Settings, load_settings
from fake_ollama.server import create_app


def _write_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_loads_from_json_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write_config(
        cfg,
        {
            "host": "0.0.0.0",
            "port": 31434,
            "default_max_tokens": 2048,
            "upstreams": [
                {
                    "name": "anthropic",
                    "base_url": "https://api.example.com/",
                    "auth_token": "json-token",
                    "models": ["claude-x", "claude-y"],
                }
            ],
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("FAKE_OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("FAKE_OLLAMA_DEFAULT_MAX_TOKENS", raising=False)

    s = load_settings()
    assert s.host == "0.0.0.0"
    assert s.port == 31434
    assert s.default_max_tokens == 2048
    assert len(s.upstreams) == 1
    up = s.upstreams[0]
    assert up.name == "anthropic"
    # base_url trailing slash gets stripped
    assert up.base_url == "https://api.example.com"
    assert up.auth_token == "json-token"
    assert s.models == ["claude-x", "claude-y"]


def test_env_vars_override_json(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write_config(
        cfg,
        {
            "port": 31434,
            "upstreams": [
                {
                    "name": LEGACY_UPSTREAM_NAME,
                    "base_url": "https://json-only.example.com",
                    "auth_token": "json-token",
                    "models": ["json-model"],
                }
            ],
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    monkeypatch.setenv("FAKE_OLLAMA_PORT", "41434")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env-wins.example.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
    monkeypatch.delenv("FAKE_OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("FAKE_OLLAMA_MODEL_MAP", raising=False)

    s = load_settings()
    assert s.port == 41434
    # The "default" upstream should be merged: env wins for url + token, but
    # models came from JSON because env didn't supply any.
    default = s.upstreams[0]
    assert default.name == LEGACY_UPSTREAM_NAME
    assert default.base_url == "https://env-wins.example.com"
    assert default.auth_token == "env-token"
    assert default.models == ["json-model"]


def test_legacy_env_only_creates_default_upstream(monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", "/no/such/path.json")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://legacy.example.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "legacy")
    monkeypatch.setenv("FAKE_OLLAMA_MODELS", "claude-3-5-sonnet-20241022")

    s = load_settings()
    assert len(s.upstreams) == 1
    assert s.upstreams[0].name == LEGACY_UPSTREAM_NAME
    assert s.upstreams[0].base_url == "http://legacy.example.com"
    assert s.upstreams[0].models == ["claude-3-5-sonnet-20241022"]


def test_no_upstream_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(Exception):
        load_settings()


def test_resolve_model_respects_per_upstream_map(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    _write_config(
        cfg,
        {
            "upstreams": [
                {
                    "name": "anthropic",
                    "base_url": "https://anthropic.example.com",
                    "auth_token": "a-tok",
                    "models": ["sonnet"],
                    "model_map": {"sonnet": "claude-3-5-sonnet-20241022"},
                },
                {
                    "name": "deepseek",
                    "base_url": "https://deepseek.example.com",
                    "auth_token": "d-tok",
                    "models": ["dpsk"],
                    "model_map": {"dpsk": "deepseek-v4-pro"},
                },
            ]
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    s = load_settings()
    assert s.models == ["sonnet", "dpsk"]
    assert s.upstream_name_for("sonnet") == "anthropic"
    assert s.upstream_name_for("dpsk") == "deepseek"
    assert s.resolve_model("sonnet") == "claude-3-5-sonnet-20241022"
    assert s.resolve_model("dpsk") == "deepseek-v4-pro"


def test_routes_request_to_correct_upstream(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    _write_config(
        cfg,
        {
            "upstreams": [
                {
                    "name": "anthropic",
                    "base_url": "https://anthropic.example.com",
                    "auth_token": "a-tok",
                    "models": ["sonnet"],
                    "model_map": {"sonnet": "claude-3-5-sonnet-20241022"},
                },
                {
                    "name": "deepseek",
                    "base_url": "https://deepseek.example.com",
                    "auth_token": "d-tok",
                    "models": ["dpsk"],
                    "model_map": {"dpsk": "deepseek-v4-pro"},
                },
            ]
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    s = load_settings()

    hits: dict[str, list[str]] = {"anthropic": [], "deepseek": []}

    def make_handler(name: str):
        def _h(req: httpx.Request) -> httpx.Response:
            hits[name].append(json.loads(req.content)["model"])
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": f"hi from {name}"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        return _h

    app = create_app(s)
    # Inject one mocked AnthropicClient per upstream, each pointed at its own
    # MockTransport so we can verify routing.
    app.state.clients = {
        "anthropic": AnthropicClient(
            "https://anthropic.example.com",
            "a-tok",
            client=httpx.AsyncClient(transport=httpx.MockTransport(make_handler("anthropic"))),
        ),
        "deepseek": AnthropicClient(
            "https://deepseek.example.com",
            "d-tok",
            client=httpx.AsyncClient(transport=httpx.MockTransport(make_handler("deepseek"))),
        ),
    }

    with TestClient(app) as tc:
        r1 = tc.post("/api/chat", json={"model": "sonnet", "stream": False,
                                        "messages": [{"role": "user", "content": "hi"}]})
        r2 = tc.post("/api/chat", json={"model": "dpsk", "stream": False,
                                        "messages": [{"role": "user", "content": "hi"}]})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert hits["anthropic"] == ["claude-3-5-sonnet-20241022"]
    assert hits["deepseek"] == ["deepseek-v4-pro"]


def test_tags_unions_models_across_upstreams(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    _write_config(
        cfg,
        {
            "upstreams": [
                {"name": "u1", "base_url": "https://a", "auth_token": "x",
                 "models": ["alpha", "beta"]},
                {"name": "u2", "base_url": "https://b", "auth_token": "y",
                 "models": ["beta", "gamma"]},
            ]
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    s = load_settings()
    # dedupe, order preserved (first occurrence wins)
    assert s.models == ["alpha", "beta", "gamma"]
    # routing: 'beta' goes to u1 because it appears first
    assert s.upstream_name_for("beta") == "u1"
    assert s.upstream_name_for("gamma") == "u2"
