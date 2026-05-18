"""Tests for the JSON-based config loader and composite-id routing."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.config import Settings, load_settings
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

    s = load_settings()
    assert s.host == "0.0.0.0"
    assert s.port == 31434
    assert s.default_max_tokens == 2048
    assert len(s.upstreams) == 1
    up = s.upstreams[0]
    assert up.name == "anthropic"
    # trailing slash stripped
    assert up.base_url == "https://api.example.com"
    assert up.auth_token == "json-token"
    # Composite ids reflect (model, target)
    assert s.all_composite_ids() == ["claude-x@anthropic", "claude-y@anthropic"]
    assert s.models == ["claude-x@anthropic", "claude-y@anthropic"]


def test_admin_listener_defaults_and_legacy_null_mode():
    s = Settings(
        upstreams=[
            {
                "name": "u",
                "base_url": "http://upstream.test",
                "auth_token": "tk",
                "models": ["m"],
            }
        ]
    )
    assert s.admin_host == "127.0.0.1"
    assert s.admin_port == 21433
    assert s.admin_listener_enabled is True
    assert s.dashboard_host == "127.0.0.1"
    assert s.dashboard_port == 21432
    assert s.dashboard_data_path == "logs/dashboard_history.json"
    assert s.dashboard_model_reclaim_enabled is False
    assert s.dashboard_listener_enabled is True

    legacy = Settings(
        admin_port=None,
        upstreams=[
            {
                "name": "u",
                "base_url": "http://upstream.test",
                "auth_token": "tk",
                "models": ["m"],
            }
        ],
    )
    assert legacy.admin_listener_enabled is False


def test_listener_ports_must_be_distinct():
    with pytest.raises(ValueError, match="conflicts"):
        Settings(
            admin_port=21434,
            upstreams=[
                {
                    "name": "u",
                    "base_url": "http://upstream.test",
                    "auth_token": "tk",
                    "models": ["m"],
                }
            ],
        )

    with pytest.raises(ValueError, match="conflicts"):
        Settings(
            dashboard_port=21433,
            upstreams=[
                {
                    "name": "u",
                    "base_url": "http://upstream.test",
                    "auth_token": "tk",
                    "models": ["m"],
                }
            ],
        )


def test_empty_config_is_allowed():
    # No upstreams / targets at all is now valid; the server simply has no
    # exposed models.
    s = Settings()
    assert s.all_composite_ids() == []
    assert s.exposed_composite_ids("internal") == []
    assert s.exposed_composite_ids("external") == []


def test_source_names_must_be_unique_across_kinds():
    with pytest.raises(ValueError, match="unique"):
        Settings(
            upstreams=[
                {
                    "name": "shared",
                    "base_url": "https://a",
                    "auth_token": "x",
                    "models": ["m"],
                }
            ],
            ollama_targets=[
                {
                    "name": "shared",
                    "base_url": "http://127.0.0.1:11434",
                    "models": ["n"],
                }
            ],
        )


def test_source_names_reject_at_sign():
    with pytest.raises(ValueError, match="@"):
        Settings(
            upstreams=[
                {
                    "name": "weird@name",
                    "base_url": "https://a",
                    "auth_token": "x",
                    "models": ["m"],
                }
            ]
        )


def test_resolve_request_routes_to_correct_upstream(monkeypatch, tmp_path):
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
            ],
            "internal_exposed_models": ["sonnet@anthropic", "dpsk@deepseek"],
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    s = load_settings()
    assert s.all_composite_ids() == ["sonnet@anthropic", "dpsk@deepseek"]

    backend_a, real_a = s.resolve_request("sonnet@anthropic", surface="internal")
    assert backend_a.name == "anthropic"
    assert real_a == "sonnet"
    assert backend_a.source.resolve_model(real_a) == "claude-3-5-sonnet-20241022"

    backend_d, real_d = s.resolve_request("dpsk@deepseek", surface="internal")
    assert backend_d.name == "deepseek"
    assert real_d == "dpsk"
    assert backend_d.source.resolve_model(real_d) == "deepseek-v4-pro"


def test_resolve_request_rejects_bare_model_with_helpful_error():
    s = Settings(
        upstreams=[
            {
                "name": "anthropic",
                "base_url": "https://a",
                "auth_token": "x",
                "models": ["sonnet"],
            },
            {
                "name": "deepseek",
                "base_url": "https://d",
                "auth_token": "y",
                "models": ["dpsk"],
            },
        ],
        internal_exposed_models=["sonnet@anthropic", "dpsk@deepseek"],
    )
    with pytest.raises(ValueError) as exc:
        s.resolve_request("sonnet", surface="internal")
    msg = str(exc.value)
    assert "model@target" in msg
    assert "sonnet@anthropic" in msg


def test_resolve_request_rejects_unknown_target():
    s = Settings(
        upstreams=[
            {
                "name": "u1",
                "base_url": "https://a",
                "auth_token": "x",
                "models": ["m"],
            }
        ],
        internal_exposed_models=["m@u1"],
    )
    with pytest.raises(ValueError, match="unknown target"):
        s.resolve_request("m@u2", surface="internal")


def test_resolve_request_rejects_model_not_served_by_target():
    s = Settings(
        upstreams=[
            {
                "name": "u1",
                "base_url": "https://a",
                "auth_token": "x",
                "models": ["m1"],
            }
        ],
        internal_exposed_models=["m1@u1"],
    )
    with pytest.raises(ValueError, match="does not serve"):
        s.resolve_request("m2@u1", surface="internal")


def test_resolve_request_rejects_unexposed_model():
    s = Settings(
        upstreams=[
            {
                "name": "u1",
                "base_url": "https://a",
                "auth_token": "x",
                "models": ["m1"],
            }
        ],
        internal_exposed_models=[],
    )
    with pytest.raises(ValueError, match="not exposed"):
        s.resolve_request("m1@u1", surface="internal")


def test_exposed_surfaces_are_independent():
    s = Settings(
        upstreams=[
            {
                "name": "u1",
                "base_url": "https://a",
                "auth_token": "x",
                "models": ["m1", "m2"],
            }
        ],
        internal_exposed_models=["m1@u1", "m2@u1"],
        external_exposed_models=["m1@u1"],
    )
    assert s.exposed_composite_ids("internal") == ["m1@u1", "m2@u1"]
    assert s.exposed_composite_ids("external") == ["m1@u1"]
    assert s.is_exposed("external", "m1@u1") is True
    assert s.is_exposed("external", "m2@u1") is False


def test_tagless_model_match_via_latest_alias():
    s = Settings(
        upstreams=[
            {
                "name": "tagged",
                "base_url": "https://upstream.example.com",
                "auth_token": "tok",
                "models": ["qwen3.5-2b:latest"],
            }
        ],
        internal_exposed_models=["qwen3.5-2b:latest@tagged"],
        model_profiles={
            "qwen3.5-2b:latest": {
                "capabilities": ["completion", "tools"],
                "context_length": 8192,
            }
        },
    )
    # resolve_request only accepts composite ids — bare names always 400 at
    # the API boundary, but ``profile_for`` falls back to base name lookup.
    assert s.profile_for("qwen3.5-2b").context_length == 8192


def test_llama_cpp_target_routing_via_composite_id():
    s = Settings(
        llama_cpp_targets=[
            {
                "base_url": "http://127.0.0.1:21436/",
                "model": "qwen3.6:latest",
                "model_alias": "qwen3.6-alias",
                "auto_start": True,
                "start_command": "run-qwen36",
                "idle_timeout_seconds": 1800,
                "health_path": "health",
            }
        ],
        internal_exposed_models=["qwen3.6:latest@qwen3.6:latest"],
        model_profiles={
            "qwen3.6:latest": {"context_length": 262144, "estimated_vram_gb": 18}
        },
    )

    # llama_cpp target's source name defaults to its single model name
    target = s.llama_cpp_targets[0]
    assert target.name == "qwen3.6:latest"
    assert target.base_url == "http://127.0.0.1:21436"
    assert target.health_path == "/health"
    assert target.resolve_model("qwen3.6:latest") == "qwen3.6-alias"
    assert s.profile_for("qwen3.6:latest").context_length == 262144


def test_llama_cpp_defaults_are_inherited_and_overridden():
    s = Settings(
        llama_cpp_defaults={
            "auto_start": True,
            "idle_timeout_seconds": 1800,
            "startup_timeout_seconds": 600,
            "health_path": "ready",
            "cwd": "I:\\Projects\\llama.cpp",
        },
        llama_cpp_targets=[
            {
                "model": "hidden-qwen",
                "base_url": "http://127.0.0.1:21436",
                "start_command": "run-hidden",
            },
            {
                "model": "visible-qwen",
                "base_url": "http://127.0.0.1:21437",
                "auto_start": False,
                "health_path": "healthz",
            },
        ],
    )

    hidden = s.effective_llama_cpp_target(s.llama_cpp_targets[0])
    visible = s.effective_llama_cpp_target(s.llama_cpp_targets[1])
    assert hidden.name == "hidden-qwen"
    assert hidden.auto_start is True
    assert hidden.idle_timeout_seconds == 1800
    assert hidden.startup_timeout_seconds == 600
    assert hidden.health_path == "/ready"
    assert hidden.cwd == "I:\\Projects\\llama.cpp"
    assert visible.auto_start is False
    assert visible.health_path == "/healthz"


def test_llama_cpp_target_accepts_legacy_single_model_fields():
    s = Settings(
        llama_cpp_targets=[
            {
                "base_url": "http://127.0.0.1:21436",
                "models": ["qwen3.6"],
                "model_map": {"qwen3.6": "qwen3.6-alias"},
            }
        ],
    )
    target = s.llama_cpp_targets[0]
    assert target.name == "qwen3.6"
    assert target.model == "qwen3.6"
    assert target.model_alias == "qwen3.6-alias"


def test_llama_cpp_target_requires_one_model_per_process():
    with pytest.raises(ValueError, match="exactly one model"):
        Settings(
            llama_cpp_targets=[
                {
                    "name": "shared-process",
                    "base_url": "http://127.0.0.1:21436",
                    "models": ["qwen3.6", "qwen3.6-coder"],
                    "start_command": "run-shared",
                }
            ],
        )


def test_routes_request_via_composite_id(monkeypatch, tmp_path):
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
            ],
            "internal_exposed_models": ["sonnet@anthropic", "dpsk@deepseek"],
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
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
        r1 = tc.post(
            "/api/chat",
            json={
                "model": "sonnet@anthropic",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        r2 = tc.post(
            "/api/chat",
            json={
                "model": "dpsk@deepseek",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert hits["anthropic"] == ["claude-3-5-sonnet-20241022"]
    assert hits["deepseek"] == ["deepseek-v4-pro"]


def test_tags_unions_models_across_upstreams(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    _write_config(
        cfg,
        {
            "upstreams": [
                {
                    "name": "u1",
                    "base_url": "https://a",
                    "auth_token": "x",
                    "models": ["alpha", "beta"],
                },
                {
                    "name": "u2",
                    "base_url": "https://b",
                    "auth_token": "y",
                    "models": ["beta", "gamma"],
                },
            ]
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    s = load_settings()
    # Each (model, target) becomes its own composite id; "beta" appears in
    # both upstreams as distinct ids.
    assert s.all_composite_ids() == [
        "alpha@u1",
        "beta@u1",
        "beta@u2",
        "gamma@u2",
    ]
    a_backend, a_real = s.resolve_request(
        "alpha@u1", surface="internal"
    ) if "alpha@u1" in s.exposed_composite_ids("internal") else (
        s.backend_by_name("u1"),
        "alpha",
    )
    # No exposure declared => bare exposure-aware routing rejects, but
    # backend_for() bypasses exposure.
    assert s.backend_for("alpha@u1").name == "u1"
    assert s.backend_for("beta@u2").name == "u2"
