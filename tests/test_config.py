"""Tests for the JSON-based config loader and the new interface/source API."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.config import Settings, load_settings
from fake_ollama.server import create_app


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Loader & top-level shape
# ---------------------------------------------------------------------------


def test_loads_from_json_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write(
        cfg,
        {
            "default_max_tokens": 2048,
            "anthropic_upstreams": [
                {
                    "name": "anthropic",
                    "base_url": "https://api.example.com/",
                    "auth_token": "json-token",
                    "models": [{"name": "claude-x"}, {"name": "claude-y"}],
                }
            ],
            "ollama_interfaces": [
                {
                    "name": "ollama",
                    "host": "127.0.0.1",
                    "port": 21434,
                    "access_tokens": [],
                    "exposed_models": [
                        {"model": "claude-x", "target": "anthropic"},
                        {"model": "claude-y", "target": "anthropic"},
                    ],
                }
            ],
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))

    s = load_settings()
    assert s.default_max_tokens == 2048
    assert len(s.anthropic_upstreams) == 1
    up = s.anthropic_upstreams[0]
    assert up.name == "anthropic"
    assert up.base_url == "https://api.example.com"  # trailing slash stripped
    assert up.auth_token == "json-token"
    iface = s.ollama_interfaces[0]
    assert iface.public_ids() == ["claude-x@anthropic", "claude-y@anthropic"]


def test_removed_top_level_keys_are_rejected(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write(cfg, {"host": "0.0.0.0", "port": 31434})
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    with pytest.raises(ValueError):
        load_settings()


def test_legacy_upstreams_key_is_rejected(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write(cfg, {"upstreams": []})
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    with pytest.raises(ValueError):
        load_settings()


def test_admin_dashboard_defaults():
    s = Settings()
    assert s.admin_host == "127.0.0.1"
    assert s.admin_port == 21433
    assert s.admin_listener_enabled is True
    assert s.dashboard_host == "127.0.0.1"
    assert s.dashboard_port == 21432
    assert s.dashboard_listener_enabled is True


def test_admin_port_null_disables_admin_listener():
    s = Settings(admin_port=None)
    assert s.admin_listener_enabled is False


# ---------------------------------------------------------------------------
# Port-conflict detection
# ---------------------------------------------------------------------------


def test_interface_port_conflict_rejected():
    with pytest.raises(ValueError, match="conflicts|duplicate|port"):
        Settings(
            ollama_interfaces=[
                {"name": "a", "port": 21434, "exposed_models": []},
                {"name": "b", "port": 21434, "exposed_models": []},
            ],
        )


def test_admin_port_must_differ_from_interface_port():
    with pytest.raises(ValueError, match="conflicts|duplicate|port"):
        Settings(
            admin_port=21434,
            ollama_interfaces=[{"name": "ollama", "port": 21434, "exposed_models": []}],
        )


# ---------------------------------------------------------------------------
# Source names
# ---------------------------------------------------------------------------


def test_source_names_must_be_unique_across_kinds():
    with pytest.raises(ValueError, match="unique|duplicate"):
        Settings(
            anthropic_upstreams=[
                {"name": "shared", "base_url": "https://a", "auth_token": "x",
                 "models": [{"name": "m"}]},
            ],
            ollama_targets=[
                {"name": "shared", "base_url": "http://127.0.0.1:11434",
                 "models": [{"name": "n"}]},
            ],
        )


def test_source_names_reject_at_sign():
    with pytest.raises(ValueError, match="@"):
        Settings(
            anthropic_upstreams=[
                {"name": "weird@name", "base_url": "https://a", "auth_token": "x",
                 "models": [{"name": "m"}]},
            ],
        )


def test_model_entry_alias_rejects_at_sign():
    with pytest.raises(ValueError, match="@"):
        Settings(
            anthropic_upstreams=[
                {"name": "u", "base_url": "https://a", "auth_token": "x",
                 "models": [{"name": "m", "alias": "bad@alias"}]},
            ],
        )


# ---------------------------------------------------------------------------
# Aliases & composite ids
# ---------------------------------------------------------------------------


def test_alias_replaces_display_name_in_composite_id():
    s = Settings(
        anthropic_upstreams=[
            {"name": "ds", "base_url": "https://d", "auth_token": "x",
             "models": [{"name": "deepseek-reasoner", "alias": "r1"}]},
        ],
    )
    assert s.all_source_composite_ids() == ["r1@ds"]


def test_duplicate_alias_in_same_source_rejected():
    with pytest.raises(ValueError, match="alias|duplicate"):
        Settings(
            anthropic_upstreams=[
                {"name": "u", "base_url": "https://a", "auth_token": "x",
                 "models": [
                     {"name": "a", "alias": "x"},
                     {"name": "b", "alias": "x"},
                 ]},
            ],
        )


# ---------------------------------------------------------------------------
# Interface exposure
# ---------------------------------------------------------------------------


def test_interface_duplicate_public_ids_rejected():
    with pytest.raises(ValueError, match="duplicate|public"):
        Settings(
            anthropic_upstreams=[
                {"name": "u", "base_url": "https://a", "auth_token": "x",
                 "models": [{"name": "m"}, {"name": "n"}]},
            ],
            ollama_interfaces=[
                {"name": "ollama", "port": 21434, "exposed_models": [
                    {"model": "m", "target": "u", "alias": "shared"},
                    {"model": "n", "target": "u", "alias": "shared"},
                ]},
            ],
        )


def test_resolve_request_routes_via_interface(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    _write(
        cfg,
        {
            "anthropic_upstreams": [
                {"name": "anthropic", "base_url": "https://a", "auth_token": "a-tok",
                 "models": [{"name": "sonnet", "alias": "claude-sonnet"}]},
                {"name": "deepseek", "base_url": "https://d", "auth_token": "d-tok",
                 "models": [{"name": "deepseek-chat"}]},
            ],
            "ollama_interfaces": [
                {"name": "ollama", "port": 21434, "exposed_models": [
                    {"model": "claude-sonnet", "target": "anthropic"},
                    {"model": "deepseek-chat", "target": "deepseek"},
                ]},
            ],
        },
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    s = load_settings()
    backend_a, real_a = s.resolve_request("claude-sonnet@anthropic", interface_name="ollama")
    assert backend_a.name == "anthropic"
    assert real_a == "claude-sonnet"
    # alias is the display name; source maps it back to wire id "sonnet"
    assert backend_a.source.resolve_model(real_a) == "sonnet"

    backend_d, real_d = s.resolve_request("deepseek-chat@deepseek", interface_name="ollama")
    assert backend_d.name == "deepseek"
    assert real_d == "deepseek-chat"


def test_resolve_request_rejects_unexposed_id():
    s = Settings(
        anthropic_upstreams=[
            {"name": "u", "base_url": "https://a", "auth_token": "x",
             "models": [{"name": "public"}, {"name": "private"}]},
        ],
        ollama_interfaces=[
            {"name": "ollama", "port": 21434, "exposed_models": [
                {"model": "public", "target": "u"},
            ]},
        ],
    )
    with pytest.raises(ValueError, match="not exposed"):
        s.resolve_request("private@u", interface_name="ollama")


def test_resolve_request_rejects_unknown_interface():
    s = Settings(
        anthropic_upstreams=[
            {"name": "u", "base_url": "https://a", "auth_token": "x",
             "models": [{"name": "m"}]},
        ],
    )
    with pytest.raises(ValueError, match="unknown interface"):
        s.resolve_request("m@u", interface_name="nope")


# ---------------------------------------------------------------------------
# llama.cpp targets
# ---------------------------------------------------------------------------


def test_llama_cpp_target_routing_via_alias():
    s = Settings(
        llama_cpp_targets=[
            {
                "base_url": "http://127.0.0.1:21436/",
                "model": "qwen3.6",
                "upstream_id": "qwen3.6-alias",
                "auto_start": True,
                "start_command": "run-qwen36",
                "idle_timeout_seconds": 1800,
                "health_path": "health",
            }
        ],
        model_profiles={
            "qwen3.6": {"context_length": 262144, "estimated_vram_gb": 18}
        },
    )
    target = s.llama_cpp_targets[0]
    assert target.name == "qwen3.6"  # source name defaults to model
    assert target.base_url == "http://127.0.0.1:21436"
    assert target.health_path == "/health"
    assert target.resolve_model("qwen3.6") == "qwen3.6-alias"
    assert s.profile_for("qwen3.6").context_length == 262144


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
            {"model": "hidden-qwen", "base_url": "http://127.0.0.1:21436",
             "start_command": "run-hidden"},
            {"model": "visible-qwen", "base_url": "http://127.0.0.1:21437",
             "auto_start": False, "health_path": "healthz"},
        ],
    )
    hidden = s.effective_llama_cpp_target(s.llama_cpp_targets[0])
    visible = s.effective_llama_cpp_target(s.llama_cpp_targets[1])
    assert hidden.auto_start is True
    assert hidden.idle_timeout_seconds == 1800
    assert hidden.startup_timeout_seconds == 600
    assert hidden.health_path == "/ready"
    assert hidden.cwd == "I:\\Projects\\llama.cpp"
    assert visible.auto_start is False
    assert visible.health_path == "/healthz"


# ---------------------------------------------------------------------------
# End-to-end smoke test through the FastAPI app
# ---------------------------------------------------------------------------


def test_routes_request_via_composite_id_through_app(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    _write(
        cfg,
        {
            "anthropic_upstreams": [
                {"name": "anthropic", "base_url": "https://anthropic.example.com",
                 "auth_token": "a-tok",
                 "models": [{"name": "sonnet"}]},
                {"name": "deepseek", "base_url": "https://deepseek.example.com",
                 "auth_token": "d-tok",
                 "models": [{"name": "dpsk"}]},
            ],
            "ollama_interfaces": [
                {"name": "ollama", "port": 21434, "exposed_models": [
                    {"model": "sonnet", "target": "anthropic"},
                    {"model": "dpsk", "target": "deepseek"},
                ]},
            ],
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
            "https://anthropic.example.com", "a-tok",
            client=httpx.AsyncClient(transport=httpx.MockTransport(make_handler("anthropic"))),
        ),
        "deepseek": AnthropicClient(
            "https://deepseek.example.com", "d-tok",
            client=httpx.AsyncClient(transport=httpx.MockTransport(make_handler("deepseek"))),
        ),
    }

    with TestClient(app) as tc:
        r1 = tc.post("/api/chat", json={
            "model": "sonnet@anthropic", "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        })
        r2 = tc.post("/api/chat", json={
            "model": "dpsk@deepseek", "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        })

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert hits["anthropic"] == ["sonnet"]
    assert hits["deepseek"] == ["dpsk"]

# ---------------------------------------------------------------------------
# model_profiles list-form + dashboard_reclaim_idle_seconds
# ---------------------------------------------------------------------------


def test_model_profiles_list_form_loads():
    s = Settings(
        model_profiles=[
            {"model": "foo", "context_length": 1000},
            {"model": "bar", "target": "openai-a", "thinking_mode": "disabled"},
        ]
    )
    assert s.profile_for("foo").context_length == 1000
    assert s.profile_for("bar@openai-a").thinking_mode == "disabled"


def test_model_profiles_list_form_rejects_missing_model():
    with pytest.raises(Exception):
        Settings(model_profiles=[{"target": "x", "context_length": 1}])


def test_model_profiles_list_form_rejects_duplicates():
    with pytest.raises(Exception):
        Settings(model_profiles=[
            {"model": "a", "target": "t", "context_length": 1},
            {"model": "a", "target": "t", "context_length": 2},
        ])


def test_model_profiles_dict_form_still_works():
    s = Settings(model_profiles={"foo@bar": {"context_length": 42}})
    assert s.profile_for("foo@bar").context_length == 42


def test_model_profiles_dump_emits_list_form():
    s = Settings(model_profiles={
        "foo": {"context_length": 1},
        "bar@baz": {"context_length": 2},
    })
    dumped = s.model_dump()["model_profiles"]
    assert isinstance(dumped, list)
    by_key = {(e["model"], e.get("target")): e for e in dumped}
    assert by_key[("foo", None)]["context_length"] == 1
    assert by_key[("bar", "baz")]["context_length"] == 2
    # Roundtrip back through Settings produces the same internal state.
    s2 = Settings(model_profiles=dumped)
    assert s2.profile_for("foo").context_length == 1
    assert s2.profile_for("bar@baz").context_length == 2


def test_dashboard_reclaim_idle_seconds_default_is_20():
    assert Settings().dashboard_reclaim_idle_seconds == 20.0


def test_dashboard_reclaim_idle_seconds_configurable():
    s = Settings(dashboard_reclaim_idle_seconds=5.0)
    assert s.dashboard_reclaim_idle_seconds == 5.0
