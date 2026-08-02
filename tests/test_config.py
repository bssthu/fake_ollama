"""Tests for the JSON-based config loader and the new interface/source API."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.config import ComfyUITarget, Settings, load_settings
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


def test_top_level_comment_keys_are_ignored():
    s = Settings.model_validate(
        {
            "_": "human-readable note",
            "_section": "another note",
            "advertised_version": "test-version",
        }
    )
    assert s.advertised_version == "test-version"


def test_config_example_loads():
    root = Path(__file__).resolve().parents[1]
    s = load_settings(root / "config.json.example")
    assert any(t.name == "z-image-turbo-comfyui" for t in s.comfyui_targets)


def test_comfyui_default_frames_must_match_workflow_formula():
    with pytest.raises(ValueError, match="default_num_frames"):
        ComfyUITarget(
            name="joy",
            model="joyai-echo",
            min_num_frames=17,
            default_num_frames=18,
            max_num_frames=241,
            num_frames_offset=1,
            num_frames_modulo=8,
        )


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
    assert s.playground_enabled is False
    assert s.playground_host == "127.0.0.1"
    assert s.playground_port == 21431
    assert s.playground_listener_enabled is False


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


def test_playground_port_must_differ_from_other_listener_ports():
    with pytest.raises(ValueError, match="playground_port.*conflicts"):
        Settings(
            playground_enabled=True,
            playground_port=21434,
            ollama_interfaces=[
                {"name": "ollama", "port": 21434, "exposed_models": []}
            ],
        )


def test_generic_openai_target_lifecycle_fields_and_routing():
    s = Settings(
        generic_openai_targets=[
            {
                "name": "vllm",
                "base_url": "http://127.0.0.1:8062/",
                "auth_token": "tok",
                "models": [
                    {
                        "name": "/models/Qwen2.5-0.5B-Instruct",
                        "alias": "qwen-small",
                        "upstream_id": "qwen-wire",
                    }
                ],
                "auto_start": True,
                "start_command": "start-vllm",
                "stop_command": "stop-vllm",
                "idle_timeout_seconds": 30,
                "startup_timeout_seconds": 240,
                "health_path": "health",
                "cwd": "I:\\Projects\\vllm",
                "max_concurrent_requests": 1,
                "request_read_timeout_seconds": 0,
            }
        ],
        ollama_interfaces=[
            {
                "name": "ollama",
                "port": 21434,
                "exposed_models": [{"model": "qwen-small", "target": "vllm"}],
            }
        ],
    )
    tgt = s.generic_openai_targets[0]
    assert tgt.base_url == "http://127.0.0.1:8062"
    assert tgt.health_path == "/health"
    assert tgt.resolve_model("qwen-small") == "qwen-wire"
    assert s.all_source_composite_ids() == ["qwen-small@vllm"]
    backend, real = s.resolve_request("qwen-small@vllm", interface_name="ollama")
    assert backend.protocol == "openai"
    assert backend.kind == "local"
    assert real == "qwen-small"


def test_legacy_local_openai_targets_migrate_to_generic_openai_targets():
    s = Settings(
        local_openai_targets=[
            {
                "name": "legacy-vllm",
                "base_url": "http://127.0.0.1:8062",
                "models": [{"name": "qwen-small"}],
            }
        ],
        ollama_interfaces=[],
    )
    assert [t.name for t in s.generic_openai_targets] == ["legacy-vllm"]


def test_legacy_and_new_generic_openai_targets_cannot_both_be_set():
    with pytest.raises(ValueError, match="local_openai_targets.*generic_openai_targets"):
        Settings(
            local_openai_targets=[
                {
                    "name": "legacy-vllm",
                    "base_url": "http://127.0.0.1:8062",
                    "models": [{"name": "qwen-small"}],
                }
            ],
            generic_openai_targets=[
                {
                    "name": "new-vllm",
                    "base_url": "http://127.0.0.1:8063",
                    "models": [{"name": "qwen-small"}],
                }
            ],
            ollama_interfaces=[],
        )


def test_generic_openai_target_requires_models():
    with pytest.raises(ValueError, match="generic_openai_target.*model"):
        Settings(
            generic_openai_targets=[
                {"name": "vllm", "base_url": "http://127.0.0.1:8062"}
            ],
            ollama_interfaces=[],
        )


def test_generic_openai_duplicate_base_url_rejected():
    with pytest.raises(ValueError, match="generic_openai_target.*same base_url"):
        Settings(
            generic_openai_targets=[
                {
                    "name": "a",
                    "base_url": "http://127.0.0.1:8062",
                    "models": [{"name": "a"}],
                },
                {
                    "name": "b",
                    "base_url": "http://127.0.0.1:8062/",
                    "models": [{"name": "b"}],
                },
            ],
            ollama_interfaces=[],
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


def test_llama_cpp_duplicate_base_url_rejected():
    with pytest.raises(ValueError, match="same base_url"):
        Settings(
            llama_cpp_targets=[
                {"model": "modelA", "base_url": "http://127.0.0.1:8080"},
                {"model": "modelB", "base_url": "http://127.0.0.1:8080"},
            ]
        )


def test_llama_cpp_duplicate_base_url_trailing_slash_normalised():
    with pytest.raises(ValueError, match="same base_url"):
        Settings(
            llama_cpp_targets=[
                {"model": "modelA", "base_url": "http://127.0.0.1:8080/"},
                {"model": "modelB", "base_url": "http://127.0.0.1:8080"},
            ]
        )


def test_llama_cpp_distinct_base_urls_ok():
    s = Settings(
        llama_cpp_targets=[
            {"model": "modelA", "base_url": "http://127.0.0.1:8080"},
            {"model": "modelB", "base_url": "http://127.0.0.1:8081"},
        ]
    )
    assert len(s.llama_cpp_targets) == 2


def test_llama_cpp_quantized_v_cache_without_flash_attn_rejected():
    """``cache_type_v`` quantized without ``flash_attn`` would crash
    llama-server with 0xC0000005 on the first request. The loader must
    reject it up front instead of letting the launch fail opaquely."""
    with pytest.raises(ValueError, match="flash_attn=true"):
        Settings(
            llama_cpp_targets=[
                {
                    "model": "qa",
                    "base_url": "http://127.0.0.1:8080",
                    "cache_type_v": "q8_0",
                }
            ]
        )


def test_llama_cpp_quantized_k_cache_without_flash_attn_rejected():
    with pytest.raises(ValueError, match="cache_type_k='q8_0'"):
        Settings(
            llama_cpp_targets=[
                {
                    "model": "qa",
                    "base_url": "http://127.0.0.1:8080",
                    "cache_type_k": "q8_0",
                }
            ]
        )


def test_llama_cpp_quantized_cache_with_flash_attn_ok():
    s = Settings(
        llama_cpp_targets=[
            {
                "model": "qa",
                "base_url": "http://127.0.0.1:8080",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": True,
            }
        ]
    )
    assert s.llama_cpp_targets[0].flash_attn is True


def test_llama_cpp_quantized_cache_inherits_flash_attn_from_defaults():
    """``flash_attn`` set on ``llama_cpp_defaults`` satisfies the check;
    the validator runs *after* defaults are folded into each target."""
    s = Settings(
        llama_cpp_defaults={"flash_attn": True},
        llama_cpp_targets=[
            {
                "model": "qa",
                "base_url": "http://127.0.0.1:8080",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            }
        ],
    )
    eff = s.effective_llama_cpp_target(s.llama_cpp_targets[0])
    assert eff.flash_attn is True


def test_llama_cpp_empty_cache_type_k_ignored():
    """Empty-string ``cache_type_k`` (the shape the admin UI emits when
    the field is cleared) must not be treated as a quantized type."""
    s = Settings(
        llama_cpp_targets=[
            {
                "model": "qa",
                "base_url": "http://127.0.0.1:8080",
                "cache_type_k": "",
                "cache_type_v": "f16",
            }
        ]
    )
    assert s.llama_cpp_targets[0].cache_type_k == ""


def test_llama_cpp_f16_cache_without_flash_attn_ok():
    s = Settings(
        llama_cpp_targets=[
            {
                "model": "qa",
                "base_url": "http://127.0.0.1:8080",
                "cache_type_k": "f16",
                "cache_type_v": "f16",
            }
        ]
    )
    assert s.llama_cpp_targets[0].cache_type_v == "f16"


# ---------------------------------------------------------------------------
# Asymmetric K/V cache types — slow generic CUDA path
# ---------------------------------------------------------------------------


def test_llama_cpp_only_v_quantized_rejected_even_with_flash_attn():
    """K defaults to f16 when omitted, so ``cache_type_v="q8_0"`` alone
    leaves the kernel running an asymmetric f16/q8_0 attention path."""
    with pytest.raises(ValueError, match="resolve to different types"):
        Settings(
            llama_cpp_targets=[
                {
                    "model": "qa",
                    "base_url": "http://127.0.0.1:8080",
                    "cache_type_v": "q8_0",
                    "flash_attn": True,
                }
            ]
        )


def test_llama_cpp_empty_k_with_quantized_v_rejected():
    """The admin UI emits ``cache_type_k=""`` when the field is cleared.
    Combined with a quantized V, that is still asymmetric and must be
    rejected even though K is not ``None``."""
    with pytest.raises(ValueError, match="resolve to different types"):
        Settings(
            llama_cpp_targets=[
                {
                    "model": "qa",
                    "base_url": "http://127.0.0.1:8080",
                    "cache_type_k": "",
                    "cache_type_v": "q8_0",
                    "flash_attn": True,
                }
            ]
        )


def test_llama_cpp_mixed_quant_types_rejected():
    with pytest.raises(ValueError, match="resolve to different types"):
        Settings(
            llama_cpp_targets=[
                {
                    "model": "qa",
                    "base_url": "http://127.0.0.1:8080",
                    "cache_type_k": "q4_0",
                    "cache_type_v": "q8_0",
                    "flash_attn": True,
                }
            ]
        )


def test_llama_cpp_explicit_f16_k_with_missing_v_ok():
    """``f16`` is llama-server's default for an omitted ``-ctv``, so an
    explicit ``cache_type_k="f16"`` paired with no V is still symmetric."""
    s = Settings(
        llama_cpp_targets=[
            {
                "model": "qa",
                "base_url": "http://127.0.0.1:8080",
                "cache_type_k": "f16",
            }
        ]
    )
    assert s.llama_cpp_targets[0].cache_type_k == "f16"


def test_llama_cpp_matched_quant_cache_with_flash_attn_ok():
    s = Settings(
        llama_cpp_targets=[
            {
                "model": "qa",
                "base_url": "http://127.0.0.1:8080",
                "cache_type_k": "q5_1",
                "cache_type_v": "q5_1",
                "flash_attn": True,
            }
        ]
    )
    assert s.llama_cpp_targets[0].cache_type_k == "q5_1"
    assert s.llama_cpp_targets[0].cache_type_v == "q5_1"


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
