"""Tests for the unified ``Backend`` view and the new request-resolution API."""

from __future__ import annotations

import pytest

from fake_ollama.config import (
    AnthropicUpstream,
    ApiInterface,
    Backend,
    ExposureEntry,
    LlamaCppDefaults,
    LlamaCppTarget,
    ModelEntry,
    OllamaInterface,
    OllamaTarget,
    OpenAIUpstream,
    Settings,
)


# ---------------------------------------------------------------------------
# Backend.from_source
# ---------------------------------------------------------------------------


def _settings_with(**kw):
    """Build a Settings without requiring any interfaces."""
    kw.setdefault("ollama_interfaces", [])
    return Settings(**kw)


def test_backend_from_anthropic_upstream_is_remote() -> None:
    up = AnthropicUpstream(
        name="claude",
        base_url="https://api.anthropic.com",
        auth_token="sk-ant",
        models=[ModelEntry(name="claude-3-5-sonnet")],
    )
    s = _settings_with(anthropic_upstreams=[up])
    b = Backend.from_source(up, s)
    assert b.name == "claude"
    assert b.protocol == "anthropic"
    assert b.kind == "remote"
    assert b.base_url == "https://api.anthropic.com"
    assert b.auth_token == "sk-ant"
    assert b.models == ["claude-3-5-sonnet"]
    assert b.source is up
    assert b.supports_lifecycle is False
    assert b.auto_start is False


def test_backend_from_ollama_target_remote_when_no_lifecycle() -> None:
    tgt = OllamaTarget(
        name="local",
        base_url="http://127.0.0.1:11434",
        models=[ModelEntry(name="llama3.1")],
    )
    s = _settings_with(ollama_targets=[tgt])
    b = Backend.from_source(tgt, s)
    assert b.protocol == "ollama"
    assert b.kind == "remote"
    assert b.supports_lifecycle is False


def test_backend_from_ollama_target_local_when_auto_start() -> None:
    tgt = OllamaTarget(
        name="local",
        base_url="http://127.0.0.1:11434",
        models=[ModelEntry(name="llama3.1")],
        auto_start=True,
        start_command="ollama serve",
    )
    s = _settings_with(ollama_targets=[tgt])
    b = Backend.from_source(tgt, s)
    assert b.kind == "local"
    assert b.supports_lifecycle is True
    assert b.auto_start is True


def test_backend_from_llama_cpp_target_is_always_local_openai() -> None:
    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:8080",
        auth_token="tok",
        model="qwen3",
    )
    s = _settings_with(llama_cpp_targets=[tgt])
    b = Backend.from_source(tgt, s)
    assert b.name == "qwen3"
    assert b.protocol == "openai"
    assert b.kind == "local"
    assert b.auth_token == "tok"
    assert b.models == ["qwen3"]


def test_backend_serves_checks_display_name() -> None:
    up = AnthropicUpstream(
        name="u", base_url="http://u", auth_token="x",
        models=[ModelEntry(name="a"), ModelEntry(name="b")],
    )
    s = _settings_with(anthropic_upstreams=[up])
    b = Backend.from_source(up, s)
    assert b.serves("a") is True
    assert b.serves("missing") is False


def test_alias_overrides_display_name() -> None:
    up = AnthropicUpstream(
        name="u", base_url="http://u", auth_token="x",
        models=[ModelEntry(name="real-model", alias="pretty")],
    )
    s = _settings_with(anthropic_upstreams=[up])
    b = Backend.from_source(up, s)
    assert b.models == ["pretty"]
    assert b.serves("pretty") is True
    assert b.serves("real-model") is False
    # resolve_model maps display alias back to the wire id
    assert b.resolve_model("pretty") == "real-model"


# ---------------------------------------------------------------------------
# Settings.backend_by_name + resolve_request
# ---------------------------------------------------------------------------


def test_backend_by_name_finds_each_kind() -> None:
    settings = _settings_with(
        anthropic_upstreams=[
            AnthropicUpstream(name="up", base_url="http://up", auth_token="x",
                              models=[ModelEntry(name="m")]),
        ],
        ollama_targets=[
            OllamaTarget(name="ot", base_url="http://ot",
                         models=[ModelEntry(name="m2")]),
        ],
        llama_cpp_targets=[
            LlamaCppTarget(base_url="http://lc", model="lcm"),
        ],
    )
    assert settings.backend_by_name("up").protocol == "anthropic"
    assert settings.backend_by_name("ot").protocol == "ollama"
    assert settings.backend_by_name("lcm").protocol == "openai"
    assert settings.backend_by_name("nope") is None


def test_resolve_request_via_interface_uses_composite_or_alias() -> None:
    settings = Settings(
        anthropic_upstreams=[
            AnthropicUpstream(name="up", base_url="http://up", auth_token="x",
                              models=[ModelEntry(name="m")]),
        ],
        ollama_targets=[
            OllamaTarget(name="ot", base_url="http://ot",
                         models=[ModelEntry(name="m")]),
        ],
        ollama_interfaces=[
            OllamaInterface(
                name="ollama", port=21434,
                exposed_models=[
                    ExposureEntry(model="m", target="up"),
                    ExposureEntry(model="m", target="ot", alias="m-local"),
                ],
            ),
        ],
    )
    # composite id picks upstream
    backend, real = settings.resolve_request("m@up", interface_name="ollama")
    assert backend.protocol == "anthropic"
    assert real == "m"
    # alias picks the ollama target
    backend, real = settings.resolve_request("m-local", interface_name="ollama")
    assert backend.protocol == "ollama"
    assert real == "m"


def test_resolve_request_rejects_unexposed_id() -> None:
    settings = Settings(
        anthropic_upstreams=[
            AnthropicUpstream(name="up", base_url="http://up", auth_token="x",
                              models=[ModelEntry(name="public"), ModelEntry(name="private")]),
        ],
        ollama_interfaces=[
            OllamaInterface(
                name="ollama", port=21434,
                exposed_models=[ExposureEntry(model="public", target="up")],
            ),
        ],
    )
    with pytest.raises(ValueError):
        settings.resolve_request("private@up", interface_name="ollama")
    backend, real = settings.resolve_request("public@up", interface_name="ollama")
    assert backend.name == "up"
    assert real == "public"


def test_resolve_request_unknown_interface_raises() -> None:
    settings = _settings_with(
        anthropic_upstreams=[
            AnthropicUpstream(name="up", base_url="http://up", auth_token="x",
                              models=[ModelEntry(name="m")]),
        ],
    )
    with pytest.raises(ValueError):
        settings.resolve_request("m@up", interface_name="nope")


def test_llama_cpp_defaults_applied() -> None:
    settings = _settings_with(
        llama_cpp_defaults=LlamaCppDefaults(auto_start=True, idle_timeout_seconds=900.0),
        llama_cpp_targets=[LlamaCppTarget(base_url="http://lc", model="m")],
    )
    eff = settings.effective_llama_cpp_target(settings.llama_cpp_targets[0])
    assert eff.auto_start is True
    assert eff.idle_timeout_seconds == 900.0
