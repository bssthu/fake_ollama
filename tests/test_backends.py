"""Tests for the unified ``Backend`` view layer in ``config.py``."""

from __future__ import annotations

import pytest

from fake_ollama.config import (
    Backend,
    LlamaCppDefaults,
    LlamaCppTarget,
    OllamaTarget,
    Settings,
    Upstream,
)


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


def test_backend_from_anthropic_upstream_is_remote() -> None:
    up = Upstream(
        name="claude",
        base_url="https://api.anthropic.com",
        auth_token="sk-ant",
        models=["claude-3-5-sonnet"],
    )
    b = Backend.from_anthropic_upstream(up)
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
        models=["llama3.1"],
    )
    b = Backend.from_ollama_target(tgt)
    assert b.protocol == "ollama"
    assert b.kind == "remote"
    assert b.supports_lifecycle is False


def test_backend_from_ollama_target_local_when_auto_start() -> None:
    tgt = OllamaTarget(
        name="local",
        base_url="http://127.0.0.1:11434",
        models=["llama3.1"],
        auto_start=True,
        start_command="ollama serve",
    )
    b = Backend.from_ollama_target(tgt)
    assert b.kind == "local"
    assert b.supports_lifecycle is True
    assert b.auto_start is True


def test_backend_from_llama_cpp_target_is_always_local_openai() -> None:
    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:8080",
        auth_token="tok",
        model="qwen3",
    )
    b = Backend.from_llama_cpp_target(tgt)
    assert b.name == "qwen3"
    assert b.protocol == "openai"
    assert b.kind == "local"
    assert b.auth_token == "tok"
    assert b.models == ["qwen3"]


def test_backend_serves_checks_membership() -> None:
    b = Backend.from_anthropic_upstream(
        Upstream(name="u", base_url="http://u", auth_token="x", models=["a", "b"])
    )
    assert b.serves("a") is True
    assert b.serves("missing") is False


# ---------------------------------------------------------------------------
# Settings.backends aggregation + ordering
# ---------------------------------------------------------------------------


def test_settings_backends_order_and_protocols() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="u1", base_url="http://u1", auth_token="x", models=["m1"]),
        ],
        ollama_targets=[
            OllamaTarget(name="o1", base_url="http://o1", models=["m2"]),
        ],
        llama_cpp_targets=[
            LlamaCppTarget(base_url="http://lc1", model="m3"),
        ],
    )
    protocols = [(b.name, b.protocol, b.kind) for b in settings.backends]
    # Routing priority order: ollama → llama.cpp → anthropic.
    assert protocols == [
        ("o1", "ollama", "remote"),
        ("m3", "openai", "local"),
        ("u1", "anthropic", "remote"),
    ]


def test_settings_backends_applies_llama_cpp_defaults() -> None:
    settings = Settings(
        llama_cpp_defaults=LlamaCppDefaults(
            auto_start=True, idle_timeout_seconds=900.0
        ),
        llama_cpp_targets=[LlamaCppTarget(base_url="http://lc", model="m")],
    )
    backends = [b for b in settings.backends if b.protocol == "openai"]
    assert len(backends) == 1
    src = backends[0].source
    assert src.auto_start is True
    assert src.idle_timeout_seconds == 900.0


# ---------------------------------------------------------------------------
# Composite-id lookup
# ---------------------------------------------------------------------------


def test_backend_by_name_finds_each_kind() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["m"]),
        ],
        ollama_targets=[
            OllamaTarget(name="ot", base_url="http://ot", models=["m"]),
        ],
        llama_cpp_targets=[
            LlamaCppTarget(base_url="http://lc", model="lcm"),
        ],
    )
    assert settings.backend_by_name("up").protocol == "anthropic"
    assert settings.backend_by_name("ot").protocol == "ollama"
    assert settings.backend_by_name("lcm").protocol == "openai"
    assert settings.backend_by_name("nope") is None


def test_backend_for_uses_composite_id_to_disambiguate_duplicate_model() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["dup"]),
        ],
        ollama_targets=[
            OllamaTarget(name="ot", base_url="http://ot", models=["dup"]),
        ],
    )
    # Composite ids let callers pick exactly which backend serves the
    # duplicate name — no implicit priority.
    via_ollama = settings.backend_for("dup@ot")
    via_upstream = settings.backend_for("dup@up")
    assert via_ollama is not None and via_ollama.protocol == "ollama"
    assert via_upstream is not None and via_upstream.protocol == "anthropic"


def test_backend_for_returns_none_for_unknown_target() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["m"]),
        ],
    )
    assert settings.backend_for("m@nope") is None


def test_backend_for_returns_none_when_target_does_not_serve_model() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["a"]),
        ],
    )
    assert settings.backend_for("b@up") is None


def test_backend_for_returns_none_on_bare_model() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["m"]),
        ],
    )
    # ``backend_for`` requires composite ids; bare names → None.
    assert settings.backend_for("m") is None


# ---------------------------------------------------------------------------
# Surface-aware exposure (now driven by Settings.{internal,external}_exposed_models)
# ---------------------------------------------------------------------------


def test_backend_for_external_surface_skips_unexposed() -> None:
    settings = Settings(
        upstreams=[
            Upstream(
                name="up",
                base_url="http://up",
                auth_token="x",
                models=["public", "private"],
            ),
        ],
        internal_exposed_models=["public@up", "private@up"],
        external_exposed_models=["public@up"],
    )
    assert settings.backend_for("private@up", surface="internal") is not None
    assert settings.backend_for("private@up", surface="external") is None
    assert settings.backend_for("public@up", surface="external") is not None


def test_backend_for_no_surface_ignores_exposure() -> None:
    settings = Settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["m"]),
        ],
        # nothing exposed
        internal_exposed_models=[],
        external_exposed_models=[],
    )
    # Without ``surface`` the exposure check is skipped.
    assert settings.backend_for("m@up") is not None
