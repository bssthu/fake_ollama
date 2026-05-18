"""Tests for the unified ``Backend`` view layer in ``config.py``.

The ``Backend`` view is a transitional façade that lets routing code
treat ``upstreams`` / ``ollama_targets`` / ``llama_cpp_targets`` as a
single protocol-tagged list. These tests pin the semantics so the
upcoming ``server.py`` routing refactor and the OpenAI backend addition
land against a stable contract.
"""

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


def _make_settings(
    *,
    upstreams=None,
    ollama_targets=None,
    llama_cpp_targets=None,
    llama_cpp_defaults=None,
) -> Settings:
    return Settings(
        upstreams=upstreams or [
            Upstream(
                name="anthropic",
                base_url="http://upstream.test",
                auth_token="t",
                models=["claude-3-5-sonnet"],
            ),
        ],
        ollama_targets=ollama_targets or [],
        llama_cpp_targets=llama_cpp_targets or [],
        llama_cpp_defaults=llama_cpp_defaults or LlamaCppDefaults(),
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
    # name falls back to model when target.name is empty
    assert b.name == "qwen3"
    assert b.protocol == "openai"
    assert b.kind == "local"
    assert b.auth_token == "tok"
    assert b.models == ["qwen3"]


# ---------------------------------------------------------------------------
# Settings.backends aggregation + ordering
# ---------------------------------------------------------------------------


def test_settings_backends_order_and_protocols() -> None:
    settings = _make_settings(
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
    # Routing priority: ollama → llama.cpp → anthropic.
    assert protocols == [
        ("o1", "ollama", "remote"),
        ("m3", "openai", "local"),
        ("u1", "anthropic", "remote"),
    ]


def test_settings_backends_applies_llama_cpp_defaults() -> None:
    settings = _make_settings(
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
# backend_for routing
# ---------------------------------------------------------------------------


def test_backend_for_prefers_ollama_target_over_upstream() -> None:
    settings = _make_settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["dup"]),
        ],
        ollama_targets=[
            OllamaTarget(name="ot", base_url="http://ot", models=["dup"]),
        ],
    )
    b = settings.backend_for("dup")
    assert b is not None and b.protocol == "ollama" and b.name == "ot"


def test_backend_for_prefers_llama_cpp_over_upstream() -> None:
    settings = _make_settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["dup"]),
        ],
        llama_cpp_targets=[
            LlamaCppTarget(base_url="http://lc", model="dup"),
        ],
    )
    b = settings.backend_for("dup")
    assert b is not None and b.protocol == "openai" and b.kind == "local"


def test_backend_for_falls_back_to_upstream_when_no_target_serves() -> None:
    settings = _make_settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["only"]),
        ],
        ollama_targets=[
            OllamaTarget(name="ot", base_url="http://ot", models=["other"]),
        ],
    )
    b = settings.backend_for("only")
    assert b is not None and b.protocol == "anthropic" and b.name == "up"


def test_backend_for_returns_none_when_no_one_serves() -> None:
    settings = _make_settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["a"]),
        ],
    )
    assert settings.backend_for("not-served") is None


# ---------------------------------------------------------------------------
# Surface-aware exposure
# ---------------------------------------------------------------------------


def test_backend_for_external_surface_skips_hidden_models() -> None:
    settings = _make_settings(
        upstreams=[
            Upstream(
                name="up",
                base_url="http://up",
                auth_token="x",
                models=["public", "private"],
                expose_external=["public"],
            ),
        ],
    )
    # Internal sees both; external only sees the whitelisted one.
    assert settings.backend_for("private", surface="internal") is not None
    assert settings.backend_for("private", surface="external") is None
    assert settings.backend_for("public", surface="external") is not None


def test_backend_for_external_skips_hidden_llama_cpp() -> None:
    settings = _make_settings(
        llama_cpp_targets=[
            LlamaCppTarget(base_url="http://lc", model="m", expose_external=False),
        ],
    )
    assert settings.backend_for("m", surface="internal") is not None
    assert settings.backend_for("m", surface="external") is None


def test_backend_for_external_skips_hidden_ollama_target() -> None:
    settings = _make_settings(
        ollama_targets=[
            OllamaTarget(
                name="ot",
                base_url="http://ot",
                models=["pub", "priv"],
                expose_external=["pub"],
            ),
        ],
    )
    assert settings.backend_for("priv", surface="internal") is not None
    assert settings.backend_for("priv", surface="external") is None


def test_backend_for_unknown_surface_raises() -> None:
    settings = _make_settings(
        upstreams=[
            Upstream(name="up", base_url="http://up", auth_token="x", models=["m"]),
        ],
    )
    with pytest.raises(ValueError):
        settings.backend_for("m", surface="bogus")  # type: ignore[arg-type]
