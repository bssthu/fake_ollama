"""Unit tests for the model-graph cycle detector and forwarded-by header.

The cycle detector lives in :func:`fake_ollama.config.Settings.detect_upstream_cycles`
and runs in :func:`Settings.validate_settings_post`. It validates that no
self-referencing chain of exposures would force a request to loop forever
through this same process.

The runtime safety net is :class:`fake_ollama.server.ForwardedCycleMiddleware`,
which short-circuits any inbound request that already carries our
``INSTANCE_ID`` in the ``x-fake-ollama-forwarded-by`` header.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from fake_ollama import config as cfg
from fake_ollama.config import (
    AnthropicUpstream,
    ApiInterface,
    ExposureEntry,
    FORWARDED_BY_HEADER,
    INSTANCE_ID,
    ModelEntry,
    OllamaInterface,
    OpenAIUpstream,
    Settings,
    outbound_cycle_headers,
    outbound_forwarded_chain,
    parse_forwarded_chain,
)


# ---------------------------------------------------------------------------
# Static detector
# ---------------------------------------------------------------------------


def _ollama_iface(port: int, exposures, name: str = "ollama") -> OllamaInterface:
    return OllamaInterface(
        name=name, host="127.0.0.1", port=port, exposed_models=exposures
    )


def test_case1_alias_chain_allowed_with_warning(caplog) -> None:
    """Different alias at each hop is fine — log a WARNING, don't raise.

    iface ``front`` exposes ``front-name`` -> upstream ``loop`` whose
    base_url loops back to iface ``back`` on the same process, where
    we expose ``back-name`` -> a real external Anthropic upstream.
    The wire-side model id is ``back-name`` on the second hop, which
    is distinct from ``front-name``, so this is *not* a true loop.
    """
    front = _ollama_iface(
        29501,
        [ExposureEntry(model="back-name", target="loop", alias="front-name")],
        name="front",
    )
    back = _ollama_iface(
        29502,
        [ExposureEntry(model="claude-x", target="anthropic", alias="back-name")],
        name="back",
    )
    loop_up = OpenAIUpstream(
        name="loop",
        base_url="http://127.0.0.1:29502",
        auth_token="x",
        models=[ModelEntry(name="back-name")],
    )
    ant = AnthropicUpstream(
        name="anthropic",
        base_url="https://api.example.com",
        auth_token="ak",
        models=[ModelEntry(name="claude-x")],
    )
    with caplog.at_level(logging.WARNING, logger="fake_ollama"):
        Settings(
            ollama_interfaces=[front, back],
            openai_upstreams=[loop_up],
            anthropic_upstreams=[ant],
        )
    assert any(
        "self-referential" in r.message and "linear" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_case2_same_model_name_cycle_blocked() -> None:
    """Reuse the same public id on both hops -> hard error."""
    front = _ollama_iface(
        29503,
        [ExposureEntry(model="qwen3", target="loop")],
        name="front",
    )
    back = _ollama_iface(
        29504,
        [ExposureEntry(model="qwen3", target="loop2")],
        name="back",
    )
    loop_up = OpenAIUpstream(
        name="loop",
        base_url="http://127.0.0.1:29504",
        auth_token="x",
        models=[ModelEntry(name="qwen3")],
    )
    loop2 = OpenAIUpstream(
        name="loop2",
        base_url="http://127.0.0.1:29503",
        auth_token="x",
        models=[ModelEntry(name="qwen3")],
    )
    with pytest.raises(ValueError, match="cycle detected"):
        Settings(
            ollama_interfaces=[front, back],
            openai_upstreams=[loop_up, loop2],
        )


def test_latest_tag_normalization_collision() -> None:
    """``foo`` and ``foo:latest`` must be folded together for cycle math.

    The next-hop wire id is whatever the source advertises. Here ``loop``
    advertises ``qwen3:latest`` and ``loop2`` advertises ``qwen3``;
    after normalisation both nodes collapse to ``(iface, qwen3)`` and the
    DFS sees the cycle.
    """
    front = _ollama_iface(
        29505,
        [ExposureEntry(model="qwen3:latest", target="loop")],
        name="front",
    )
    back = _ollama_iface(
        29506,
        [ExposureEntry(model="qwen3", target="loop2")],
        name="back",
    )
    loop_up = OpenAIUpstream(
        name="loop",
        base_url="http://127.0.0.1:29506",
        auth_token="x",
        models=[ModelEntry(name="qwen3:latest")],
    )
    loop2 = OpenAIUpstream(
        name="loop2",
        base_url="http://127.0.0.1:29505",
        auth_token="x",
        models=[ModelEntry(name="qwen3")],
    )
    with pytest.raises(ValueError, match="cycle detected"):
        Settings(
            ollama_interfaces=[front, back],
            openai_upstreams=[loop_up, loop2],
        )


def test_upstream_pointing_at_admin_port_is_blocked() -> None:
    """A base_url that lands on admin/dashboard is always an outright error."""
    iface = _ollama_iface(
        29507,
        [ExposureEntry(model="qwen3", target="loop")],
        name="front",
    )
    loop_up = OpenAIUpstream(
        name="loop",
        base_url="http://127.0.0.1:29508",  # admin_port below
        auth_token="x",
        models=[ModelEntry(name="qwen3")],
    )
    with pytest.raises(ValueError, match="admin/dashboard"):
        Settings(
            ollama_interfaces=[iface],
            openai_upstreams=[loop_up],
            admin_enabled=True,
            admin_port=29508,
        )


# ---------------------------------------------------------------------------
# Runtime header helpers
# ---------------------------------------------------------------------------


def test_outbound_forwarded_chain_appends_instance_id(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "INSTANCE_ID", "abc123")
    token = cfg.set_inbound_forwarded_chain(("upstream1", "upstream2"))
    try:
        chain = outbound_forwarded_chain()
        assert chain.endswith("abc123")
        assert chain.split(",") == ["upstream1", "upstream2", "abc123"]
        headers = outbound_cycle_headers()
        assert headers[FORWARDED_BY_HEADER] == chain
    finally:
        cfg.reset_inbound_forwarded_chain(token)


def test_parse_forwarded_chain_strips_blanks() -> None:
    assert parse_forwarded_chain(" a , , b ,c") == ("a", "b", "c")
    assert parse_forwarded_chain("") == ()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _minimal_settings() -> Settings:
    return Settings(
        anthropic_upstreams=[
            AnthropicUpstream(
                name="anthropic",
                base_url="http://anthropic.test",
                auth_token="ak",
                models=[ModelEntry(name="claude-x")],
            )
        ],
        ollama_interfaces=[
            OllamaInterface(
                name="ollama",
                host="127.0.0.1",
                port=29599,
                exposed_models=[
                    ExposureEntry(model="claude-x", target="anthropic")
                ],
            )
        ],
        api_interfaces=[
            ApiInterface(
                name="api",
                host="127.0.0.1",
                port=29600,
                exposed_models=[
                    ExposureEntry(model="claude-x", target="anthropic")
                ],
            )
        ],
    )


def test_middleware_blocks_self_loop_with_508() -> None:
    from fake_ollama.server import create_app

    app = create_app(_minimal_settings())
    with TestClient(app) as client:
        resp = client.get(
            "/api/tags",
            headers={FORWARDED_BY_HEADER: f"upstream-x,{INSTANCE_ID}"},
        )
    assert resp.status_code == 508
    body = resp.json()
    assert body["instance_id"] == INSTANCE_ID
    assert INSTANCE_ID in body["chain"]


def test_middleware_passes_through_non_loop_requests() -> None:
    from fake_ollama.server import create_app

    app = create_app(_minimal_settings())
    with TestClient(app) as client:
        resp = client.get(
            "/api/tags",
            headers={FORWARDED_BY_HEADER: "some-other-instance"},
        )
    assert resp.status_code == 200


def test_middleware_ignores_non_model_paths() -> None:
    """Dashboard / admin paths must not be subject to the loop check."""
    from fake_ollama.server import create_app

    app = create_app(_minimal_settings())
    with TestClient(app) as client:
        # Even with our own id in the header, a non /api|/v1 path is allowed.
        resp = client.get(
            "/",
            headers={FORWARDED_BY_HEADER: INSTANCE_ID},
        )
    # Root may be 200 or 404 depending on routes, but never the 508 we raise.
    assert resp.status_code != 508
