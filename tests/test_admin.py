"""Tests for the /admin web UI and config-reload endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.config import Settings, load_settings
from fake_ollama.server import create_app


def _admin_client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings), base_url="http://testserver:21433")


@pytest.fixture
def admin_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": 21434,
                "upstreams": [
                    {
                        "name": "default",
                        "base_url": "http://upstream.test",
                        "auth_token": "tok",
                        "models": ["claude-3-5-sonnet-20241022"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg_path))
    return load_settings(config_path=str(cfg_path))


def test_admin_index_html(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "fake-ollama config editor" in resp.text


def test_admin_index_no_trailing_slash(admin_settings):
    """Regression: /admin (no slash) must also serve the page so that the
    JS's relative URL resolution still hits /admin/config not /config."""
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin", follow_redirects=False)
    # Either it directly serves (200) or redirects to /admin/.
    assert resp.status_code in (200, 307, 308)


def test_admin_schema(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin/schema")
    assert resp.status_code == 200
    schema = resp.json()
    fields = schema["fields"]
    keys = {f["key"] for f in fields}
    assert {
        "host",
        "port",
        "admin_host",
        "admin_port",
        "upstreams",
        "ollama_targets",
        "llama_cpp_targets",
        "model_profiles",
    } <= keys
    upstreams = next(f for f in fields if f["key"] == "upstreams")
    assert upstreams["required"] is True
    assert upstreams["type"] == "object_list"
    assert {f["key"] for f in upstreams["item_schema"]} >= {"name", "base_url"}
    upstream_models = next(f for f in upstreams["item_schema"] if f["key"] == "models")
    assert upstream_models["autocomplete"] == "model_names"
    # ollama_target items no longer carry per-target api_token.
    ollama = next(f for f in fields if f["key"] == "ollama_targets")
    item_keys = {f["key"] for f in ollama["item_schema"]}
    assert "api_token" not in item_keys
    assert {
        "name",
        "base_url",
        "models",
        "auto_start",
        "start_command",
        "idle_timeout_seconds",
        "health_path",
    } <= item_keys
    ollama_models = next(f for f in ollama["item_schema"] if f["key"] == "models")
    assert ollama_models["autocomplete"] == "model_names"
    llama_cpp = next(f for f in fields if f["key"] == "llama_cpp_targets")
    assert llama_cpp["detect_models"] == "llama_cpp"
    llama_item_keys = {f["key"] for f in llama_cpp["item_schema"]}
    assert {
        "name",
        "base_url",
        "auth_token",
        "models",
        "auto_start",
        "start_command",
        "idle_timeout_seconds",
    } <= llama_item_keys
    llama_models = next(f for f in llama_cpp["item_schema"] if f["key"] == "models")
    assert llama_models["autocomplete"] == "model_names"
    # external_access_tokens lives at the Settings level with secret_each + generate_each.
    ext = next(f for f in fields if f["key"] == "external_access_tokens")
    assert ext["type"] == "string_list"
    assert ext["secret_each"] is True
    assert ext["generate_each"] is True
    # The config page is organized by proxy direction first, then shared/admin.
    group_order = [g["key"] for g in schema["groups"]]
    assert group_order == [
        "forward_listener",
        "forward_upstreams",
        "reverse_listener",
        "reverse_ollama",
        "reverse_llamacpp",
        "shared_runtime",
        "profiles",
        "admin",
    ]
    section_order = []
    for group in schema["groups"]:
        if group["section"] not in section_order:
            section_order.append(group["section"])
    assert section_order == ["forward", "reverse", "shared", "admin"]
    field_order = [f["key"] for f in fields]
    assert field_order[:4] == ["host", "port", "advertised_version", "upstreams"]
    assert field_order[4:7] == ["external_host", "external_port", "external_access_tokens"]
    assert field_order[-4] == "model_profiles"
    assert field_order[-3:] == ["admin_enabled", "admin_host", "admin_port"]
    group_keys = set(group_order)
    for f in fields:
        assert f["group"] in group_keys


def test_admin_get_config(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["upstreams"][0]["name"] == "default"
    assert body["_path"].endswith("config.json")


def test_admin_put_config_persists_and_reloads(admin_settings, tmp_path: Path):
    app = create_app(admin_settings)
    client = TestClient(app, base_url="http://testserver:21433")
    new_cfg = {
        "host": "127.0.0.1",
        "port": 21999,
        "upstreams": [
            {
                "name": "newup",
                "base_url": "http://other.test",
                "auth_token": "tok2",
                "models": ["another-model"],
            }
        ],
        "ollama_targets": [
            {"name": "local", "base_url": "http://127.0.0.1:11434",
             "models": ["llama3.1"]}
        ],
        "llama_cpp_targets": [
            {"name": "qwen36", "base_url": "http://127.0.0.1:21436",
             "models": ["qwen3.6"], "auto_start": False}
        ],
    }
    with client:
        resp = client.put("/admin/config", json=new_cfg)
        assert resp.status_code == 200, resp.text
        # In-memory settings must reflect the change.
        assert app.state.settings.port == 21999
        assert app.state.settings.upstreams[0].name == "newup"
        assert "newup" in app.state.clients
        assert "local" in app.state.ollama_clients
        assert "qwen36" in app.state.llama_cpp_clients
        # File on disk must reflect the change.
        cfg_path = Path(admin_settings.config_path)
        on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert on_disk["port"] == 21999


def test_admin_put_invalid_returns_400(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        # Missing required upstreams -> Settings validator raises.
        resp = client.put("/admin/config", json={"host": "127.0.0.1", "upstreams": []})
    assert resp.status_code == 400


def test_admin_disabled_returns_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(
        json.dumps(
            {
                "admin_enabled": False,
                "upstreams": [
                    {
                        "name": "u",
                        "base_url": "http://x.test",
                        "auth_token": "t",
                        "models": ["m"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg_path))
    s = load_settings(config_path=str(cfg_path))
    client = _admin_client(s)
    with client:
        resp = client.get("/admin/")
    assert resp.status_code == 404


def test_admin_routes_are_only_on_admin_port(admin_settings):
    app = create_app(admin_settings)
    with TestClient(app, base_url="http://testserver:21434") as internal:
        assert internal.get("/admin/").status_code == 404
        assert internal.get("/api/version").status_code == 200
    with TestClient(app, base_url="http://testserver:21433") as admin:
        assert admin.get("/admin/").status_code == 200
        assert admin.get("/api/version").status_code == 404


def test_admin_probe_models_ollama(admin_settings, monkeypatch: pytest.MonkeyPatch):
    """Probe an Ollama-style /api/tags endpoint via /admin/probe-models."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url).endswith("/api/tags")
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3.1:8b"},
                    {"name": "qwen2.5-coder:7b"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs.pop("trust_env", None)
        return real_client_cls(**kwargs)

    monkeypatch.setattr("fake_ollama.admin.httpx.AsyncClient", fake_async_client)

    client = _admin_client(admin_settings)
    with client:
        resp = client.post(
            "/admin/probe-models",
            json={"kind": "ollama", "base_url": "http://127.0.0.1:11434"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"models": ["llama3.1:8b", "qwen2.5-coder:7b"]}


def test_admin_probe_models_anthropic(admin_settings, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"data": [{"id": "claude-3-5-sonnet-20241022"}, {"id": "claude-3-5-haiku-20241022"}]},
        )

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs.pop("trust_env", None)
        return real_client_cls(**kwargs)

    monkeypatch.setattr("fake_ollama.admin.httpx.AsyncClient", fake_async_client)

    client = _admin_client(admin_settings)
    with client:
        resp = client.post(
            "/admin/probe-models",
            json={
                "kind": "anthropic",
                "base_url": "https://api.anthropic.com",
                "auth_token": "sk-ant-test",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["models"] == [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]
    # Auth header forwarded.
    assert captured["headers"].get("x-api-key") == "sk-ant-test"
    assert captured["url"].endswith("/v1/models")


def test_admin_probe_models_llama_cpp(admin_settings, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "qwen3.6-27b-hauhau-q2kp", "object": "model"}
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs.pop("trust_env", None)
        return real_client_cls(**kwargs)

    monkeypatch.setattr("fake_ollama.admin.httpx.AsyncClient", fake_async_client)

    client = _admin_client(admin_settings)
    with client:
        resp = client.post(
            "/admin/probe-models",
            json={
                "kind": "llama_cpp",
                "base_url": "http://127.0.0.1:21436",
                "auth_token": "local-key",
            },
        )
    assert resp.status_code == 200
    assert resp.json() == {"models": ["qwen3.6-27b-hauhau-q2kp"]}
    assert captured["headers"].get("authorization") == "Bearer local-key"
    assert captured["url"].endswith("/v1/models")


def test_admin_probe_models_missing_base_url(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.post("/admin/probe-models", json={"kind": "ollama"})
    assert resp.status_code == 400
