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
    # Admin listener default port is 21433.
    return TestClient(create_app(settings), base_url="http://testserver:21433")


@pytest.fixture
def admin_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "anthropic_upstreams": [
                    {
                        "name": "default",
                        "base_url": "http://upstream.test",
                        "auth_token": "tok",
                        "models": [{"name": "claude-3-5-sonnet-20241022"}],
                    }
                ],
                "ollama_interfaces": [
                    {
                        "name": "ollama",
                        "host": "127.0.0.1",
                        "port": 21434,
                        "access_tokens": [],
                        "exposed_models": [
                            {"model": "claude-3-5-sonnet-20241022", "target": "default"}
                        ],
                    }
                ],
                "api_interfaces": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg_path))
    return load_settings(config_path=str(cfg_path))


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def test_admin_index_html(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "fake-ollama config editor" in resp.text


def test_admin_index_no_trailing_slash(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin", follow_redirects=False)
    assert resp.status_code in (200, 307, 308)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_admin_schema(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin/schema")
    assert resp.status_code == 200
    schema = resp.json()
    fields = schema["fields"]
    keys = {f["key"] for f in fields}
    # Sanity-check the new top-level schema surface.
    assert {
        "anthropic_upstreams",
        "openai_upstreams",
        "generic_openai_targets",
        "ollama_targets",
            "llama_cpp_defaults",
            "llama_cpp_targets",
            "comfyui_targets",
            "ollama_interfaces",
            "api_interfaces",
            "model_profiles",
        "admin_host",
        "admin_port",
        "dashboard_enabled",
        "dashboard_host",
        "dashboard_port",
        "dashboard_data_path",
        "dashboard_model_reclaim_enabled",
    } <= keys

    # All removed legacy keys must be absent.
    for legacy in (
        "host",
        "port",
        "upstreams",
        "internal_exposed_models",
        "external_exposed_models",
        "external_host",
        "external_port",
        "external_access_tokens",
    ):
        assert legacy not in keys, f"legacy field {legacy!r} should be gone"

    upstreams = next(f for f in fields if f["key"] == "anthropic_upstreams")
    assert upstreams["type"] == "object_list"
    upstream_item_keys = {f["key"] for f in upstreams["item_schema"]}
    assert {"name", "base_url", "auth_token", "models"} <= upstream_item_keys
    assert "expose_external" not in upstream_item_keys
    # models is now an object_list of ModelEntry(name, alias).
    upstream_models = next(f for f in upstreams["item_schema"] if f["key"] == "models")
    assert upstream_models["type"] == "object_list"
    model_entry_keys = {f["key"] for f in upstream_models["item_schema"]}
    assert {"name", "alias"} <= model_entry_keys

    openai_ups = next(f for f in fields if f["key"] == "openai_upstreams")
    assert openai_ups["type"] == "object_list"
    assert openai_ups["detect_models"] == "openai"
    openai_item_keys = {f["key"] for f in openai_ups["item_schema"]}
    assert {"name", "base_url", "auth_token", "models"} <= openai_item_keys

    generic_openai = next(f for f in fields if f["key"] == "generic_openai_targets")
    assert generic_openai["type"] == "object_list"
    assert generic_openai["detect_models"] == "openai"
    generic_openai_item_keys = {f["key"] for f in generic_openai["item_schema"]}
    assert {
        "name",
        "base_url",
        "auth_token",
        "models",
        "auto_start",
        "start_command",
        "stop_command",
        "idle_timeout_seconds",
        "startup_timeout_seconds",
        "health_path",
        "cwd",
        "max_concurrent_requests",
        "request_read_timeout_seconds",
    } <= generic_openai_item_keys

    ollama = next(f for f in fields if f["key"] == "ollama_targets")
    ollama_item_keys = {f["key"] for f in ollama["item_schema"]}
    assert "expose_external" not in ollama_item_keys
    assert {
        "name",
        "base_url",
        "models",
        "auto_start",
        "start_command",
        "idle_timeout_seconds",
        "health_path",
    } <= ollama_item_keys

    llama_cpp = next(f for f in fields if f["key"] == "llama_cpp_targets")
    assert llama_cpp["detect_models"] == "llama_cpp"
    llama_item_keys = {f["key"] for f in llama_cpp["item_schema"]}
    assert "expose_external" not in llama_item_keys
    # model_alias is replaced by upstream_id, with a new top-level alias field.
    assert "model_alias" not in llama_item_keys
    assert {
        "name",
        "base_url",
        "auth_token",
        "model",
        "alias",
        "upstream_id",
        "auto_start",
        "start_command",
        "idle_timeout_seconds",
    } <= llama_item_keys

    llama_defaults = next(f for f in fields if f["key"] == "llama_cpp_defaults")
    assert llama_defaults["type"] == "object"
    defaults_keys = {f["key"] for f in llama_defaults["item_schema"]}
    assert {
        "auto_start",
        "idle_timeout_seconds",
        "startup_timeout_seconds",
        "health_path",
        "cwd",
    } <= defaults_keys

    comfyui = next(f for f in fields if f["key"] == "comfyui_targets")
    comfyui_item_keys = {f["key"] for f in comfyui["item_schema"]}
    assert {
        "name",
        "base_url",
        "model",
        "auto_start",
        "start_command",
        "preset",
        "bindings",
        "static_inputs",
        "text_to_image_workflow_path",
        "image_to_image_workflow_path",
        "video_workflow_path",
        "image_to_video_workflow_path",
        "default_width",
        "default_height",
        "default_steps",
        "default_edit_denoise",
    } <= comfyui_item_keys
    comfyui_item_by_key = {f["key"]: f for f in comfyui["item_schema"]}
    assert comfyui_item_by_key["bindings"]["type"] == "json"
    assert comfyui_item_by_key["static_inputs"]["type"] == "json"

    # Interface arrays: each entry has its own host/port/tokens/exposed_models.
    ollama_iface = next(f for f in fields if f["key"] == "ollama_interfaces")
    assert ollama_iface["type"] == "object_list"
    ollama_iface_keys = {f["key"] for f in ollama_iface["item_schema"]}
    assert {"name", "host", "port", "access_tokens", "exposed_models"} <= ollama_iface_keys
    exposed = next(f for f in ollama_iface["item_schema"] if f["key"] == "exposed_models")
    assert exposed["type"] == "object_list"
    exposed_item_keys = {f["key"] for f in exposed["item_schema"]}
    assert {"model", "target", "alias"} <= exposed_item_keys

    api_iface = next(f for f in fields if f["key"] == "api_interfaces")
    assert api_iface["type"] == "object_list"
    api_iface_keys = {f["key"] for f in api_iface["item_schema"]}
    assert {"name", "host", "port", "access_tokens", "exposed_models"} <= api_iface_keys

    # Groups reorganized along source / interface lines.
    group_order = [g["key"] for g in schema["groups"]]
    assert group_order == [
        "model_sources_remote",
        "model_sources_generic_openai",
        "model_sources_ollama",
        "model_sources_llama_cpp",
        "model_sources_comfyui",
        "interface_ollama",
        "interface_api",
        "runtime",
        "dashboard",
        "admin",
    ]
    section_order: list[str] = []
    for group in schema["groups"]:
        if group["section"] not in section_order:
            section_order.append(group["section"])
    assert section_order == [
        "model_sources",
        "interfaces",
        "runtime",
        "dashboard",
        "admin",
    ]
    group_keys = set(group_order)
    for f in fields:
        assert f["group"] in group_keys


# ---------------------------------------------------------------------------
# /admin/config GET + PUT
# ---------------------------------------------------------------------------


def test_admin_get_config(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        resp = client.get("/admin/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["anthropic_upstreams"][0]["name"] == "default"
    assert body["_path"].endswith("config.json")


def test_admin_put_config_persists_and_reloads(admin_settings, tmp_path: Path):
    app = create_app(admin_settings)
    client = TestClient(app, base_url="http://testserver:21433")
    new_cfg = {
        "anthropic_upstreams": [
            {
                "name": "newup",
                "base_url": "http://other.test",
                "auth_token": "tok2",
                "models": [{"name": "another-model"}],
            }
        ],
        "ollama_targets": [
            {"name": "local", "base_url": "http://127.0.0.1:11434",
             "models": [{"name": "llama3.1"}]}
        ],
        "llama_cpp_defaults": {"auto_start": True, "health_path": "/health"},
        "llama_cpp_targets": [
            {"name": "qwen36", "base_url": "http://127.0.0.1:21436",
             "model": "qwen3.6", "auto_start": False}
        ],
        "generic_openai_targets": [
            {
                "name": "vllm",
                "base_url": "http://127.0.0.1:8062",
                "models": [{"name": "qwen-small"}],
                "auto_start": True,
                "start_command": "run-vllm",
            }
        ],
        "ollama_interfaces": [
            {
                "name": "ollama",
                "host": "127.0.0.1",
                "port": 21999,
                "access_tokens": [],
                "exposed_models": [
                    {"model": "another-model", "target": "newup"},
                    {"model": "llama3.1", "target": "local"},
                    {"model": "qwen3.6", "target": "qwen36"},
                    {"model": "qwen-small", "target": "vllm"},
                ],
            }
        ],
        "api_interfaces": [],
    }
    with client:
        resp = client.put("/admin/config", json=new_cfg)
        assert resp.status_code == 200, resp.text
        # In-memory settings reflect the change.
        assert app.state.settings.ollama_interfaces[0].port == 21999
        assert app.state.settings.anthropic_upstreams[0].name == "newup"
        assert "newup" in app.state.clients
        assert "local" in app.state.ollama_clients
        assert "qwen36" in app.state.llama_cpp_clients
        assert "vllm" in app.state.generic_openai_clients
        # File on disk reflects the change.
        cfg_path = Path(admin_settings.config_path)
        on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert on_disk["ollama_interfaces"][0]["port"] == 21999


def test_admin_put_invalid_returns_400(admin_settings):
    client = _admin_client(admin_settings)
    with client:
        # Duplicate source names across kinds is the main structural error.
        resp = client.put(
            "/admin/config",
            json={
                "anthropic_upstreams": [
                    {
                        "name": "dup",
                        "base_url": "http://a.test",
                        "auth_token": "t",
                        "models": [{"name": "m"}],
                    }
                ],
                "ollama_targets": [
                    {
                        "name": "dup",
                        "base_url": "http://127.0.0.1:11434",
                        "models": [{"name": "n"}],
                    }
                ],
            },
        )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------


def test_admin_hot_reload_reuses_unchanged_local_clients(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    settings = Settings(
        config_path=str(cfg_path),
        anthropic_upstreams=[
            {
                "name": "u",
                "base_url": "http://upstream.test",
                "auth_token": "tok",
                "models": [{"name": "remote"}],
            }
        ],
        ollama_targets=[
            {
                "name": "local",
                "base_url": "http://127.0.0.1:11434",
                "models": [{"name": "local-ollama"}],
            }
        ],
        llama_cpp_targets=[
            {
                "name": "qwen",
                "base_url": "http://127.0.0.1:21436",
                "model": "qwen",
                "start_command": "run-qwen",
            }
        ],
        generic_openai_targets=[
            {
                "name": "vllm",
                "base_url": "http://127.0.0.1:8062",
                "models": [{"name": "qwen-small"}],
                "start_command": "run-vllm",
            }
        ],
        ollama_interfaces=[
            {"name": "ollama", "port": 21434, "exposed_models": [
                {"model": "remote", "target": "u"},
                {"model": "local-ollama", "target": "local"},
                {"model": "qwen", "target": "qwen"},
                {"model": "qwen-small", "target": "vllm"},
            ]}
        ],
    )
    app = create_app(settings)
    client = TestClient(app, base_url="http://testserver:21433")

    with client:
        old_ollama = app.state.ollama_clients["local"]
        old_llama = app.state.llama_cpp_clients["qwen"]
        old_generic_openai = app.state.generic_openai_clients["vllm"]
        coord = app.state.vram_coordinator
        assert coord._participants[old_ollama.target_id] is old_ollama
        assert coord._participants[old_llama.target_id] is old_llama
        assert coord._participants[old_generic_openai.target_id] is old_generic_openai

        new_cfg = settings.model_dump()
        # Mutate one model on the unrelated remote upstream.
        new_cfg["anthropic_upstreams"][0]["models"].append({"name": "remote-2"})
        resp = client.put("/admin/config", json=new_cfg)

        assert resp.status_code == 200, resp.text
        assert app.state.ollama_clients["local"] is old_ollama
        assert app.state.llama_cpp_clients["qwen"] is old_llama
        assert app.state.generic_openai_clients["vllm"] is old_generic_openai
        assert coord._participants[old_ollama.target_id] is old_ollama
        assert coord._participants[old_llama.target_id] is old_llama
        assert coord._participants[old_generic_openai.target_id] is old_generic_openai


# ---------------------------------------------------------------------------
# Admin / dashboard listener gating
# ---------------------------------------------------------------------------


def test_admin_disabled_returns_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(
        json.dumps(
            {
                "admin_enabled": False,
                "anthropic_upstreams": [
                    {
                        "name": "u",
                        "base_url": "http://x.test",
                        "auth_token": "t",
                        "models": [{"name": "m"}],
                    }
                ],
                "ollama_interfaces": [
                    {"name": "ollama", "port": 21434, "exposed_models": [
                        {"model": "m", "target": "u"},
                    ]}
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
    with TestClient(app, base_url="http://testserver:21434") as ollama:
        assert ollama.get("/admin/").status_code == 404
        assert ollama.get("/api/version").status_code == 200
    with TestClient(app, base_url="http://testserver:21433") as admin:
        assert admin.get("/admin/").status_code == 200
        assert admin.get("/api/version").status_code == 404


def test_dashboard_routes_are_only_on_dashboard_port(admin_settings):
    app = create_app(admin_settings)
    with TestClient(app, base_url="http://testserver:21434") as ollama:
        assert ollama.get("/dashboard/").status_code == 404
        assert ollama.get("/api/version").status_code == 200
    with TestClient(app, base_url="http://testserver:21433") as admin:
        assert admin.get("/dashboard/").status_code == 404
    with TestClient(app, base_url="http://testserver:21432") as dashboard:
        assert dashboard.get("/dashboard/").status_code == 200
        assert dashboard.get("/", follow_redirects=False).status_code in (307, 308)
        assert dashboard.get("/api/version").status_code == 404


# ---------------------------------------------------------------------------
# /admin/probe-models
# ---------------------------------------------------------------------------


def _patch_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    real_client_cls = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs.pop("trust_env", None)
        return real_client_cls(**kwargs)

    monkeypatch.setattr("fake_ollama.admin.httpx.AsyncClient", fake_async_client)


def test_admin_probe_models_ollama(admin_settings, monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url).endswith("/api/tags")
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5-coder:7b"}]},
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
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
            json={"data": [
                {"id": "claude-3-5-sonnet-20241022"},
                {"id": "claude-3-5-haiku-20241022"},
            ]},
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
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
                "data": [{"id": "qwen3.6-27b-hauhau-q2kp", "object": "model"}],
            },
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
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
