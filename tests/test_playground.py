"""Tests for the lightweight model playground listener."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.server import create_app


def _settings() -> Settings:
    return Settings(
        anthropic_upstreams=[
            {
                "name": "remote",
                "base_url": "http://upstream.test",
                "auth_token": "upstream-token",
                "models": [{"name": "model-a"}, {"name": "model-b"}],
            }
        ],
        ollama_interfaces=[],
        api_interfaces=[
            {
                "name": "first",
                "host": "127.0.0.1",
                "port": 21435,
                "access_tokens": ["key-a"],
                "exposed_models": [
                    {"model": "model-a", "target": "remote", "alias": "alpha"},
                ],
            },
            {
                "name": "second",
                "host": "127.0.0.1",
                "port": 21436,
                "access_tokens": ["key-b"],
                "exposed_models": [
                    {"model": "model-b", "target": "remote", "alias": "beta"},
                ],
            },
        ],
        playground_enabled=True,
        playground_host="127.0.0.1",
        playground_port=21431,
    )


def test_playground_static_page_and_security_headers():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21431") as client:
        response = client.get("/playground/")
        css = client.get("/playground/playground.css")
        js = client.get("/playground/playground.js")

    assert response.status_code == 200
    assert "Model Playground" in response.text
    assert "轻量、多能力、即时调试" in response.text
    assert 'id="apiKey"' in response.text
    assert 'id="model"' in response.text
    assert 'id="fileInput"' in response.text
    assert 'id="operation"' in response.text
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200
    assert "clipboardData" in js.text
    assert "image_generation" in js.text
    assert "video_generation" in js.text
    assert "new FormData()" in js.text
    assert "stream: true" in js.text
    assert "estimated_vram_gb" in js.text
    assert "showRequestError" in js.text
    assert response.headers["cache-control"] == "no-store"
    assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert "'unsafe-inline'" not in response.headers["content-security-policy"]


def test_playground_port_only_exposes_playground_and_model_surfaces():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21431") as client:
        root = client.get("/", follow_redirects=False)
        admin = client.get("/admin/")
        dashboard = client.get("/dashboard/")
        version = client.get("/api/version")

    assert root.status_code in (307, 308)
    assert root.headers["location"] == "/playground/"
    assert admin.status_code == 404
    assert dashboard.status_code == 404
    assert version.status_code == 404


def test_playground_api_key_selects_the_matching_interface_models():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21431") as client:
        missing = client.get("/v1/models")
        first = client.get("/v1/models", headers={"Authorization": "Bearer key-a"})
        second = client.get("/v1/models", headers={"x-api-key": "key-b"})

    assert missing.status_code == 401
    assert [item["id"] for item in first.json()["data"]] == ["alpha"]
    assert [item["id"] for item in second.json()["data"]] == ["beta"]
    alpha = first.json()["data"][0]
    assert alpha["capabilities"] == ["completion", "tools", "vision"]
    assert alpha["operations"] == [
        {
            "id": "chat",
            "endpoint": "/v1/chat/completions",
            "stream": True,
            "accepts_images": True,
            "tool_calling": True,
        }
    ]


def test_playground_route_is_not_available_on_other_ports():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21435") as api_client:
        assert api_client.get("/playground/").status_code == 404


def test_disabled_playground_does_not_register_static_route():
    settings = _settings().model_copy(update={"playground_enabled": False})
    app = create_app(settings)
    with TestClient(app, base_url="http://testserver:21431") as client:
        assert client.get("/playground/").status_code == 404
