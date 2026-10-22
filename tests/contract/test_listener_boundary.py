"""Unknown and retired sockets must never borrow another listener's auth."""

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from fake_ollama.server import _interface_for, create_app


@pytest.mark.parametrize("port", [80, 29999])
@pytest.mark.parametrize("path", ["/api/tags", "/v1/models", "/v1/chat/completions"])
def test_unknown_listener_is_denied(settings, port, path):
    app = create_app(settings)
    with TestClient(app, base_url=f"http://127.0.0.1:{port}") as client:
        assert client.get(path).status_code == 404


@pytest.mark.parametrize("kind,path", [
    ("ollama_interfaces", "/api/tags"),
    ("api_interfaces", "/v1/models"),
])
def test_retired_listener_is_denied_after_reload(settings, kind, path):
    app = create_app(settings)
    port = getattr(settings, kind)[0].port
    with TestClient(app, base_url=f"http://127.0.0.1:{port}") as client:
        assert client.get(path).status_code == 200
        updated = settings.model_copy(deep=True)
        getattr(updated, kind)[0].port += 1000
        app.state.settings = updated
        assert client.get(path).status_code == 404


def test_interface_lookup_does_not_guess(settings):
    app = create_app(settings)
    request = Request({"type": "http", "app": app, "server": ("0.0.0.0", 29999), "path": "/api/tags"})
    assert _interface_for(request) is None


def test_explicit_test_fallback_uses_matching_surface_auth(settings):
    settings.api_interfaces[0].access_tokens = ["synthetic-api-token"]
    app = create_app(settings, test_listener_fallback=True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/tags").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"x-api-key": "synthetic-api-token"}).status_code == 200
