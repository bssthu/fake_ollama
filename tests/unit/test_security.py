"""Offline policy and admission regressions; no live targets or attack payloads."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from fake_ollama.config import H3ContextIRProfile, Settings
from fake_ollama.request_data_log import (
    MAX_LOG_BODY_BYTES, RequestDataLogMiddleware, body_from_bytes,
    body_from_json, configure_request_data_logging,
)
from fake_ollama.security import (
    RequestBodyLimitMiddleware, management_session_token, require_management_access,
)
from fake_ollama.server import (
    _authorise_external_planner_url, _playground_interface_for,
    _select_context_ir_provider, create_app,
)


def _request(app, *, path="/admin/config", token="", origin=None, host="127.0.0.1", port=21433):
    headers = [(b"host", f"{host}:{port}".encode())]
    if token:
        headers.append((b"x-management-token", token.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    return Request({
        "type": "http", "method": "GET", "path": path, "scheme": "http",
        "headers": headers, "app": app, "server": ("127.0.0.1", port),
        "client": ("127.0.0.1", 50100), "query_string": b"",
    })


def test_local_management_page_bootstraps_a_separate_credential(settings):
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:21433", client=("127.0.0.1", 50100)) as client:
        page = client.get("/admin/")
        assert page.status_code == 200
        token = re.search(r"const managementToken = '([a-f0-9]+)'", page.text).group(1)
        assert page.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert client.get("/admin/config").status_code == 401
        assert client.get("/admin/config", headers={"X-Management-Token": token}).status_code == 200


@pytest.mark.parametrize("origin", [None, "http://127.0.0.1:21433"])
def test_management_accepts_local_authorized_requests(settings, origin):
    app = create_app(settings)
    require_management_access(_request(app, token=management_session_token(app), origin=origin))


@pytest.mark.parametrize("change", ["origin", "host", "port", "credential"])
def test_management_policy_rejects_a_mismatched_boundary(settings, change):
    app = create_app(settings)
    args = {"token": management_session_token(app)}
    args.update({
        "origin": {"origin": "http://docs.example"},
        "host": {"host": "workstation.example"},
        "port": {"port": 21435},
        "credential": {"token": "expired-test-token"},
    }[change])
    with pytest.raises(HTTPException) as error:
        require_management_access(_request(app, **args))
    assert error.value.status_code in (401, 403, 404)


def test_management_credential_rotation_invalidates_page_token(settings):
    app = create_app(settings)
    previous = management_session_token(app)
    app.state.settings = settings.model_copy(update={"management_access_tokens": ["new-management-key"]})
    with pytest.raises(HTTPException) as error:
        require_management_access(_request(app, token=previous))
    assert error.value.status_code == 401
    require_management_access(_request(app, token="new-management-key"))


def test_remote_management_uses_its_own_configured_credential(settings):
    settings = settings.model_copy(update={"admin_host": "0.0.0.0", "management_access_tokens": ["management-key"]})
    app = create_app(settings)
    with TestClient(app, base_url="http://workstation.example:21433") as client:
        assert client.get("/admin/").status_code == 401
        page = client.get("/admin/", auth=("admin", "management-key"))
        assert page.status_code == 200
        token = re.search(r"const managementToken = '([a-f0-9]+)'", page.text).group(1)
        assert client.get("/admin/schema", headers={"X-Management-Token": token}).status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_playground_anonymous_access_is_limited_to_local_binding(settings, host):
    settings = settings.model_copy(update={"playground_enabled": True, "playground_host": host})
    request = _request(create_app(settings), path="/playground/api/models", port=21431)
    if host == "127.0.0.1":
        assert _playground_interface_for(request, settings).name == "api"
    else:
        with pytest.raises(HTTPException) as error:
            _playground_interface_for(request, settings)
        assert error.value.status_code == 401


def test_playground_invalid_key_does_not_fall_back_to_open_interface(settings):
    settings = settings.model_copy(update={"playground_enabled": True})
    request = _request(create_app(settings), path="/playground/api/models", port=21431)
    request.scope["headers"].append((b"x-api-key", b"expired-test-key"))
    with pytest.raises(HTTPException) as error:
        _playground_interface_for(request, settings)
    assert error.value.status_code == 401


@pytest.mark.parametrize("path,port", [
    ("/api/tags", 21434), ("/api/show", 21434), ("/api/chat", 21434),
    ("/api/version", 21434), ("/api/ps", 21434),
    ("/v1/models", 21435), ("/v1/images/generations", 21435),
    ("/v1/videos/generations", 21435), ("/playground/api/models", 21431),
])
async def test_model_authentication_precedes_body_read_and_logging(settings, tmp_path, path, port):
    data = settings.model_dump()
    for iface in data["api_interfaces"] + data["ollama_interfaces"]:
        iface["access_tokens"] = ["model-key"]
    data["playground_enabled"] = True
    app = create_app(Settings.model_validate(data))
    request = _request(app, path=path, port=port)
    request.scope["method"] = "POST" if path.endswith(("show", "chat", "generations")) else "GET"
    request.scope["http_version"] = "1.1"
    sent = []

    async def receive():
        raise AssertionError("authentication must finish before reading the body")

    async def send(message):
        sent.append(message)

    log_file = tmp_path / "requests.jsonl"
    configure_request_data_logging(str(log_file))
    try:
        await app(request.scope, receive, send)
    finally:
        configure_request_data_logging(None)
    assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 401
    assert log_file.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("size,expected", [(8, 200), (9, 413)])
async def test_body_limit_applies_with_and_without_content_length(chunked, size, expected):
    app = FastAPI()
    app.state.settings = SimpleNamespace(max_request_body_bytes=8, request_body_timeout_seconds=1)
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.post("/upload")
    async def upload(request: Request):
        return {"bytes": len(await request.body())}

    async def chunks():
        yield b"a" * 4
        yield b"b" * (size - 4)

    content = chunks() if chunked else b"a" * size
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://unit.test") as client:
        response = await client.post("/upload", content=content)
    assert response.status_code == expected


def test_planner_permission_covers_default_named_and_model_selection(settings):
    profile = H3ContextIRProfile(name="plan", providers=[{
        "name": "main", "target": "default", "model": "llama-test",
    }])
    for selection in ("auto", "main", "model:llama-test@default"):
        assert _select_context_ir_provider(settings, profile, selection, has_images=False, interface_name="api").target == "default"
    data = settings.model_dump()
    data["api_interfaces"][0]["exposed_models"] = []
    restricted = Settings.model_validate(data)
    for selection in ("auto", "main", "model:llama-test@default"):
        with pytest.raises(HTTPException):
            _select_context_ir_provider(restricted, profile, selection, has_images=False, interface_name="api")


def test_external_planner_urls_require_explicit_exact_authorization():
    profile = H3ContextIRProfile(name="plan", providers=[{"name": "main", "target": "local", "model": "m"}], allow_external_api=True)
    with pytest.raises(HTTPException) as error:
        _authorise_external_planner_url(profile, "https://provider.example/api")
    assert error.value.status_code == 403
    profile.external_api_allowed_base_urls = ["https://provider.example/api", "http://127.0.0.1:8001"]
    assert _authorise_external_planner_url(profile, "https://provider.example/api/v1/models") == "https://provider.example/api"
    assert _authorise_external_planner_url(profile, "http://127.0.0.1:8001") == "http://127.0.0.1:8001"
    with pytest.raises(HTTPException):
        _authorise_external_planner_url(profile, "http://127.0.0.1:8002")


def test_show_obeys_interface_visibility(settings):
    data = settings.model_dump()
    data["ollama_interfaces"][0]["exposed_models"] = [data["ollama_interfaces"][0]["exposed_models"][0]]
    with TestClient(create_app(Settings.model_validate(data))) as client:
        assert client.post("/api/show", json={"model": "llama-test@default"}).status_code == 404


async def test_logger_streams_body_and_caps_preview(tmp_path, monkeypatch):
    import fake_ollama.request_data_log as data_log
    monkeypatch.setattr(data_log, "MAX_LOG_BODY_BYTES", 4)
    log_file = tmp_path / "preview.jsonl"
    configure_request_data_logging(str(log_file))
    messages = [
        {"type": "http.request", "body": b"abcd", "more_body": True},
        {"type": "http.request", "body": b"efgh", "more_body": False},
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        pass

    async def app(scope, read, write):
        assert len(messages) == 2  # middleware has not read ahead
        assert (await read())["body"] == b"abcd"
        assert len(messages) == 1
        assert (await read())["body"] == b"efgh"
        await write({"type": "http.response.start", "status": 200, "headers": []})
        await write({"type": "http.response.body", "body": b"abcdefgh"})

    try:
        await RequestDataLogMiddleware(app)({"type": "http", "path": "/api/chat", "headers": []}, receive, send)
    finally:
        configure_request_data_logging(None)
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    for event in ("http_request_body", "http_response_body"):
        body = next(r["body"] for r in records if r["event"] == event)
        assert body == {"bytes": 8, "encoding": "utf-8", "text": "abcd", "truncated": True}


def test_backend_log_helpers_also_bound_large_previews():
    assert body_from_bytes(b"a" * (MAX_LOG_BODY_BYTES + 1))["truncated"] is True
    result = body_from_json({"media": "a" * (MAX_LOG_BODY_BYTES + 1)})
    assert result["truncated"] is True
    assert len(result["text"].encode()) <= MAX_LOG_BODY_BYTES


async def test_slot_telemetry_keeps_numeric_fallback_fields_and_drops_invalid_values():
    from fake_ollama.llama_cpp_client import LlamaCppClient

    slots = [
        {"id": 0, "task_id": 7, "n_predict": 12, "state": 1, "prompt": {"n_past": 4}},
        {"id": "not reported", "n_ctx": "unknown", "params": {"n_ctx": 2048}},
        None,
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=slots))) as http:
        client = LlamaCppClient("http://fixture.test", client=http)
        result = await client.fetch_upstream_slots()
    assert result["error"] is None
    first, second = result["slots"]
    assert first["id"] == 0 and first["task_id"] == 7 and first["state"] == 1
    assert first["n_past"] == 4 and first["n_predict"] == 12
    assert "id_task" not in first and "n_decoded" not in first
    assert second["id"] is None and second["n_ctx"] == 2048


async def test_body_read_timeout_is_separate_from_inference_timeout():
    import asyncio
    app = FastAPI()
    app.state.settings = SimpleNamespace(max_request_body_bytes=8, request_body_timeout_seconds=0.01)
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.post("/upload")
    async def upload(request: Request):
        await request.body()
        await asyncio.sleep(0.02)  # receiving has finished, so this is allowed
        return {"ok": True}

    async def delayed_input():
        await asyncio.sleep(0.02)
        yield b"a"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://unit.test") as client:
        assert (await client.post("/upload", content=b"a")).status_code == 200
        assert (await client.post("/upload", content=delayed_input())).status_code == 408
