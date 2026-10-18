"""Real ASGI routing with offline native decision HTTP transports."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.decision_client import DecisionClient
from fake_ollama.server import create_app


def harness(*, dialect="systemone", token="", port=21435, responder=None, capabilities=None):
    settings = Settings(decision_targets=[{
        "name": "decision", "base_url": "http://decision.test", "dialect": dialect,
        "models": [{"name": "wire-model", "alias": "friendly"}],
        "decision_capabilities": capabilities or {"question_types": ["predicate", "choice", "score"], "modalities": ["text"]},
    }], ollama_interfaces=[{"name": "ollama", "port": 21434,
        "exposed_models": [{"model": "friendly", "target": "decision", "alias": "judge"}]}],
        api_interfaces=[{"name": "api", "port": 21435, "access_tokens": [token] if token else [],
        "exposed_models": [{"model": "friendly", "target": "decision", "alias": "judge"}]}],
        admin_enabled=True, admin_port=21436, dashboard_enabled=True, dashboard_port=21437,
        playground_enabled=True, playground_port=21438)
    seen = []
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"info": {"version": "test"}, "paths": {"/v1/systemone": {"post": {}}, "/v1/decisions": {"post": {}}}})
        body = json.loads(request.content)
        seen.append((request, body))
        if responder:
            return responder(request, body)
        if dialect == "systemone":
            return httpx.Response(200, json={"model": "wire-model", "answers": {
                identifier: {"type": "noul", "noul": .8} for identifier in body["questions"]}, "usage": {"input_tokens": 8}})
        return httpx.Response(200, json={"model": "wire-model", "answers": [
            {"type": "predicate", "name": q.get("name"), "probability": .8} for q in body["questions"]],
            "usage": {"input_tokens": 8, "output_tokens": 0, "total_tokens": 8,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}})
    app = create_app(settings)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    app.state.decision_clients = {"decision": DecisionClient(settings.decision_targets[0], client=transport)}
    return TestClient(app, base_url=f"http://127.0.0.1:{port}"), seen, app


def body(dialect="systemone", model="judge"):
    if dialect == "systemone":
        return {"model": model, "state": {"ticket": "broken"}, "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}}}
    return {"model": model, "input": "broken", "questions": [{"type": "predicate", "name": "urgent", "instructions": "Urgent?"}]}


@pytest.mark.parametrize("frontend", ["systemone", "openai_decisions"])
@pytest.mark.parametrize("upstream", ["systemone", "openai_decisions"])
def test_two_public_dialects_dispatch_to_two_native_dialects(frontend, upstream):
    client, seen, _ = harness(dialect=upstream)
    path = "/v1/systemone" if frontend == "systemone" else "/v1/decisions"
    response = client.post(path, json=body(frontend))
    assert response.status_code == 200, response.text
    public = response.json()
    assert public["model"] == "judge"
    assert isinstance(public["answers"], dict if frontend == "systemone" else list)
    assert public["usage"]["input_tokens"] == 8
    assert public["decision_metadata"]["usage_complete"] is (upstream == "openai_decisions")
    assert public["decision_metadata"]["source"].startswith("native_")
    request, wire = seen[0]
    assert request.url.path == ("/v1/systemone" if upstream == "systemone" else "/v1/decisions")
    assert wire["model"] == "wire-model"
    assert ("state" in wire) == (upstream == "systemone")
    assert isinstance(wire["questions"], dict if upstream == "systemone" else list)


@pytest.mark.parametrize("path,payload", [("/v1/systemone", body("openai_decisions")), ("/v1/decisions", body())])
def test_dialects_are_not_aliases(path, payload):
    client, seen, _ = harness()
    response = client.post(path, json=payload)
    assert response.status_code == 400
    assert not seen


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions"])
@pytest.mark.parametrize("port", [21434, 21436, 21437, 29999])
def test_decisions_fail_closed_on_ollama_management_and_unknown_listeners(path, port):
    client, seen, _ = harness(port=port)
    assert client.post(path, json=body()).status_code == 404
    assert not seen


def test_auth_and_exposure_precede_upstream_calls():
    client, seen, _ = harness(token="private")
    assert client.post("/v1/systemone", content="not json").status_code == 401
    assert client.post("/v1/systemone", json=body()).status_code == 401
    assert client.post("/v1/systemone", json=body(model="wire-model"), headers={"x-api-key": "private"}).status_code == 404
    assert not seen
    response = client.post("/v1/systemone", json=body(), headers={"Authorization": "Bearer private"})
    assert response.status_code == 200
    assert seen


def test_playground_uses_existing_interface_token():
    client, seen, _ = harness(port=21438, token="private")
    assert client.post("/v1/systemone", json=body()).status_code == 401
    response = client.post("/v1/systemone", json=body(), headers={"x-api-key": "private"})
    assert response.status_code == 200, response.text
    assert seen


def test_retired_listener_is_not_reassigned():
    client, seen, app = harness()
    changed = app.state.settings.model_copy(deep=True)
    changed.api_interfaces[0].port = 21535
    app.state.settings = changed
    assert client.post("/v1/systemone", json=body()).status_code == 404
    assert not seen


@pytest.mark.parametrize("payload", ["[]", "null", '{"model":"judge","model":"other"}', '{"model":"judge","state":NaN,"questions":{}}'])
def test_malformed_bodies_fail_without_inference(payload):
    client, seen, _ = harness()
    assert client.post("/v1/systemone", content=payload).status_code == 400
    assert not seen


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"])
def test_decision_model_cannot_fall_through_to_chat_or_fake_token_estimates(path):
    client, seen, _ = harness()
    response = client.post(path, json={"model": "judge", "messages": []})
    assert response.status_code == 400
    assert "decision" in response.text.lower()
    assert not seen


def test_upstream_refusal_remains_explicit():
    client, _, _ = harness(dialect="openai_decisions", responder=lambda r, p: httpx.Response(200,
        json={"answers": [{"name": "urgent", "type": "refusal"}]}))
    response = client.post("/v1/systemone", json=body())
    assert response.status_code == 400
    assert "refusal cannot be represented" in response.text
    response = client.post("/v1/decisions", json=body("openai_decisions"))
    assert response.status_code == 200
    assert response.json()["answers"] == [{"name": "urgent", "type": "refusal"}]
    assert "usage" not in response.json()


def test_upstream_malformed_or_http_error_is_not_success():
    client, _, _ = harness(responder=lambda r, p: httpx.Response(200, json={"answers": []}))
    assert client.post("/v1/systemone", json=body()).status_code == 502
    client, _, _ = harness(responder=lambda r, p: httpx.Response(429, json={"error": "slow down"}))
    assert client.post("/v1/systemone", json=body()).status_code == 429


def test_images_require_declared_modality_and_native_opt_in():
    client, seen, _ = harness()
    payload = body()
    payload["state"] = "ref"
    payload["assets"] = {"ref": {"format": "png", "data_base64": "AA=="}}
    assert client.post("/v1/systemone", json=payload).status_code == 400
    response = client.post("/v1/systemone", json=payload, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert response.status_code == 400
    assert not seen


@pytest.mark.parametrize("payload", [b"\xef\xbb\xbf{}", "{}".encode("utf-16"), b"\xff"])
def test_decision_json_requires_utf8_without_bom(payload):
    client, seen, _ = harness()
    assert client.post("/v1/systemone", content=payload).status_code == 400
    assert not seen


@pytest.mark.parametrize("dialect", ["systemone", "openai_decisions"])
@pytest.mark.parametrize("kind", [[], {}, 1, None])
def test_invalid_question_discriminators_return_client_errors(dialect, kind):
    client, seen, _ = harness()
    payload = body(dialect)
    question = payload["questions"]["urgent"] if dialect == "systemone" else payload["questions"][0]
    question["type"] = kind
    path = "/v1/systemone" if dialect == "systemone" else "/v1/decisions"
    assert client.post(path, json=payload).status_code == 400
    assert not seen


@pytest.mark.parametrize("frontend", ["systemone", "openai_decisions"])
@pytest.mark.parametrize("missing", ["confidence", "probabilities"])
def test_incomplete_native_metrics_fail_instead_of_fabricating(frontend, missing):
    def respond(request, payload):
        answer = {"type": "choice", "choice": "a", "confidence": .5, "probabilities": {"a": .75, "b": .25}}
        answer.pop(missing)
        return httpx.Response(200, json={"answers": {next(iter(payload["questions"])): answer}})
    client, _, _ = harness(responder=respond)
    if frontend == "systemone":
        payload = {"model": "judge", "state": "s", "questions": {"q": {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None}}}}
        path = "/v1/systemone"
    else:
        payload = {"model": "judge", "input": "s", "questions": [{"type": "choice", "instructions": "q", "choices": [{"value": "a"}, {"value": "b"}]}]}
        path = "/v1/decisions"
    response = client.post(path, json=payload)
    assert response.status_code == 502
    assert "cannot be synthesized" in response.text


@pytest.mark.parametrize("reload", [False, True])
async def test_http_disconnect_drains_submitted_inference_before_reload_or_release(reload):
    import asyncio

    _, _, app = harness()
    old_client = app.state.decision_clients["decision"]
    await old_client.aclose()
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    async def upstream(request):
        if request.method == "GET":
            return httpx.Response(404)
        entered.set()
        try:
            await release.wait()
            return httpx.Response(200, json={"answers": {"urgent": {"type": "noul", "noul": .8}}})
        finally:
            completed.set()
    transport = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = DecisionClient(app.state.settings.decision_targets[0], client=transport)
    app.state.decision_clients = {"decision": client}
    incoming = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": json.dumps(body()).encode(), "more_body": False})
    sent = []
    async def send(message):
        sent.append(message)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST", "http_version": "1.1", "scheme": "http",
        "path": "/v1/systemone", "raw_path": b"/v1/systemone", "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:21435"), (b"content-type", b"application/json")],
        "server": ("127.0.0.1", 21435), "client": ("127.0.0.1", 12345),
    }
    task = asyncio.create_task(app(scope, incoming.get, send))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert client.active_requests == 1
        await incoming.put({"type": "http.disconnect"})
        for _ in range(10):
            await asyncio.sleep(0)
        assert not completed.is_set()
        assert client._active == client._request_refs == 1
        assert app.state.runtime_resources.references
        if reload:
            from fake_ollama.admin import _swap_settings
            from fastapi import HTTPException
            settings = app.state.settings
            changed = settings.decision_targets[0].model_copy(update={"probe_interval_seconds": 123})
            await _swap_settings(app, settings.model_copy(update={"decision_targets": [changed]}))
            replacement = app.state.decision_clients["decision"]
            with pytest.raises(HTTPException, match="draining"):
                app.state.runtime_resources.check(replacement)
        release.set()
        await asyncio.wait_for(task, 2)
        assert completed.is_set()
        assert client._active == client._request_refs == client._queued == 0
        assert not app.state.runtime_resources.references
        assert any(message.get("status") == 499 for message in sent)
        if reload:
            assert app.state.runtime_resources.check(replacement) is replacement
            assert client._closed
            await replacement.aclose()
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.aclose()
        await transport.aclose()
