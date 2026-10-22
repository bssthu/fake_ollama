import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.openai_client import OpenAIClient
from fake_ollama.ollama_client import OllamaClient
from fake_ollama.config import Settings
from fake_ollama.server import create_app


def harness(upstream, frontend, counts):
    sources = {"anthropic": "anthropic_upstreams", "openai": "openai_upstreams", "ollama": "ollama_targets"}
    settings = Settings(**{sources[upstream]: [{"name": "up", "base_url": "http://up.test", "models": [{"name": "m"}]}],
        "ollama_interfaces": [{"name": "ollama", "port": 21434, "exposed_models": [{"model": "m", "target": "up"}]}],
        "api_interfaces": [{"name": "api", "port": 21435, "exposed_models": [{"model": "m", "target": "up"}]}]})
    seen = []
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        seen.append(body)
        usage = {key: value for key, value in zip(("input_tokens", "output_tokens"), counts) if value is not None}
        if upstream == "anthropic":
            if not body.get("stream"):
                return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": usage})
            events = [("message_start", {"message": {"id": "msg", "usage": {key: value for key, value in usage.items() if key == "input_tokens"}}}),
                      ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": "ok"}}),
                      ("content_block_stop", {"index": 0}),
                      ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": usage}), ("message_stop", {})]
            return httpx.Response(200, text="".join(f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in events))
        if upstream == "openai":
            usage = {new: usage[old] for old, new in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")) if old in usage}
            if not body.get("stream"):
                return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": usage})
            chunks = [{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}, {"choices": [], "usage": usage}]
            return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")
        usage = {new: usage[old] for old, new in (("input_tokens", "prompt_eval_count"), ("output_tokens", "eval_count")) if old in usage}
        response = {"message": {"content": "ok"}, "done": True, "done_reason": "stop", **usage}
        return httpx.Response(200, text=json.dumps(response) + "\n") if body.get("stream") else httpx.Response(200, json=response)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    app = create_app(settings)
    if upstream == "anthropic":
        app.state.clients = {"up": AnthropicClient("http://up.test", "synthetic", client=transport)}
    elif upstream == "openai":
        app.state.openai_clients = {"up": OpenAIClient("http://up.test", client=transport)}
    else:
        app.state.ollama_clients = {"up": OllamaClient("http://up.test", client=transport)}
    port = 21434 if frontend == "ollama" else 21435
    path = {"openai": "/v1/chat/completions", "anthropic": "/v1/messages", "ollama": "/api/chat"}[frontend]
    return TestClient(app, base_url=f"http://127.0.0.1:{port}"), path, seen


def frames(response):
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:") and "[DONE]" not in line]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["none", "auto", "required", "weather"])
@pytest.mark.parametrize("direction", ["openai_to_anthropic", "anthropic_to_openai"])
def test_tool_controls_reach_actual_mock_http_request(direction, choice, stream):
    frontend, upstream = direction.split("_to_")
    client, path, seen = harness(upstream, frontend, (7, 3))
    if frontend == "openai":
        tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
        control = {"type": "function", "function": {"name": "weather"}} if choice == "weather" else choice
        fields = {"tool_choice": control, "parallel_tool_calls": False}
        expected = {"type": {"required": "any", "weather": "tool"}.get(choice, choice)}
        if choice == "weather":
            expected["name"] = "weather"
        if choice != "none":
            expected["disable_parallel_tool_use"] = True
    else:
        tools = [{"name": "weather", "input_schema": {"type": "object"}}]
        control = {"type": {"required": "any", "weather": "tool"}.get(choice, choice)}
        if choice == "weather":
            control["name"] = "weather"
        if choice != "none":
            control["disable_parallel_tool_use"] = True
        fields = {"tool_choice": control}
        expected = {"type": "function", "function": {"name": "weather"}} if choice == "weather" else choice
    with client:
        response = client.post(path, json={"model": "m@up", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100,
                                          "stream": stream, "tools": tools, **fields})
    assert response.status_code == 200 and "ok" in response.text
    assert seen[0]["tool_choice"] == expected
    if upstream == "openai" and choice != "none":
        assert seen[0]["parallel_tool_calls"] is False


@pytest.mark.parametrize("upstream,frontend", [
    ("anthropic", "openai"), ("openai", "anthropic"), ("openai", "ollama"),
    ("anthropic", "ollama"), ("ollama", "anthropic"), ("ollama", "openai"),
])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("counts", [(None, None), (0, None), (7, 3)])
def test_usage_agrees_between_stream_and_nonstream(upstream, frontend, stream, counts):
    client, path, seen = harness(upstream, frontend, counts)
    body = {"model": "m@up", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100, "stream": stream}
    if frontend == "openai" and stream:
        body["stream_options"] = {"include_usage": True}
    with client:
        response = client.post(path, json=body)
    assert response.status_code == 200 and "ok" in response.text
    if upstream == "openai" and stream:
        assert seen[0]["stream_options"] == {"include_usage": True}
    if frontend == "openai":
        chunks = frames(response) if stream else []
        usage = (chunks[-1]["usage"] if stream else response.json()["usage"]) or {}
        if stream:
            assert chunks[-1]["choices"] == []
            assert all(chunk["usage"] is None for chunk in chunks[:-1])
        keys = ("prompt_tokens", "completion_tokens")
    elif frontend == "anthropic":
        usage = frames(response)[-2]["usage"] if stream else response.json()["usage"]
        keys = ("input_tokens", "output_tokens")
    else:
        usage = json.loads(response.text.splitlines()[-1]) if stream else response.json()
        keys = ("prompt_eval_count", "eval_count")
    for key, expected in zip(keys, counts):
        if expected is None:
            assert key not in usage
        else:
            assert usage[key] == expected
    if frontend == "openai":
        assert usage.get("total_tokens") == (sum(counts) if None not in counts else None)


@pytest.mark.parametrize("fields", [{"tool_choice": "required"}, {"tool_choice": "auto", "parallel_tool_calls": False}])
def test_native_ollama_rejects_constraints_it_cannot_express(fields):
    client, path, seen = harness("ollama", "openai", (7, 3))
    with client:
        response = client.post(path, json={"model": "m@up", "messages": [], "tools": [
            {"type": "function", "function": {"name": "weather"}}], **fields})
    assert response.status_code == 400 and not seen


def test_count_tokens_does_not_claim_zero_when_upstream_count_is_missing():
    client, _, _ = harness("anthropic", "anthropic", (None, None))
    with client:
        response = client.post("/v1/messages/count_tokens", json={"model": "m@up", "messages": []})
    assert response.status_code == 502
    assert "error" in response.json()
