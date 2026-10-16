import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.config import Settings
from fake_ollama.ollama_client import OllamaClient
from fake_ollama.openai_client import OpenAIClient
from fake_ollama.server import create_app


DIRECTIONS = [("anthropic", "ollama"), ("openai", "ollama"),
              ("ollama", "anthropic"), ("anthropic", "openai")]


def wire_response(protocol, stream, text, failure=None):
    calls = [{"id": f"tool-{i}", "function": {"index": i, "name": "weather", "arguments": {"city": str(i)}}} for i in range(2)]
    blocks = ([{"type": "text", "text": "checking"}] if text else []) + [
        {"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": c["function"]["arguments"]} for c in calls]
    if protocol == "anthropic":
        if not stream:
            return httpx.Response(200, json={"content": blocks, "stop_reason": "tool_use"})
        events = [("message_start", {"message": {"id": "m"}})]
        for i, block in enumerate(blocks):
            initial = dict(block)
            if block["type"] == "tool_use":
                initial["input"] = {}
            events.append(("content_block_start", {"index": i, "content_block": initial}))
            if block["type"] == "tool_use":
                encoded = json.dumps(block["input"])
                for fragment in (encoded[:5], encoded[5:]):
                    events.append(("content_block_delta", {"index": i, "delta": {"type": "input_json_delta", "partial_json": fragment}}))
            events.append(("content_block_stop", {"index": i}))
        if failure == "event":
            events.append(("error", {"error": {"type": "overloaded_error", "message": "unavailable"}}))
        elif not failure:
            events.extend([("message_delta", {"delta": {"stop_reason": "tool_use"}}), ("message_stop", {})])
        return httpx.Response(200, text="".join(f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in events))
    if protocol == "openai":
        converted = [{"id": c["id"], "type": "function", "function": {"name": "weather", "arguments": json.dumps(c["function"]["arguments"])}} for c in calls]
        if not stream:
            return httpx.Response(200, json={"choices": [{"message": {"content": "checking" if text else "", "tool_calls": converted}, "finish_reason": "tool_calls"}]})
        chunks = [{"choices": [{"delta": {"content": "checking" if text else ""}}]}]
        for i, c in enumerate(converted):
            args = c["function"]["arguments"]
            chunks += [{"choices": [{"delta": {"tool_calls": [{"index": i, "id": c["id"], "function": {"name": "weather", "arguments": args[:5]}}]}}]},
                       {"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"arguments": args[5:]}}]}}]}]
        if failure == "event":
            chunks.append({"error": {"message": "unavailable"}})
        elif not failure:
            chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + ("data: [DONE]\n\n" if not failure else ""))
    if not stream:
        return httpx.Response(200, json={"message": {"content": "checking" if text else "", "tool_calls": calls}, "done": True, "done_reason": "stop"})
    chunks = [{"message": {"content": "checking" if text else ""}}, {"message": {"tool_calls": calls}}]
    if failure == "event":
        chunks.append({"error": "unavailable"})
    elif not failure:
        chunks.append({"message": {"content": ""}, "done": True, "done_reason": "stop"})
    return httpx.Response(200, text="".join(json.dumps(c) + "\n" for c in chunks))


def make_client(upstream, frontend, *, text=True, failure=None):
    source_key = {"anthropic": "anthropic_upstreams", "openai": "openai_upstreams", "ollama": "ollama_targets"}[upstream]
    settings = Settings(**{source_key: [{"name": "up", "base_url": "http://up.test", "models": [{"name": "m"}]}],
        "ollama_interfaces": [{"name": "ollama", "port": 21434, "exposed_models": [{"model": "m", "target": "up"}]}],
        "api_interfaces": [{"name": "api", "port": 21435, "exposed_models": [{"model": "m", "target": "up"}]}]})
    seen = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={})
        payload = json.loads(request.content)
        seen.append(payload)
        if failure == "exception":
            raise RuntimeError("private diagnostic must not escape")
        return wire_response(upstream, payload.get("stream"), text, failure)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(settings)
    if upstream == "anthropic":
        app.state.clients = {"up": AnthropicClient("http://up.test", "synthetic", client=transport)}
    elif upstream == "openai":
        app.state.openai_clients = {"up": OpenAIClient("http://up.test", client=transport)}
    else:
        app.state.ollama_clients = {"up": OllamaClient("http://up.test", client=transport)}
    port = 21434 if frontend == "ollama" else 21435
    path = {"ollama": "/api/chat", "openai": "/v1/chat/completions", "anthropic": "/v1/messages"}[frontend]
    return TestClient(app, base_url=f"http://testserver:{port}"), path, seen


def frames(response, frontend):
    if frontend == "ollama":
        return [json.loads(line) for line in response.text.splitlines()]
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:") and "[DONE]" not in line]


@pytest.mark.parametrize("upstream,frontend", DIRECTIONS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_text", [False, True])
def test_endpoint_tool_semantics_and_next_turn(upstream, frontend, stream, with_text):
    client, path, seen = make_client(upstream, frontend, text=with_text)
    history = [{"role": "user", "content": "weather"}]
    with client:
        response = client.post(path, json={"model": "m@up", "messages": history, "stream": stream, "max_tokens": 100})
        assert response.status_code == 200
        if frontend == "ollama":
            chunks = frames(response, frontend) if stream else [response.json()]
            calls = [tc for chunk in chunks for tc in chunk.get("message", {}).get("tool_calls", [])]
            assert calls == [{"id": f"tool-{i}", "function": {"index": i, "name": "weather", "arguments": {"city": str(i)}}} for i in range(2)]
            assert chunks[-1]["done_reason"] == "tool_calls"
            assert "".join(c["message"].get("content", "") for c in chunks) == ("checking" if with_text else "")
            history += [{"role": "assistant", "content": "checking" if with_text else "", "tool_calls": calls}]
            history += [{"role": "tool", "tool_name": "weather", "content": str(i)} for i in range(2)]
            result = client.post(path, json={"model": "m@up", "messages": history, "stream": False})
            assert result.status_code == 200
            if upstream == "anthropic":
                results = seen[-1]["messages"][-1]["content"]
                assert [b["tool_use_id"] for b in results] == ["tool-0", "tool-1"]
            else:
                assert [m["tool_call_id"] for m in seen[-1]["messages"] if m["role"] == "tool"] == ["tool-0", "tool-1"]
        elif frontend == "anthropic":
            chunks = frames(response, frontend) if stream else []
            if stream:
                starts = [c for c in chunks if c["type"] == "content_block_start"]
                assert [c["index"] for c in starts] == list(range(len(starts)))
                tools = [c["content_block"] for c in starts if c["content_block"]["type"] == "tool_use"]
                args = [json.loads(c["delta"]["partial_json"]) for c in chunks if c.get("delta", {}).get("type") == "input_json_delta"]
                assert chunks[-2]["delta"]["stop_reason"] == "tool_use"
            else:
                tools = [c for c in response.json()["content"] if c["type"] == "tool_use"]
                args = [c["input"] for c in tools]
                assert response.json()["stop_reason"] == "tool_use"
            assert [(t["id"], t["name"]) for t in tools] == [("tool-0", "weather"), ("tool-1", "weather")]
            assert args == [{"city": "0"}, {"city": "1"}]
        else:
            if stream:
                chunks = frames(response, frontend)
                calls = {}
                for chunk in chunks:
                    for tc in chunk["choices"][0]["delta"].get("tool_calls", []):
                        acc = calls.setdefault(tc["index"], {"arguments": ""})
                        if tc.get("id"):
                            acc.update(id=tc["id"], name=tc["function"]["name"])
                        acc["arguments"] += tc["function"].get("arguments", "")
                assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
                assert list(calls) == [0, 1]
                values = list(calls.values())
            else:
                body = response.json()["choices"][0]
                assert body["finish_reason"] == "tool_calls"
                values = [{"id": c["id"], **c["function"]} for c in body["message"]["tool_calls"]]
            assert [(v["id"], v["name"], json.loads(v["arguments"])) for v in values] == [(f"tool-{i}", "weather", {"city": str(i)}) for i in range(2)]


@pytest.mark.parametrize("upstream,frontend", DIRECTIONS)
@pytest.mark.parametrize("failure", ["event", "eof", "exception"])
def test_endpoint_stream_failure_never_synthesizes_success(upstream, frontend, failure):
    client, path, _ = make_client(upstream, frontend, failure=failure)
    with client:
        response = client.post(path, json={"model": "m@up", "messages": [{"role": "user", "content": "test"}], "max_tokens": 100, "stream": True})
    result = frames(response, frontend)
    assert result[-1].get("error")
    assert "private diagnostic" not in response.text
    assert not any(c.get("done_reason") or c.get("type") == "message_stop" or any(ch.get("finish_reason") for ch in c.get("choices", [])) for c in result)
