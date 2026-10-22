"""Exercise credential/model binding through HTTP and actual mock wire bodies."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.server import create_app


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("protocol", ["openai", "ollama"])
def test_http_thinking_roundtrip_is_bound_to_history_identity(settings, protocol, stream):
    seen = []

    def upstream(request):
        seen.append(json.loads(request.content))
        if not seen[-1]["stream"]:
            return httpx.Response(200, json={"content": [
                {"type": "thinking", "thinking": "private", "signature": "signature"},
                {"type": "text", "text": "好的"},
            ], "stop_reason": "end_turn"})
        events = [
            ("message_start", {"message": {"id": "m"}}),
            ("content_block_start", {"index": 0, "content_block": {"type": "thinking"}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "private"}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "signature_delta", "signature": "signature"}}),
            ("content_block_stop", {"index": 0}),
            ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "好的"}}),
            ("content_block_stop", {"index": 1}),
            ("message_delta", {"delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {}),
        ]
        return httpx.Response(200, text="".join(f"event: {kind}\ndata: {json.dumps(data)}\n\n" for kind, data in events))

    settings.api_interfaces[0].access_tokens = ["credential-A", "credential-B"]
    settings.ollama_interfaces[0].access_tokens = ["credential-A", "credential-B"]
    app = create_app(settings)
    app.state.clients = {"default": AnthropicClient("http://upstream.test", "test-token",
                        client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))}
    port, path = (21435, "/v1/chat/completions") if protocol == "openai" else (21434, "/api/chat")
    history = [{"role": "user", "content": "session-A"}]
    model = "llama-test@default"
    with TestClient(app, base_url=f"http://testserver:{port}") as client:
        def post(messages, token="credential-A", selected_model=model, streaming=False):
            result = client.post(path, headers={"x-api-key": token}, json={
                "model": selected_model, "messages": messages, "stream": streaming,
            })
            assert result.status_code == 200
            return result
        response = post(history, streaming=stream)
        assert "signature" not in response.text
        roundtrip = history + [{"role": "assistant", "content": "好的"}, {"role": "user", "content": "continue"}]
        post(roundtrip)
        assert seen[-1]["messages"][1]["content"][0] == {
            "type": "thinking", "thinking": "private", "signature": "signature",
        }
        for token, selected_model, previous in [
            ("credential-B", model, "session-A"),
            ("credential-A", "claude-3-5-sonnet-20241022@default", "session-A"),
            ("credential-A", model, "session-B"),
        ]:
            post([{"role": "user", "content": previous}] + roundtrip[1:], token, selected_model)
            assert seen[-1]["messages"][1]["content"] == "好的"
        # Simulate changed upstream credentials/config without touching real files.
        updated = settings.model_copy(deep=True)
        updated.anthropic_upstreams[0].auth_token = "new-synthetic-upstream"
        app.state.settings = updated
        post(roundtrip)
        assert seen[-1]["messages"][1]["content"] == "好的"


def test_missing_thinking_policy_rejects_before_upstream(settings):
    settings.anthropic_upstreams[0].missing_thinking_policy = "error"
    app = create_app(settings)
    app.state.clients = {"default": AnthropicClient("http://upstream.test", "test-token",
                        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: pytest.fail("Must not contact upstream"))))}
    with TestClient(app, base_url="http://127.0.0.1:21435") as client:
        response = client.post("/v1/chat/completions", json={
            "model": "llama-test@default", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer", "reasoning_content": "unsigned"},
                {"role": "user", "content": "continue"},
            ], "stream": False,
        })
    assert response.status_code == 400
    assert "recoverable thinking" in response.json()["detail"]
    assert "unsigned" not in response.text
