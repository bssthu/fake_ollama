"""End-to-end tests for the FastAPI server using a mocked upstream."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.server import create_app


def _build_sse(events: List[Dict[str, Any]]) -> bytes:
    chunks: List[str] = []
    for ev in events:
        chunks.append(f"event: {ev['type']}\n")
        chunks.append(f"data: {json.dumps(ev['data'])}\n\n")
    return "".join(chunks).encode("utf-8")


def _make_client(settings, transport: httpx.MockTransport) -> TestClient:
    from fake_ollama.anthropic_client import AnthropicClient

    app = create_app(settings)
    # Inject the mocked client BEFORE lifespan startup so it is preserved.
    app.state.client = AnthropicClient(
        settings.upstream_url,
        settings.anthropic_auth_token,
        client=httpx.AsyncClient(transport=transport),
    )
    return TestClient(app)


def test_tags_lists_models(settings):
    client = _make_client(settings, httpx.MockTransport(lambda req: httpx.Response(404)))
    with client:
        resp = client.get("/api/tags")
    assert resp.status_code == 200
    body = resp.json()
    names = [m["name"] for m in body["models"]]
    assert "claude-3-5-sonnet-20241022" in names
    assert "llama-test" in names


def test_version_endpoint(settings):
    client = _make_client(settings, httpx.MockTransport(lambda req: httpx.Response(404)))
    with client:
        resp = client.get("/api/version")
    assert resp.status_code == 200
    assert resp.json() == {"version": settings.advertised_version}


def test_show_advertises_capabilities(settings):
    client = _make_client(settings, httpx.MockTransport(lambda req: httpx.Response(404)))
    with client:
        resp = client.post("/api/show", json={"model": "claude-3-5-sonnet-20241022"})
    assert resp.status_code == 200
    body = resp.json()
    # GitHub Copilot and other clients filter out models that don't advertise
    # at least "completion" (and "tools" for tool-calling).
    assert "completion" in body["capabilities"]
    assert "tools" in body["capabilities"]


def test_chat_non_streaming(settings):
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "pong"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == {"role": "assistant", "content": "pong"}
    assert body["done"] is True
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"].get("x-api-key") == "test-token"
    assert captured["body"]["model"] == "claude-3-5-sonnet-20241022"
    assert captured["body"]["max_tokens"] == 1024  # from default


def test_chat_streaming(settings):
    sse_events = [
        {
            "type": "message_start",
            "data": {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 2, "output_tokens": 0}},
            },
        },
        {
            "type": "content_block_delta",
            "data": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
        },
        {
            "type": "content_block_delta",
            "data": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}},
        },
        {
            "type": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
        },
        {"type": "message_stop", "data": {"type": "message_stop"}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_build_sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            lines = [json.loads(line) for line in resp.iter_lines() if line]

    text = "".join(c["message"]["content"] for c in lines if not c["done"])
    assert text == "Hi!"
    final = [c for c in lines if c["done"]]
    assert len(final) == 1
    assert final[0]["done_reason"] == "stop"
    assert final[0]["eval_count"] == 2


def test_generate_non_streaming(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # generate -> a single user message with the prompt
        assert body["messages"][-1]["content"] == "say hi"
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/generate",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "prompt": "say hi",
                "stream": False,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "hi"
    assert body["done"] is True


def test_embeddings_returns_501(settings):
    client = _make_client(settings, httpx.MockTransport(lambda req: httpx.Response(404)))
    with client:
        resp = client.post("/api/embeddings", json={"model": "x", "prompt": "y"})
    assert resp.status_code == 501


def test_openai_chat_completions_non_streaming(settings):
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_xy",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "pong"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "ping"},
                ],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "pong"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 4
    # system prompt forwarded as Anthropic `system`
    assert captured["body"]["system"] == "be brief"


def test_openai_chat_completions_streaming(settings):
    sse_events = [
        {"type": "message_start", "data": {"message": {"id": "msg_z", "usage": {"input_tokens": 1, "output_tokens": 0}}}},
        {"type": "content_block_delta", "data": {"delta": {"type": "text_delta", "text": "Hi"}}},
        {"type": "content_block_delta", "data": {"delta": {"type": "text_delta", "text": "!"}}},
        {"type": "message_delta", "data": {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}},
        {"type": "message_stop", "data": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_build_sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    # Should be SSE: data: {...}\n\n ... data: [DONE]\n\n
    assert raw.endswith("data: [DONE]\n\n")
    frames = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
    assert text == "Hi!"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


def test_openai_models_list(settings):
    client = _make_client(settings, httpx.MockTransport(lambda req: httpx.Response(404)))
    with client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "claude-3-5-sonnet-20241022" in ids


def test_openai_chat_streams_tool_calls(settings):
    """Anthropic tool_use blocks must surface as OpenAI tool_calls deltas."""
    sse_events = [
        {"type": "message_start", "data": {"message": {"id": "msg_t", "usage": {"input_tokens": 1, "output_tokens": 0}}}},
        {"type": "content_block_start", "data": {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}},
        {"type": "content_block_delta", "data": {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}},
        {"type": "content_block_delta", "data": {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}}},
        {"type": "message_delta", "data": {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}},
        {"type": "message_stop", "data": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_build_sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "x",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        },
                    }
                ],
            },
        ) as resp:
            assert resp.status_code == 200
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    frames = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    tool_call_frames = [
        f for f in frames if f["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_call_frames, "expected at least one tool_calls delta frame"
    first = tool_call_frames[0]["choices"][0]["delta"]["tool_calls"][0]
    assert first["id"] == "toolu_1"
    assert first["function"]["name"] == "get_weather"
    args = "".join(
        tc["function"].get("arguments", "")
        for f in tool_call_frames
        for tc in f["choices"][0]["delta"]["tool_calls"]
    )
    assert json.loads(args) == {"city": "Paris"}
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ---- Per-model profile / context guardrail -------------------------------


def _profile_settings(monkeypatch):
    """Helper: build Settings with a tiny custom profile."""
    from fake_ollama.config import get_settings as _gs

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://upstream.test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("FAKE_OLLAMA_MODELS", "tiny-model,big-model")
    monkeypatch.setenv("FAKE_OLLAMA_DEFAULT_MAX_TOKENS", "64")
    monkeypatch.setenv(
        "FAKE_OLLAMA_MODEL_PROFILES",
        json.dumps(
            {
                "tiny-model": {
                    "capabilities": ["completion"],
                    "context_length": 256,
                    "max_output_tokens": 32,
                },
                "big-model": {
                    "capabilities": ["completion", "tools", "vision"],
                    "context_length": 200000,
                },
            }
        ),
    )
    _gs.cache_clear()
    return _gs()


def test_show_uses_per_model_profile(monkeypatch):
    settings = _profile_settings(monkeypatch)
    client = _make_client(settings, httpx.MockTransport(lambda r: httpx.Response(404)))
    with client:
        resp = client.post("/api/show", json={"model": "tiny-model"})
    body = resp.json()
    assert body["capabilities"] == ["completion"]
    assert body["context_length"] == 256
    assert body["model_info"]["general.context_length"] == 256
    assert body["model_info"]["claude.context_length"] == 256


def test_tags_includes_profile_fields(monkeypatch):
    settings = _profile_settings(monkeypatch)
    client = _make_client(settings, httpx.MockTransport(lambda r: httpx.Response(404)))
    with client:
        resp = client.get("/api/tags")
    by_name = {m["name"]: m for m in resp.json()["models"]}
    assert by_name["tiny-model"]["context_length"] == 256
    assert by_name["tiny-model"]["capabilities"] == ["completion"]
    assert "vision" in by_name["big-model"]["capabilities"]


def test_context_limit_enforced(monkeypatch):
    settings = _profile_settings(monkeypatch)

    def upstream_should_not_be_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be reached when limit is exceeded")

    client = _make_client(settings, httpx.MockTransport(upstream_should_not_be_called))
    big_prompt = "x" * 4000  # ~1333 tokens estimate, > 256 ctx
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "tiny-model",
                "messages": [{"role": "user", "content": big_prompt}],
                "stream": False,
            },
        )
    assert resp.status_code == 400
    assert "context window" in resp.json()["detail"]


def test_context_limit_can_be_disabled(monkeypatch):
    settings = _profile_settings(monkeypatch)
    monkeypatch.setenv("FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT", "false")
    from fake_ollama.config import get_settings as _gs
    _gs.cache_clear()
    settings = _gs()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    big_prompt = "x" * 4000
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "tiny-model",
                "messages": [{"role": "user", "content": big_prompt}],
                "stream": False,
            },
        )
    assert resp.status_code == 200


def test_openai_chat_respects_context_limit(monkeypatch):
    settings = _profile_settings(monkeypatch)

    def upstream_should_not_be_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be reached")

    client = _make_client(settings, httpx.MockTransport(upstream_should_not_be_called))
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-model",
                "messages": [{"role": "user", "content": "y" * 4000}],
                "stream": False,
            },
        )
    assert resp.status_code == 400


# ---- Vision (image) handling ---------------------------------------------


import base64 as _b64

PNG_1x1 = _b64.b64encode(
    bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
        "0000000A49444154789C6300010000000500010D0A2DB40000000049454E44AE426082"
    )
).decode()
JPEG_1x1 = _b64.b64encode(
    bytes.fromhex(
        "FFD8FFE000104A46494600010100000100010000FFDB004300080606070605080707"
        "07090908"
    )
    + b"\x00" * 16
).decode()
GIF_1x1 = _b64.b64encode(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;").decode()
WEBP_HEADER = _b64.b64encode(
    b"RIFF\x1c\x00\x00\x00WEBPVP8 \x10\x00\x00\x00" + b"\x00" * 16
).decode()


def test_ollama_image_media_type_detected(settings):
    """Different image magic bytes -> correct Anthropic media_type."""
    captured: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    cases = [
        (PNG_1x1, "image/png"),
        (JPEG_1x1, "image/jpeg"),
        (GIF_1x1, "image/gif"),
        (WEBP_HEADER, "image/webp"),
    ]
    with client:
        for b64, expected in cases:
            captured.clear()
            resp = client.post(
                "/api/chat",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "what?", "images": [b64]}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            sent = captured[0]
            blocks = sent["messages"][0]["content"]
            image_block = next(b for b in blocks if b["type"] == "image")
            assert image_block["source"]["media_type"] == expected, (b64, expected)


def test_openai_image_url_passthrough(settings):
    """OpenAI data: URLs preserve the media_type from the data URI prefix."""
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{JPEG_1x1}"
                                },
                            },
                        ],
                    }
                ],
            },
        )
    assert resp.status_code == 200
    blocks = captured["body"]["messages"][0]["content"]
    img = next(b for b in blocks if b["type"] == "image")
    assert img["source"]["media_type"] == "image/jpeg"


# ---- Thinking / reasoning ------------------------------------------------


def _thinking_settings(monkeypatch, *, mode="enabled", show=True):
    from fake_ollama.config import get_settings as _gs

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://upstream.test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("FAKE_OLLAMA_MODELS", "deepseek-v4-pro")
    monkeypatch.setenv("FAKE_OLLAMA_DEFAULT_MAX_TOKENS", "256")
    monkeypatch.setenv(
        "FAKE_OLLAMA_MODEL_PROFILES",
        json.dumps(
            {
                "deepseek-v4-pro": {
                    "capabilities": ["completion", "tools"],
                    "context_length": 128000,
                    "thinking": mode,
                    "thinking_budget_tokens": 512,
                    "show_thinking": show,
                }
            }
        ),
    )
    _gs.cache_clear()
    return _gs()


def test_thinking_mode_enabled_injects_request_field(monkeypatch):
    settings = _thinking_settings(monkeypatch, mode="enabled")
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    assert captured["body"]["thinking"] == {"type": "enabled", "budget_tokens": 512}


def test_thinking_mode_disabled_injects_disabled(monkeypatch):
    settings = _thinking_settings(monkeypatch, mode="disabled")
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    assert captured["body"]["thinking"] == {"type": "disabled"}


def test_thinking_in_non_stream_response_wrapped_for_ollama(monkeypatch):
    settings = _thinking_settings(monkeypatch, mode="enabled", show=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "let me think"},
                    {"type": "text", "text": "the answer is 42"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 6},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "compute"}],
                "stream": False,
            },
        )
    body = resp.json()
    assert "<think>let me think</think>" in body["message"]["content"]
    assert "the answer is 42" in body["message"]["content"]
    assert body["message"]["thinking"] == "let me think"


def test_thinking_hidden_when_show_false(monkeypatch):
    settings = _thinking_settings(monkeypatch, mode="enabled", show=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {"type": "text", "text": "visible"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "?"}],
                "stream": False,
            },
        )
    body = resp.json()
    assert body["message"]["content"] == "visible"
    assert "secret" not in body["message"]["content"]
    assert "thinking" not in body["message"]


def test_thinking_streamed_to_openai_with_reasoning_content(monkeypatch):
    settings = _thinking_settings(monkeypatch, mode="enabled", show=True)
    sse_events = [
        {"type": "message_start", "data": {"message": {"id": "msg_t", "usage": {"input_tokens": 1, "output_tokens": 0}}}},
        {"type": "content_block_start", "data": {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}},
        {"type": "content_block_delta", "data": {"index": 0, "delta": {"type": "thinking_delta", "thinking": "step1"}}},
        {"type": "content_block_delta", "data": {"index": 0, "delta": {"type": "thinking_delta", "thinking": " step2"}}},
        {"type": "content_block_stop", "data": {"index": 0}},
        {"type": "content_block_start", "data": {"index": 1, "content_block": {"type": "text", "text": ""}}},
        {"type": "content_block_delta", "data": {"index": 1, "delta": {"type": "text_delta", "text": "answer"}}},
        {"type": "message_delta", "data": {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}},
        {"type": "message_stop", "data": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_build_sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "?"}],
                "stream": True,
            },
        ) as resp:
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    frames = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    content = "".join(f["choices"][0]["delta"].get("content", "") or "" for f in frames)
    reasoning = "".join(
        f["choices"][0]["delta"].get("reasoning_content", "") or "" for f in frames
    )
    assert reasoning == "step1 step2"
    assert "<think>step1 step2</think>" in content
    assert "answer" in content
    # </think> must come BEFORE the visible text
    assert content.index("</think>") < content.index("answer")


def test_thinking_streamed_to_ollama_wraps_in_think_tags(monkeypatch):
    settings = _thinking_settings(monkeypatch, mode="enabled", show=True)
    sse_events = [
        {"type": "message_start", "data": {"message": {"id": "msg_t", "usage": {"input_tokens": 1, "output_tokens": 0}}}},
        {"type": "content_block_start", "data": {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}},
        {"type": "content_block_delta", "data": {"index": 0, "delta": {"type": "thinking_delta", "thinking": "reasoning"}}},
        {"type": "content_block_stop", "data": {"index": 0}},
        {"type": "content_block_delta", "data": {"index": 1, "delta": {"type": "text_delta", "text": "result"}}},
        {"type": "message_delta", "data": {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}},
        {"type": "message_stop", "data": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_build_sse(sse_events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "?"}],
                "stream": True,
            },
        ) as resp:
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    chunks = [json.loads(l) for l in raw.splitlines() if l.strip()]
    text = "".join(c.get("message", {}).get("content", "") for c in chunks)
    assert "<think>reasoning</think>" in text
    assert "result" in text
    assert text.index("</think>") < text.index("result")


def test_thinking_auto_does_not_inject(settings):
    """Default profile (auto) must not touch the upstream `thinking` field."""
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    assert "thinking" not in captured["body"]


def test_unknown_model_falls_back_to_passthrough(settings):
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _make_client(settings, httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/api/chat",
            json={
                "model": "claude-custom:latest",
                "messages": [{"role": "user", "content": "x"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    # ":latest" suffix should be stripped before passthrough
    assert captured["body"]["model"] == "claude-custom"
