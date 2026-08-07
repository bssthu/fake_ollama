"""Contract tests for OpenAI-compatible upstream support."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.config import (
    AnthropicUpstream,
    ApiInterface,
    Backend,
    ExposureEntry,
    LlamaCppDefaults,
    ModelEntry,
    OllamaInterface,
    OllamaTarget,
    OpenAIUpstream,
    Settings,
)
from fake_ollama.openai_client import OpenAIClient
from fake_ollama.server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings_with_openai(**overrides: Any) -> Settings:
    base: Dict[str, Any] = dict(
        anthropic_upstreams=[
            AnthropicUpstream(
                name="anthropic",
                base_url="http://anthropic.test",
                auth_token="ak",
                models=[ModelEntry(name="claude-3-5-sonnet")],
            )
        ],
        openai_upstreams=[
            OpenAIUpstream(
                name="deepseek",
                base_url="http://openai.test",
                auth_token="sk-openai",
                models=[
                    ModelEntry(name="deepseek-chat"),
                    # display "deepseek-reasoner" → wire "deepseek-r1"
                    ModelEntry(name="deepseek-r1", alias="deepseek-reasoner"),
                ],
            )
        ],
        ollama_targets=[],
        llama_cpp_targets=[],
        llama_cpp_defaults=LlamaCppDefaults(),
        ollama_interfaces=[
            OllamaInterface(
                name="ollama",
                port=21434,
                exposed_models=[
                    ExposureEntry(model="claude-3-5-sonnet", target="anthropic"),
                    ExposureEntry(model="deepseek-chat", target="deepseek"),
                    ExposureEntry(model="deepseek-reasoner", target="deepseek"),
                ],
            )
        ],
        api_interfaces=[
            ApiInterface(
                name="api",
                port=21435,
                exposed_models=[
                    ExposureEntry(model="claude-3-5-sonnet", target="anthropic"),
                    ExposureEntry(model="deepseek-chat", target="deepseek"),
                    ExposureEntry(model="deepseek-reasoner", target="deepseek"),
                ],
            )
        ],
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Config layer
# ---------------------------------------------------------------------------


def test_backend_from_openai_upstream_is_remote_openai() -> None:
    up = OpenAIUpstream(
        name="dseek",
        base_url="https://api.deepseek.com",
        auth_token="sk",
        models=[ModelEntry(name="deepseek-chat")],
    )
    settings = Settings(openai_upstreams=[up], ollama_interfaces=[])
    b = Backend.from_source(up, settings)
    assert b.protocol == "openai"
    assert b.kind == "remote"
    assert b.name == "dseek"
    assert b.auth_token == "sk"
    assert b.source.serves("deepseek-chat")
    assert b.supports_lifecycle is False


def test_settings_backend_by_name_finds_openai() -> None:
    settings = _settings_with_openai()
    b = settings.backend_by_name("deepseek")
    assert b is not None
    assert b.protocol == "openai"
    assert b.kind == "remote"


def test_resolve_request_routes_to_openai_upstream() -> None:
    settings = _settings_with_openai()
    b, display = settings.resolve_request(
        "deepseek-chat@deepseek", interface_name="ollama"
    )
    assert b.protocol == "openai"
    assert b.name == "deepseek"
    assert display == "deepseek-chat"


def test_resolve_request_alias_maps_to_wire_id() -> None:
    settings = _settings_with_openai()
    b, display = settings.resolve_request(
        "deepseek-reasoner@deepseek", interface_name="api"
    )
    assert display == "deepseek-reasoner"
    # Source maps the display alias back to the wire id "deepseek-r1".
    assert b.source.resolve_model(display) == "deepseek-r1"


def test_settings_validates_duplicate_names_across_protocols() -> None:
    with pytest.raises(Exception):
        Settings(
            anthropic_upstreams=[
                AnthropicUpstream(
                    name="shared",
                    base_url="http://a.test",
                    auth_token="x",
                    models=[ModelEntry(name="m")],
                )
            ],
            openai_upstreams=[
                OpenAIUpstream(
                    name="shared",
                    base_url="http://o.test",
                    auth_token="y",
                    models=[ModelEntry(name="m2")],
                )
            ],
            ollama_interfaces=[],
        )


def test_settings_all_source_composite_ids_includes_openai() -> None:
    settings = _settings_with_openai()
    assert set(settings.all_source_composite_ids()) == {
        "claude-3-5-sonnet@anthropic",
        "deepseek-chat@deepseek",
        "deepseek-reasoner@deepseek",
    }


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


def test_openai_client_chat_posts_to_v1_chat_completions_with_bearer() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    oc = OpenAIClient(
        "http://openai.test/", auth_token="sk-x", client=http_client
    )

    async def run() -> Dict[str, Any]:
        try:
            return await oc.chat({"model": "deepseek-chat", "messages": []})
        finally:
            await oc.aclose()
            await http_client.aclose()

    resp = asyncio.run(run())
    assert resp["choices"][0]["message"]["content"] == "hi"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-x"
    assert captured["headers"]["x-api-key"] == "sk-x"
    assert captured["body"]["stream"] is False
    assert captured["body"]["model"] == "deepseek-chat"


def test_openai_client_stream_chat_yields_raw_sse_lines() -> None:
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"he"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"llo"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(chunks),
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    oc = OpenAIClient("http://openai.test", client=http_client)

    async def run() -> List[str]:
        try:
            out: List[str] = []
            async for line in oc.stream_chat(
                {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
            ):
                out.append(line)
            return out
        finally:
            await oc.aclose()
            await http_client.aclose()

    lines = asyncio.run(run())
    assert any(line.startswith("data: ") and '"he"' in line for line in lines)
    assert lines[-1].strip() == "data: [DONE]"


# ---------------------------------------------------------------------------
# End-to-end through the FastAPI app
# ---------------------------------------------------------------------------


def _install_openai_client(
    app, *, response: Dict[str, Any] | None = None, stream_lines: List[str] | None = None,
) -> Dict[str, Any]:
    captured: Dict[str, Any] = {"chat_calls": [], "stream_calls": []}

    class _Stub:
        target_id = "openai:stub"

        async def chat(self, payload):
            captured["chat_calls"].append(payload)
            return response or {
                "id": "cmpl-x",
                "model": payload.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "PONG"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        async def stream_chat(self, payload) -> AsyncIterator[str]:
            captured["stream_calls"].append(payload)
            for line in stream_lines or [
                'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"PONG"}}]}',
                "data: [DONE]",
            ]:
                yield line

        async def aclose(self) -> None:
            return None

    app.state.openai_clients = {"deepseek": _Stub()}
    return captured


def test_ollama_chat_routes_via_openai_upstream() -> None:
    settings = _settings_with_openai()
    app = create_app(settings)
    with TestClient(app) as client:
        captured = _install_openai_client(app)
        resp = client.post(
            "/api/chat",
            json={
                "model": "deepseek-chat@deepseek",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["done"] is True
        assert body["message"]["role"] == "assistant"
        assert "PONG" in body["message"]["content"]
        assert captured["chat_calls"], "OpenAIClient.chat was not called"
        assert captured["chat_calls"][0]["model"] == "deepseek-chat"


def test_openai_chat_routes_via_openai_upstream_passthrough() -> None:
    settings = _settings_with_openai()
    app = create_app(settings)
    with TestClient(app) as client:
        captured = _install_openai_client(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-reasoner@deepseek",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "deepseek-reasoner@deepseek"
        # Upstream sees the resolved (wire) model id.
        assert captured["chat_calls"][0]["model"] == "deepseek-r1"


def test_anthropic_messages_routes_via_openai_upstream() -> None:
    settings = _settings_with_openai()
    app = create_app(settings)
    with TestClient(app) as client:
        _install_openai_client(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "deepseek-chat@deepseek",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 64,
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        joined = "".join(
            blk.get("text", "")
            for blk in body.get("content", [])
            if blk.get("type") == "text"
        )
        assert "PONG" in joined
