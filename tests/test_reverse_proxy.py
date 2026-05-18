"""Tests for the reverse proxy endpoint POST /v1/messages."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.config import (
    OllamaTarget,
    Settings,
    estimate_tokens_from_anthropic_payload,
    load_settings,
)
from fake_ollama.server import create_app
from fake_ollama.vram import LocalTargetResourceError


# ---------------------------------------------------------------------------
# Fake OllamaClient used for the in-memory reverse-proxy tests.
# ---------------------------------------------------------------------------


class _FakeOllamaClient:
    def __init__(
        self,
        chat_response: Optional[Dict[str, Any]] = None,
        stream_lines: Optional[List[Dict[str, Any]]] = None,
    ):
        self.chat_response = chat_response or {}
        self.stream_lines = stream_lines or []
        self.last_chat_payload: Optional[Dict[str, Any]] = None
        self.last_stream_payload: Optional[Dict[str, Any]] = None
        self.last_estimated_vram_gb: Optional[float] = None

    async def chat(
        self, payload: Dict[str, Any], *, estimated_vram_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        self.last_chat_payload = payload
        self.last_estimated_vram_gb = estimated_vram_gb
        return self.chat_response

    def stream_chat(
        self, payload: Dict[str, Any], *, estimated_vram_gb: Optional[float] = None
    ) -> AsyncIterator[bytes]:
        self.last_stream_payload = payload
        self.last_estimated_vram_gb = estimated_vram_gb
        lines = self.stream_lines

        async def gen() -> AsyncIterator[bytes]:
            for ln in lines:
                yield (json.dumps(ln) + "\n").encode("utf-8")

        return gen()

    async def aclose(self) -> None:  # pragma: no cover - nothing to do
        return None


class _FakeLlamaCppClient:
    idle_timeout_seconds = None

    def __init__(
        self,
        chat_response: Optional[Dict[str, Any]] = None,
        stream_chunks: Optional[List[Dict[str, Any]]] = None,
    ):
        self.chat_response = chat_response or {}
        self.stream_chunks = stream_chunks or []
        self.last_chat_payload: Optional[Dict[str, Any]] = None
        self.last_stream_payload: Optional[Dict[str, Any]] = None
        self.last_estimated_vram_gb: Optional[float] = None

    async def chat(
        self, payload: Dict[str, Any], *, estimated_vram_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        self.last_chat_payload = payload
        self.last_estimated_vram_gb = estimated_vram_gb
        return self.chat_response

    async def stream_chat(
        self, payload: Dict[str, Any], *, estimated_vram_gb: Optional[float] = None
    ) -> AsyncIterator[str]:
        self.last_stream_payload = payload
        self.last_estimated_vram_gb = estimated_vram_gb
        for chunk in self.stream_chunks:
            yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"

    async def aclose(self) -> None:  # pragma: no cover - nothing to do
        return None


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def reverse_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with one upstream + one ollama_target serving 'llama3.1'."""
    return Settings(
        upstreams=[
            {
                "name": "default",
                "base_url": "http://upstream.test",
                "auth_token": "tk",
                "models": ["claude-3-5-sonnet-20241022"],
            }
        ],
        ollama_targets=[
            {
                "name": "local",
                "base_url": "http://127.0.0.1:11434",
                "models": ["llama3.1"],
                "model_map": {"llama3.1": "llama3.1:8b"},
            }
        ],
        external_access_tokens=["rev-tk-1"],
        internal_exposed_models=[
            "claude-3-5-sonnet-20241022@default",
            "llama3.1@local",
        ],
        external_exposed_models=[
            "claude-3-5-sonnet-20241022@default",
            "llama3.1@local",
        ],
    )


# Header sent by all reverse-proxy tests below.
_AUTH = {"x-api-key": "rev-tk-1"}


def _expose_all(data: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-populate internal/external_exposed_models for every backend's models.

    Lets bulk-rewritten tests construct ``Settings(**_expose_all(data))``
    without manually listing composite ids.
    """
    exposed: List[str] = []
    for section in ("upstreams", "openai_upstreams", "ollama_targets"):
        for src in data.get(section, []) or []:
            name = src["name"]
            for m in src.get("models", []) or []:
                exposed.append(f"{m}@{name}")
    for tgt in data.get("llama_cpp_targets", []) or []:
        name = tgt["name"]
        m = tgt.get("model")
        if m:
            exposed.append(f"{m}@{name}")
    # Union with any caller-provided lists.
    existing_int = set(data.get("internal_exposed_models") or [])
    existing_ext = set(data.get("external_exposed_models") or [])
    data["internal_exposed_models"] = sorted(existing_int | set(exposed))
    data["external_exposed_models"] = sorted(existing_ext | set(exposed))
    return data


def _build_client(
    settings: Settings,
    *,
    fake_ollama: Optional[_FakeOllamaClient] = None,
    fake_llama_cpp: Optional[_FakeLlamaCppClient] = None,
    upstream_transport: Optional[httpx.MockTransport] = None,
    base_url: str = "http://testserver",
) -> TestClient:
    from fake_ollama.anthropic_client import AnthropicClient

    app = create_app(settings)
    app.state.ollama_clients = (
        {tgt.name: fake_ollama for tgt in settings.ollama_targets}
        if fake_ollama is not None
        else {}
    )
    app.state.llama_cpp_clients = (
        {tgt.name: fake_llama_cpp for tgt in settings.llama_cpp_targets}
        if fake_llama_cpp is not None
        else {}
    )
    if upstream_transport is not None:
        app.state.clients = {
            up.name: AnthropicClient(
                up.base_url, up.auth_token,
                client=httpx.AsyncClient(transport=upstream_transport),
            )
            for up in settings.upstreams
        }
    return TestClient(app, base_url=base_url)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reverse_non_stream_text(reverse_settings):
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "hi there"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
        }
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "assistant"
    assert body["model"] == "llama3.1@local"
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "hi there"}]
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3}
    # Verify request was converted: model_map applied, num_predict set.
    sent = fake.last_chat_payload
    assert sent["model"] == "llama3.1:8b"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["options"]["num_predict"] == 100


def test_reverse_target_passes_per_model_estimated_vram(reverse_settings):
    data = reverse_settings.model_dump()
    data["model_profiles"] = {"llama3.1": {"estimated_vram_gb": 6.5}}
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
    )
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_estimated_vram_gb == 6.5


def test_reverse_count_tokens_estimates_for_ollama_target(reverse_settings):
    client = _build_client(reverse_settings)
    payload = {
        "model": "llama3.1@local",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "tool output"}],
                    },
                ],
            }
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "look something up",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
    }
    with client:
        resp = client.post("/v1/messages/count_tokens?beta=true", headers=_AUTH, json=payload)

    assert resp.status_code == 200
    assert resp.json() == {
        "input_tokens": estimate_tokens_from_anthropic_payload(payload)
    }


def test_reverse_messages_returns_anthropic_error_for_insufficient_vram(reverse_settings):
    class FailingOllamaClient(_FakeOllamaClient):
        async def chat(
            self,
            payload: Dict[str, Any],
            *,
            estimated_vram_gb: Optional[float] = None,
        ) -> Dict[str, Any]:
            raise LocalTargetResourceError(
                "Insufficient GPU VRAM to start local model 'llama3.1'."
            )

    client = _build_client(reverse_settings, fake_ollama=FailingOllamaClient())
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 503
    assert resp.json() == {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": "Insufficient GPU VRAM to start local model 'llama3.1'.",
        },
    }


def test_reverse_count_tokens_passthrough_to_upstream(reverse_settings):
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = request.url.query.decode()
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"input_tokens": 42})

    client = _build_client(
        reverse_settings,
        upstream_transport=httpx.MockTransport(handler),
    )
    with client:
        resp = client.post(
            "/v1/messages/count_tokens?beta=true",
            headers=_AUTH,
            json={
                "model": "claude-3-5-sonnet-20241022@default",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 42}
    assert captured["path"] == "/v1/messages/count_tokens"
    assert captured["query"] == "beta=true"
    assert captured["headers"]["x-api-key"] == "tk"
    # The composite ``@default`` suffix is fake-ollama routing syntax
    # and must be stripped before talking to a real Anthropic upstream.
    assert captured["body"] == {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_reverse_forwards_base64_image_to_ollama(reverse_settings):
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "it is an image"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    image_data = "ZmFrZS1wbmc="
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                        ],
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["messages"] == [
        {"role": "user", "content": "what is this?", "images": [image_data]}
    ]


def test_reverse_target_disables_ollama_thinking_from_profile(reverse_settings):
    data = reverse_settings.model_dump()
    data["model_profiles"] = {
        "llama3.1": {
            "capabilities": ["completion", "tools"],
            "thinking_mode": "disabled",
            "show_thinking": False,
        }
    }
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["think"] is False


def test_reverse_target_profile_disabled_overrides_client_thinking_enabled(
    reverse_settings,
):
    data = reverse_settings.model_dump()
    data["model_profiles"] = {
        "llama3.1": {
            "capabilities": ["completion", "tools"],
            "thinking_mode": "disabled",
            "show_thinking": True,
        }
    }
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "thinking": {"type": "enabled", "budget_tokens": 99},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["think"] is False


def test_reverse_target_profile_enabled_overrides_client_thinking_disabled(
    reverse_settings,
):
    data = reverse_settings.model_dump()
    data["model_profiles"] = {
        "llama3.1": {
            "capabilities": ["completion", "tools"],
            "thinking_mode": "enabled",
            "show_thinking": True,
        }
    }
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 100,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["think"] is True


def test_openai_target_disables_ollama_thinking_from_profile(reverse_settings):
    data = reverse_settings.model_dump()
    data["model_profiles"] = {
        "llama3.1": {
            "capabilities": ["completion", "tools"],
            "thinking_mode": "disabled",
            "show_thinking": False,
        }
    }
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1@local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["think"] is False


def test_openai_chat_non_stream_routes_to_ollama_target(reverse_settings):
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "hi from local"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 6,
            "eval_count": 3,
        }
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1@local",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hello"},
                ],
                "stream": False,
                "max_tokens": 50,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "llama3.1@local"
    assert body["choices"][0]["message"]["content"] == "hi from local"
    assert body["usage"] == {
        "prompt_tokens": 6,
        "completion_tokens": 3,
        "total_tokens": 9,
    }
    sent = fake.last_chat_payload
    assert sent["model"] == "llama3.1:8b"
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert sent["options"]["num_predict"] == 50


def test_openai_chat_image_routes_to_ollama_target(reverse_settings):
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "saw it"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    image_data = "ZmFrZS1qcGVn"
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1@local",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                "stream": False,
                "max_tokens": 50,
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["messages"] == [
        {"role": "user", "content": "what is this?", "images": [image_data]}
    ]


def test_openai_chat_stream_routes_to_ollama_target(reverse_settings):
    fake = _FakeOllamaClient(
        stream_lines=[
            {"message": {"role": "assistant", "content": "hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        ]
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "llama3.1@local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    assert fake.last_stream_payload["model"] == "llama3.1:8b"
    assert raw.endswith("data: [DONE]\n\n")
    frames = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert "".join(
        f["choices"][0]["delta"].get("content", "") or "" for f in frames
    ) == "hello"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


def test_reverse_non_stream_routes_to_llama_cpp_target(reverse_settings):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {
            "name": "qwen36",
            "base_url": "http://127.0.0.1:21436",
            "model": "qwen3.6",
            "model_alias": "qwen3.6-alias",
        }
    ]
    data["model_profiles"] = {"qwen3.6": {"show_thinking": False}}
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        chat_response={
            "id": "chatcmpl_local",
            "object": "chat.completion",
            "model": "qwen3.6-alias",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hi from llama.cpp",
                        "reasoning_content": "hidden",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "qwen3.6@qwen36"
    assert body["content"] == [{"type": "text", "text": "hi from llama.cpp"}]
    assert body["usage"] == {"input_tokens": 8, "output_tokens": 4}
    sent = fake.last_chat_payload
    assert sent["model"] == "qwen3.6-alias"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["max_tokens"] == 100
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_reverse_llama_cpp_profile_disabled_overrides_client_thinking_enabled(
    reverse_settings,
):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "base_url": "http://127.0.0.1:21436", "model": "qwen3.6"}
    ]
    data["model_profiles"] = {
        "qwen3.6": {
            "capabilities": ["completion", "tools"],
            "thinking_mode": "disabled",
            "show_thinking": False,
        }
    }
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        chat_response={
            "id": "chatcmpl_local",
            "object": "chat.completion",
            "model": "qwen3.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "max_tokens": 100,
                "thinking": {"type": "enabled", "budget_tokens": 99},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_reverse_llama_cpp_auto_honors_client_thinking_enabled(reverse_settings):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "base_url": "http://127.0.0.1:21436", "model": "qwen3.6"}
    ]
    data["model_profiles"] = {"qwen3.6": {"show_thinking": False}}
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        chat_response={
            "id": "chatcmpl_local",
            "object": "chat.completion",
            "model": "qwen3.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "max_tokens": 100,
                "thinking": {"type": "enabled", "budget_tokens": 99},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_reverse_non_stream_salvages_reasoning_tool_call_from_llama_cpp(
    reverse_settings,
):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "base_url": "http://127.0.0.1:21436", "model": "qwen3.6"}
    ]
    data["model_profiles"] = {"qwen3.6": {"show_thinking": False}}
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        chat_response={
            "id": "chatcmpl_local",
            "object": "chat.completion",
            "model": "qwen3.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": (
                            "<tool_call>\n"
                            "<function=Bash>\n"
                            "<parameter=command>\n"
                            "echo 1\n"
                            "</parameter>\n"
                            "<parameter=description>\n"
                            "Run first step\n"
                            "</parameter>\n"
                            "</function>\n"
                            "</tool_call>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "call_0",
            "name": "Bash",
            "input": {"command": "echo 1", "description": "Run first step"},
        }
    ]


def test_reverse_forwards_base64_image_to_llama_cpp(reverse_settings):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "model": "qwen3.6", "base_url": "http://x.test"}
    ]
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        chat_response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "saw it"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    image_data = "ZmFrZS1wbmc="
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                        ],
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
        }
    ]


def test_openai_chat_routes_to_llama_cpp_target(reverse_settings):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {
            "name": "qwen36",
            "base_url": "http://127.0.0.1:21436",
            "model": "qwen3.6",
            "model_alias": "qwen3.6-alias",
        }
    ]
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        chat_response={
            "id": "chatcmpl_local",
            "object": "chat.completion",
            "model": "qwen3.6-alias",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "direct openai"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6@qwen36",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen3.6@qwen36"
    assert resp.json()["choices"][0]["message"]["content"] == "direct openai"
    assert fake.last_chat_payload["model"] == "qwen3.6-alias"


def test_reverse_streaming_routes_to_llama_cpp_target(reverse_settings):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "base_url": "http://127.0.0.1:21436", "model": "qwen3.6"}
    ]
    data["model_profiles"] = {"qwen3.6": {"show_thinking": False}}
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        stream_chunks=[
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": "hel"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": "lo"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ]
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        with client.stream(
            "POST",
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "stream": True,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            assert resp.status_code == 200
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    assert fake.last_stream_payload["model"] == "qwen3.6"
    assert "event: message_start" in raw
    assert '"text": "hel"' in raw
    assert '"text": "lo"' in raw
    assert '"stop_reason": "end_turn"' in raw


def test_reverse_streaming_salvages_reasoning_tool_call_from_llama_cpp(
    reverse_settings,
):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "base_url": "http://127.0.0.1:21436", "model": "qwen3.6"}
    ]
    data["model_profiles"] = {"qwen3.6": {"show_thinking": False}}
    settings = Settings(**_expose_all(data))
    fake = _FakeLlamaCppClient(
        stream_chunks=[
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "<tool_call>\n<function=Bash>"
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": (
                                "\n<parameter=command>\necho 1\n</parameter>"
                            )
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": (
                                "\n<parameter=description>\nRun first step\n"
                                "</parameter>\n</function>\n</tool_call>"
                            )
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ]
    )
    client = _build_client(settings, fake_llama_cpp=fake)
    with client:
        with client.stream(
            "POST",
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6@qwen36",
                "stream": True,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            assert resp.status_code == 200
            raw = b"".join(resp.iter_bytes()).decode("utf-8")

    events = []
    for part in raw.split("\n\n"):
        data_line = next(
            (line for line in part.splitlines() if line.startswith("data: ")), None
        )
        if data_line:
            events.append(json.loads(data_line[len("data: "):]))

    tool_start = next(
        e
        for e in events
        if e["type"] == "content_block_start"
        and e["content_block"]["type"] == "tool_use"
    )
    tool_delta = next(
        e
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"]["type"] == "input_json_delta"
    )
    message_delta = next(e for e in events if e["type"] == "message_delta")

    assert "thinking_delta" not in raw
    assert tool_start["content_block"]["name"] == "Bash"
    assert json.loads(tool_delta["delta"]["partial_json"]) == {
        "command": "echo 1",
        "description": "Run first step",
    }
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_v1_models_includes_llama_cpp_targets_when_authed(reverse_settings):
    data = reverse_settings.model_dump()
    data["llama_cpp_targets"] = [
        {"name": "qwen36", "base_url": "http://127.0.0.1:21436", "model": "qwen3.6"}
    ]
    settings = Settings(**_expose_all(data))
    client = _build_client(settings, fake_llama_cpp=_FakeLlamaCppClient())
    with client:
        r = client.get("/v1/models", headers=_AUTH)
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "qwen3.6@qwen36" in ids


def test_openai_chat_internal_port_routes_to_upstream(reverse_settings):
    data = reverse_settings.model_dump()
    data["external_host"] = "127.0.0.1"
    data["external_port"] = 21435
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient()
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_upstream",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "from upstream"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 2},
            },
        )

    client = _build_client(
        settings,
        fake_ollama=fake,
        upstream_transport=httpx.MockTransport(handler),
        base_url="http://testserver:21434",
    )
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022@default",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "from upstream"
    assert captured["body"]["model"] == "claude-3-5-sonnet-20241022"
    assert fake.last_chat_payload is None


def test_openai_chat_external_port_routes_to_ollama_target(reverse_settings):
    data = reverse_settings.model_dump()
    data["external_host"] = "127.0.0.1"
    data["external_port"] = 21435
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "from local"},
            "done": True,
            "done_reason": "stop",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("external OpenAI route should use ollama target")

    client = _build_client(
        settings,
        fake_ollama=fake,
        upstream_transport=httpx.MockTransport(handler),
        base_url="http://testserver:21435",
    )
    with client:
        resp = client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "from local"
    assert fake.last_chat_payload["model"] == "llama3.1:8b"


def test_reverse_tagless_alias_routes_to_tagged_ollama_target(reverse_settings):
    data = reverse_settings.model_dump()
    data["ollama_targets"][0]["models"] = ["qwen3.5-2b:latest"]
    data["ollama_targets"][0]["model_map"] = {}
    # Tagless alias: expose both forms so a bare request can be routed.
    data["internal_exposed_models"] = ["qwen3.5-2b@local"]
    data["external_exposed_models"] = ["qwen3.5-2b@local"]
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient(
        chat_response={
            "model": "qwen3.5-2b:latest",
            "message": {"role": "assistant", "content": "local"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.5-2b@local",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert fake.last_chat_payload["model"] == "qwen3.5-2b:latest"
    assert resp.json()["model"] == "qwen3.5-2b@local"


def test_reverse_tagless_alias_does_not_route_to_non_latest_tag(reverse_settings):
    data = reverse_settings.model_dump()
    data["upstreams"][0]["expose_external"] = []
    data["ollama_targets"][0]["models"] = ["qwen3.6-27b:q2_k_p"]
    data["ollama_targets"][0]["model_map"] = {}
    data["ollama_targets"][0]["expose_external"] = ["qwen3.6-27b:q2_k_p"]
    settings = Settings(**_expose_all(data))
    fake = _FakeOllamaClient()
    client = _build_client(settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "qwen3.6-27b@local",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 404
    assert fake.last_chat_payload is None


def test_reverse_non_stream_tool_use(reverse_settings):
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                        },
                        "id": "call_abc",
                    }
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
            "prompt_eval_count": 12,
            "eval_count": 5,
        }
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 64,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {"city": "Tokyo"}
    # Request converted tools.
    sent_tools = fake.last_chat_payload["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["function"]["name"] == "get_weather"


def test_reverse_streaming(reverse_settings):
    fake = _FakeOllamaClient(
        stream_lines=[
            {"message": {"role": "assistant", "content": "hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 2,
            },
        ]
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        with client.stream(
            "POST",
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "stream": True,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            assert resp.status_code == 200
            raw = b"".join(resp.iter_bytes())

    text = raw.decode("utf-8")
    # Sequence sanity: message_start, content_block_start, deltas, stop, message_delta, message_stop
    assert "event: message_start" in text
    assert "event: content_block_start" in text
    assert "\"text\": \"hel\"" in text
    assert "\"text\": \"lo\"" in text
    assert "event: content_block_stop" in text
    assert "event: message_delta" in text
    assert "\"stop_reason\": \"end_turn\"" in text
    assert "event: message_stop" in text


def test_reverse_streaming_tool_use(reverse_settings):
    fake = _FakeOllamaClient(
        stream_lines=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "f",
                                "arguments": {"x": 1},
                            },
                            "id": "call_z",
                        }
                    ],
                },
                "done": True,
                "done_reason": "tool_calls",
                "prompt_eval_count": 4,
                "eval_count": 2,
            }
        ]
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        with client.stream(
            "POST",
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "stream": True,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "go"}],
            },
        ) as resp:
            assert resp.status_code == 200
            text = b"".join(resp.iter_bytes()).decode("utf-8")
    assert "\"type\": \"tool_use\"" in text
    assert "\"name\": \"f\"" in text
    assert "input_json_delta" in text
    assert "\"stop_reason\": \"tool_use\"" in text


def test_reverse_passthrough_when_no_target(reverse_settings):
    """Model not in any ollama_target -> falls through to upstream."""

    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "from upstream"}],
                "model": "claude-3-5-sonnet-20241022@default",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    client = _build_client(
        reverse_settings,
        upstream_transport=httpx.MockTransport(handler),
    )
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "claude-3-5-sonnet-20241022@default",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == [{"type": "text", "text": "from upstream"}]
    assert captured["body"]["model"] == "claude-3-5-sonnet-20241022"


def test_reverse_missing_model_returns_400(reverse_settings):
    client = _build_client(reverse_settings, fake_ollama=_FakeOllamaClient())
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 400


def test_reverse_tool_result_to_tool_role(reverse_settings):
    """Anthropic tool_result inside user message -> Ollama 'tool' role."""
    fake = _FakeOllamaClient(
        chat_response={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
        }
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "llama3.1@local",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_1",
                                "name": "f",
                                "input": {"a": 1},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_1",
                                "content": "sunny",
                            }
                        ],
                    },
                ],
            },
        )
    assert resp.status_code == 200
    msgs = fake.last_chat_payload["messages"]
    # last message should be a tool-role with content "sunny"
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["content"] == "sunny"
    assert msgs[-1]["tool_call_id"] == "call_1"


def test_reverse_missing_token_returns_401(reverse_settings):
    fake = _FakeOllamaClient(
        chat_response={"message": {"role": "assistant", "content": "x"}, "done": True}
    )
    client = _build_client(reverse_settings, fake_ollama=fake)
    payload = {
        "model": "llama3.1@local",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with client:
        # No auth header at all.
        resp = client.post("/v1/messages", json=payload)
        assert resp.status_code == 401
        # Wrong token.
        resp = client.post("/v1/messages", headers={"x-api-key": "nope"}, json=payload)
        assert resp.status_code == 401
        # Bearer also accepted.
        resp = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer rev-tk-1"},
            json=payload,
        )
        assert resp.status_code == 200


def test_reverse_token_required_in_settings():
    """Legacy per-target ``api_token`` is silently dropped (extra='ignore').

    The Settings instance still loads and the target is still routable;
    auth is now centralised on ``external_access_tokens``.
    """
    s = Settings(
        upstreams=[
            {
                "name": "u",
                "base_url": "http://x.test",
                "auth_token": "t",
                "models": ["m"],
            }
        ],
        ollama_targets=[
            {
                "name": "local",
                "base_url": "http://127.0.0.1:11434",
                "models": ["llama3.1"],
                "api_token": "ignored",
            }
        ],
        internal_exposed_models=["m@u", "llama3.1@local"],
        external_exposed_models=["m@u", "llama3.1@local"],
    )
    # Target is routable regardless of legacy api_token.
    assert s.backend_for("llama3.1@local") is not None
    # No central tokens -> no auth required for /v1/* (legacy permissive mode).
    assert s.auth_required_for_v1 is False
    assert s.is_valid_external_token("ignored") is False


def test_legacy_target_api_token_is_silently_dropped(tmp_path, monkeypatch):
    """Loading config.json with legacy per-target ``api_token`` no longer
    migrates it; auth tokens must be set centrally."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({
            "upstreams": [{"name": "u", "base_url": "http://x.test",
                           "auth_token": "t", "models": ["m"]}],
            "ollama_targets": [
                {"name": "local", "base_url": "http://127.0.0.1:11434",
                 "models": ["llama3.1"], "api_token": "legacy-tk"}
            ],
            "internal_exposed_models": ["m@u", "llama3.1@local"],
            "external_exposed_models": ["m@u", "llama3.1@local"],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    s = load_settings()
    assert s.external_access_tokens == []
    assert s.backend_for("llama3.1@local") is not None


def test_v1_models_includes_reverse_targets_when_authed(reverse_settings):
    client = _build_client(reverse_settings, fake_ollama=_FakeOllamaClient())
    with client:
        # Without token: 401 because external_access_tokens is non-empty.
        r = client.get("/v1/models")
        assert r.status_code == 401
        # With token: lists upstream + ollama target models.
        r = client.get("/v1/models", headers=_AUTH)
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        assert "llama3.1@local" in ids
        assert "claude-3-5-sonnet-20241022@default" in ids


def test_v1_models_open_when_no_tokens(tmp_path, monkeypatch):
    """No external_access_tokens -> /v1/models requires no auth."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({
            "upstreams": [{"name": "u", "base_url": "http://upstream.test",
                           "auth_token": "tk", "models": ["m1"]}],
            "internal_exposed_models": ["m1@u"],
            "external_exposed_models": ["m1@u"],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_OLLAMA_CONFIG", str(cfg))
    s = load_settings()
    client = _build_client(s, fake_ollama=_FakeOllamaClient())
    with client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        assert "m1@u" in ids


def test_expose_external_hides_upstream_models(reverse_settings):
    """Removing a composite id from external_exposed_models hides it from /v1/*."""
    data = reverse_settings.model_dump()
    # Keep the upstream model in internal but hide it from external.
    data["external_exposed_models"] = ["llama3.1@local"]
    s = Settings(**data)
    client = _build_client(s, fake_ollama=_FakeOllamaClient())
    with client:
        r = client.get("/v1/models", headers=_AUTH)
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        assert "llama3.1@local" in ids
        assert "claude-3-5-sonnet-20241022@default" not in ids


def test_expose_external_blocks_passthrough(reverse_settings):
    """A non-exposed upstream model gets 404 on /v1/messages passthrough."""
    data = reverse_settings.model_dump()
    data["external_exposed_models"] = ["llama3.1@local"]
    s = Settings(**data)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be reached for hidden model")

    client = _build_client(s, upstream_transport=httpx.MockTransport(handler))
    with client:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "claude-3-5-sonnet-20241022@default",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 404


def test_expose_external_explicit_subset(reverse_settings):
    """Only composite ids in external_exposed_models are visible."""
    data = reverse_settings.model_dump()
    data["upstreams"][0]["models"] = ["public-m", "private-m"]
    data["external_exposed_models"] = ["public-m@default"]
    data["internal_exposed_models"] = [
        "public-m@default", "private-m@default", "llama3.1@local",
    ]
    s = Settings(**data)
    client = _build_client(s, fake_ollama=_FakeOllamaClient())
    with client:
        r = client.get("/v1/models", headers=_AUTH)
        ids = [m["id"] for m in r.json()["data"]]
        assert "public-m@default" in ids
        assert "private-m@default" not in ids
