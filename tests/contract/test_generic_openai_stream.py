"""SSE comments from local adapters must never become completion data."""

import json

from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.server import create_app


def test_generic_openai_preserves_heartbeat_comments_and_completion_frames():
    settings = Settings(
        generic_openai_targets=[{
            "name": "adapter", "base_url": "http://adapter.test",
            "models": [{"name": "local", "upstream_id": "wire-model"}],
        }],
        api_interfaces=[{
            "name": "api", "port": 21435,
            "exposed_models": [{"model": "local", "target": "adapter", "alias": "local"}],
        }],
        ollama_interfaces=[],
        model_profiles={"local": {"estimated_vram_gb": 21, "estimated_memory_gb": 64}},
        dashboard_enabled=False,
    )
    captured = {}

    class Adapter:
        async def stream_chat(self, payload, **kwargs):
            captured.update(payload=payload, resources=kwargs)
            yield ": connected"
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield ": keep-alive"
            yield '{"choices":[{"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

        async def aclose(self):
            pass

    app = create_app(settings, test_listener_fallback=True)
    with TestClient(app, base_url="http://127.0.0.1:21435") as client:
        app.state.generic_openai_clients["adapter"] = Adapter()
        response = client.post("/v1/chat/completions", json={
            "model": "local", "messages": [{"role": "user", "content": "Hi"}], "stream": True,
        })
        assert response.status_code == 200
        frames = response.text.strip().split("\n\n")
        assert frames[0] == ": connected"
        assert ": keep-alive" in frames
        data = [frame[5:].strip() for frame in frames if frame.startswith("data:")]
        assert data[-1] == "[DONE]"
        chunks = [json.loads(value) for value in data[:-1]]
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert captured["payload"]["model"] == "wire-model"
        assert captured["resources"] == {"estimated_vram_gb": 21, "estimated_memory_gb": 64}
