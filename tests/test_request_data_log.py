from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi.testclient import TestClient

from fake_ollama.anthropic_client import AnthropicClient
from fake_ollama.llama_cpp_client import LlamaCppClient
from fake_ollama.request_data_log import configure_request_data_logging
from fake_ollama.server import create_app


def _records(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_request_data_log_records_http_and_backend_payloads(settings, tmp_path):
    log_file = tmp_path / "fake_ollama.requests.jsonl"
    configure_request_data_logging(str(log_file))

    def handler(request: httpx.Request) -> httpx.Response:
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

    try:
        app = create_app(settings)
        transport = httpx.MockTransport(handler)
        app.state.clients = {
            up.name: AnthropicClient(
                up.base_url, up.auth_token,
                client=httpx.AsyncClient(transport=transport),
            )
            for up in settings.anthropic_upstreams
        }
        with TestClient(app) as client:
            resp = client.post(
                "/api/chat",
                json={
                    "model": "claude-3-5-sonnet-20241022@default",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                },
            )
    finally:
        configure_request_data_logging(None)

    assert resp.status_code == 200
    records = _records(log_file)
    events = [record["event"] for record in records]
    assert "http_request_start" in events
    assert "http_request_body" in events
    assert "backend_request" in events
    assert "backend_response_body" in events
    assert "http_response_body" in events
    assert "http_request_end" in events

    request_ids = {record.get("request_id") for record in records}
    assert len(request_ids) == 1

    http_body = next(record for record in records if record["event"] == "http_request_body")
    assert "ping" in http_body["body"]["text"]

    backend_request = next(
        record
        for record in records
        if record["event"] == "backend_request" and record["backend"] == "anthropic"
    )
    assert backend_request["body"]["json"]["messages"][0]["content"] == "ping"
    assert backend_request["headers"]["x-api-key"].startswith("<redacted sha256:")

    response_body = next(
        record for record in records if record["event"] == "http_response_body"
    )
    assert "pong" in response_body["body"]["text"]


async def test_request_data_log_treats_close_after_done_as_complete(tmp_path):
    log_file = tmp_path / "fake_ollama.requests.jsonl"
    configure_request_data_logging(str(log_file))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    client = LlamaCppClient(
        "http://llama.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        stream = client.stream_chat({"model": "m", "messages": []})
        assert await anext(stream) == "data: [DONE]"
        await stream.aclose()
    finally:
        await client.aclose()
        configure_request_data_logging(None)

    records = _records(log_file)
    assert not any(record["event"] == "backend_error" for record in records)
    response_end = next(
        record for record in records if record["event"] == "backend_response_end"
    )
    assert response_end["outcome"] == "complete"
    assert response_end["error"] is None
