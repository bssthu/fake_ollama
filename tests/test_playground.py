"""Tests for the lightweight model playground listener."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.server import create_app


def _settings() -> Settings:
    return Settings(
        anthropic_upstreams=[
            {
                "name": "remote",
                "base_url": "http://upstream.test",
                "auth_token": "upstream-token",
                "models": [{"name": "model-a"}, {"name": "model-b"}],
            }
        ],
        ollama_interfaces=[],
        api_interfaces=[
            {
                "name": "first",
                "host": "127.0.0.1",
                "port": 21435,
                "access_tokens": ["key-a"],
                "exposed_models": [
                    {"model": "model-a", "target": "remote", "alias": "alpha"},
                ],
            },
            {
                "name": "second",
                "host": "127.0.0.1",
                "port": 21436,
                "access_tokens": ["key-b"],
                "exposed_models": [
                    {"model": "model-b", "target": "remote", "alias": "beta"},
                ],
            },
        ],
        playground_enabled=True,
        playground_host="127.0.0.1",
        playground_port=21431,
    )


def test_playground_static_page_and_security_headers():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21431") as client:
        response = client.get("/playground/")
        css = client.get("/playground/playground.css")
        js = client.get("/playground/playground.js")

    assert response.status_code == 200
    assert "Model Playground" in response.text
    assert "轻量、多能力、即时调试" in response.text
    assert 'id="apiKey"' in response.text
    assert 'id="model"' in response.text
    assert 'id="fileInput"' in response.text
    assert 'id="operation"' in response.text
    assert 'id="operationPreset"' in response.text
    assert 'id="operationParameterList"' in response.text
    assert 'id="cameraSection"' in response.text
    assert 'id="cameraPreview"' in response.text
    assert 'id="cameraStart"' in response.text
    assert 'id="cameraStop"' in response.text
    assert 'id="conversation"' in response.text
    assert 'id="modelMemory"' in response.text
    assert 'id="contextChip"' in response.text
    assert 'id="historyModeChip"' in response.text
    assert 'id="contextMetric"' in response.text
    assert 'id="contextNotice"' in response.text
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200
    assert "clipboardData" in js.text
    assert "image_generation" in js.text
    assert "video_generation" in js.text
    assert "video_understanding" in js.text
    assert "video_analysis" in js.text
    assert "video_url" in js.text
    assert "navigator.mediaDevices.getUserMedia" in js.text
    assert "new window.MediaRecorder" in js.text
    assert "video_duration_seconds" in js.text
    assert "max_pending_segments: 1" in js.text
    assert "new FormData()" in js.text
    assert "stream: true" in js.text
    assert "estimated_vram_gb" in js.text
    assert "estimated_memory_gb" in js.text
    assert "fetch('/playground/api/models'" in js.text
    assert "DISCOVERY_SCHEMA_VERSION = 1" in js.text
    assert "renderOperationParameters" in js.text
    assert "state.parameterInputs" in js.text
    assert "state.interactionHistories" in js.text
    assert "operationUsesHistory" in js.text
    assert "renderInteractionHistory" in js.text
    assert "prepareChatRequest" in js.text
    assert "prepareMediaRequest" in js.text
    assert "historyMessages" in js.text
    assert "CONTEXT_THRESHOLD_RATIO" in js.text
    assert "? Math.floor(configured)" in js.text
    assert "Math.min(requested" not in js.text
    assert "body.max_tokens = plan.outputReserve" in js.text
    assert "interaction.turns.push" in js.text
    assert "仅发送本次输入" in js.text
    assert "clearComposerInput" in js.text
    assert "Enter 执行 · Ctrl / ⌘ + Enter 换行" in response.text
    assert "Enter 发送 · Ctrl / ⌘ + Enter 换行" in js.text
    assert "els.prompt.setRangeText('\\n'" in js.text
    assert "shortcutSubmit" not in js.text

    chat_source = js.text.split("async function runChat", 1)[1].split(
        "function readParameterValue", 1
    )[0]
    assert "clearComposerInput();" not in chat_source

    media_source = js.text.split("async function runMedia", 1)[1].split(
        "async function runRequest", 1
    )[0]
    assert "clearComposerInput();" not in media_source
    request_source = js.text.split("async function runRequest", 1)[1].split(
        "els.toggleKey.addEventListener", 1
    )[0]
    assert request_source.count("clearComposerInput();") == 1
    assert request_source.index("beginRequest(op, plan);") < request_source.index(
        "clearComposerInput();"
    )
    assert request_source.index("clearComposerInput();") < request_source.index(
        "if (operationUsesChatEndpoint(op)) await runChat(op, plan);"
    )
    assert "showRequestError" in js.text
    assert response.headers["cache-control"] == "no-store"
    assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert "'unsafe-inline'" not in response.headers["content-security-policy"]


def test_playground_port_only_exposes_playground_and_model_surfaces():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21431") as client:
        root = client.get("/", follow_redirects=False)
        admin = client.get("/admin/")
        dashboard = client.get("/dashboard/")
        version = client.get("/api/version")

    assert root.status_code in (307, 308)
    assert root.headers["location"] == "/playground/"
    assert admin.status_code == 404
    assert dashboard.status_code == 404
    assert version.status_code == 404


def test_playground_api_key_selects_the_matching_interface_models():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21431") as client:
        missing = client.get("/playground/api/models")
        first = client.get(
            "/playground/api/models",
            headers={"Authorization": "Bearer key-a"},
        )
        second = client.get(
            "/playground/api/models", headers={"x-api-key": "key-b"}
        )
        openai = client.get("/v1/models", headers={"x-api-key": "key-a"})

    assert missing.status_code == 401
    assert first.headers["cache-control"] == "no-store"
    assert first.json()["schema_version"] == 1
    assert [item["id"] for item in first.json()["models"]] == ["alpha"]
    assert [item["id"] for item in second.json()["models"]] == ["beta"]
    alpha = first.json()["models"][0]
    assert alpha["capabilities"] == ["completion", "tools", "vision"]
    assert alpha["context_length"] > 0
    assert "max_output_tokens" in alpha
    assert "estimated_memory_gb" in alpha
    assert alpha["operations"] == [
        {
            "id": "chat",
            "endpoint": "/v1/chat/completions",
            "stream": True,
            "history_mode": "conversation",
            "accepts_images": True,
            "tool_calling": True,
        }
    ]
    assert set(openai.json()["data"][0]) == {
        "id",
        "object",
        "created",
        "owned_by",
    }


def test_playground_route_is_not_available_on_other_ports():
    app = create_app(_settings())
    with TestClient(app, base_url="http://testserver:21435") as api_client:
        assert api_client.get("/playground/").status_code == 404
        assert api_client.get(
            "/playground/api/models", headers={"x-api-key": "key-a"}
        ).status_code == 404


def test_video_understanding_has_single_turn_upload_operation():
    settings = _settings()
    settings = settings.model_copy(
        update={
            "model_profiles": {
                "model-a@remote": {
                    "capabilities": ["video_understanding"],
                    "context_length": 32768,
                    "max_output_tokens": 512,
                }
            }
        }
    )
    app = create_app(settings)
    with TestClient(app, base_url="http://testserver:21431") as client:
        response = client.get(
            "/playground/api/models", headers={"Authorization": "Bearer key-a"}
        )

    assert response.status_code == 200
    alpha = response.json()["models"][0]
    assert alpha["capabilities"] == ["video_understanding"]
    operation = alpha["operations"][0]
    assert operation["id"] == "video_analysis"
    assert operation["endpoint"] == "/v1/chat/completions"
    assert operation["stream"] is True
    assert operation["history_mode"] == "single_turn"
    assert operation["accepts_videos"] is True
    assert operation["requires_videos"] is True
    assert operation["limits"]["max_videos"] == 1
    assert operation["live_camera"] == {
        "supported": True,
        "capture_mode": "windowed_media_recorder",
        "max_pending_segments": 1,
    }
    assert next(
        item for item in operation["parameters"] if item["name"] == "max_segments"
    )["max"] == 120
    assert {item["name"] for item in operation["parameters"]} == {
        "segment_seconds",
        "frames_per_segment",
        "max_segments",
        "include_summary",
    }


def test_disabled_playground_does_not_register_static_route():
    settings = _settings().model_copy(update={"playground_enabled": False})
    app = create_app(settings)
    with TestClient(app, base_url="http://testserver:21431") as client:
        assert client.get("/playground/").status_code == 404
        assert client.get(
            "/playground/api/models", headers={"x-api-key": "key-a"}
        ).status_code == 404
