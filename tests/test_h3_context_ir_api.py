"""API, provider switching, Playground, and video orchestration tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from fake_ollama.comfyui_client import ComfyUIImage
from fake_ollama.config import Settings
from fake_ollama.server import create_app


def _planner_content(*, mode: str, duration: float) -> str:
    return json.dumps(
        {
            "mode": mode,
            "duration_seconds": duration,
            "shots": [
                {
                    "start_seconds": 0,
                    "description": (
                        "Cinematic medium-wide shot of a runner crossing a wet street "
                        "as the camera tracks alongside at slow speed."
                    ),
                }
            ],
            "overall_soundscape": "Rain taps on pavement under measured footsteps.",
            "non_diegetic_music": "Sparse low piano notes at a slow tempo.",
        }
    )


class _FakeOpenAIPlanner:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = list(responses or [])

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if self.responses:
            content = self.responses.pop(0)
        else:
            user_text = str(payload["messages"][1]["content"])
            mode = "i2va" if "mode: i2va" in user_text else "t2va"
            duration = 5.0
            if "duration_seconds: 5.04" in user_text:
                duration = 5.04
            content = _planner_content(mode=mode, duration=duration)
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
        }


class _FakeComfy:
    def __init__(self) -> None:
        self.video_calls: list[dict[str, Any]] = []

    async def generate_video(self, **kwargs: Any) -> list[ComfyUIImage]:
        self.video_calls.append(kwargs)
        return [
            ComfyUIImage(
                data=b"video",
                filename="result.mp4",
                subfolder="",
                image_type="output",
                mime_type="video/mp4",
            )
        ]


def _settings(*, attach_video: bool = False) -> Settings:
    comfy = []
    exposed = []
    profiles: dict[str, dict[str, Any]] = {
        "planner-text@planner": {
            "capabilities": ["completion"],
            "context_length": 8192,
            "estimated_vram_gb": 4,
            "estimated_memory_gb": 6,
        },
        "planner-vision@planner": {
            "capabilities": ["completion", "vision"],
            "context_length": 16384,
            "estimated_vram_gb": 9,
            "estimated_memory_gb": 12,
        },
    }
    if attach_video:
        comfy = [
            {
                "name": "h3-comfy",
                "base_url": "http://comfy.test",
                "model": "h3-local",
                "preset": "joyai_echo",
                "context_ir_profile": "default",
                "context_ir_prompt_mode": "auto",
            }
        ]
        exposed = [{"model": "h3-local", "target": "h3-comfy", "alias": "h3"}]
        profiles["h3-local@h3-comfy"] = {
            "capabilities": ["video_generation"],
            "estimated_vram_gb": 18,
        }
    return Settings(
        openai_upstreams=[
            {
                "name": "planner",
                "base_url": "http://planner.test",
                "models": [
                    {"name": "planner-text"},
                    {"name": "planner-vision"},
                ],
            }
        ],
        comfyui_targets=comfy,
        h3_context_ir_profiles=[
            {
                "name": "default",
                "providers": [
                    {
                        "name": "text-api",
                        "model": "planner-text",
                        "target": "planner",
                        "modalities": ["text"],
                        "json_mode": True,
                    },
                    {
                        "name": "vision-api",
                        "model": "planner-vision",
                        "target": "planner",
                        "modalities": ["text", "image"],
                    },
                ],
                "default_text_provider": "text-api",
                "default_multimodal_provider": "vision-api",
                "max_attempts": 2,
            }
        ],
        model_profiles=profiles,
        ollama_interfaces=[],
        api_interfaces=[
            {
                "name": "api",
                "host": "127.0.0.1",
                "port": 21435,
                "access_tokens": ["tk"],
                "exposed_models": exposed,
            }
        ],
        playground_enabled=True,
        playground_port=21431,
        admin_enabled=False,
        dashboard_enabled=False,
    )


def _settings_with_external(*, attach_video: bool = False) -> Settings:
    data = _settings(attach_video=attach_video).model_dump()
    data["h3_context_ir_profiles"][0]["allow_external_api"] = True
    return Settings.model_validate(data)


def test_playground_discovers_and_runs_context_ir_virtual_model() -> None:
    planner = _FakeOpenAIPlanner()
    app = create_app(_settings())
    app.state.openai_clients = {"planner": planner}
    with TestClient(app, base_url="http://testserver:21431") as client:
        discovery = client.get("/playground/api/models", headers={"x-api-key": "tk"})
        response = client.post(
            "/v1/videos/context-ir",
            headers={"x-api-key": "tk"},
            json={
                "model": "h3-context-ir-fake@default",
                "prompt": "A runner crosses a rainy street.",
                "mode": "t2va",
                "duration_seconds": 5,
            },
        )

    assert discovery.status_code == 200
    virtual = discovery.json()["models"][0]
    assert virtual["id"] == "h3-context-ir-fake@default"
    assert virtual["capabilities"] == ["h3_context_ir"]
    operation = virtual["operations"][0]
    assert operation["endpoint"] == "/v1/videos/context-ir"
    assert {item["name"] for item in operation["parameters"]} == {
        "provider",
        "mode",
        "duration_seconds",
    }

    assert response.status_code == 200
    body = response.json()
    assert body["provider"]["name"] == "text-api"
    assert body["fallback"] is False
    assert body["content"]["prompt"].startswith(
        "integrated_multimodal_description: [Shot 1]"
    )
    assert planner.calls[0]["model"] == "planner-text"
    assert planner.calls[0]["response_format"] == {"type": "json_object"}


def test_image_request_auto_selects_multimodal_provider() -> None:
    planner = _FakeOpenAIPlanner()
    app = create_app(_settings())
    app.state.openai_clients = {"planner": planner}
    with TestClient(app, base_url="http://testserver:21431") as client:
        response = client.post(
            "/v1/videos/context-ir",
            headers={"x-api-key": "tk"},
            data={
                "model": "h3-context-ir-fake@default",
                "prompt": "The subject turns toward the window.",
                "mode": "i2va",
                "duration_seconds": "5",
                "provider": "auto",
            },
            files=[("image[]", ("reference.png", b"png-bytes", "image/png"))],
        )

    assert response.status_code == 200
    assert response.json()["provider"]["name"] == "vision-api"
    call = planner.calls[0]
    assert call["model"] == "planner-vision"
    content = call["messages"][1]["content"]
    assert any(part.get("type") == "image_url" for part in content)


def test_discovery_adds_interface_exposed_compatible_models_with_resources() -> None:
    data = _settings().model_dump()
    data["openai_upstreams"][0]["models"].extend(
        [{"name": "planner-other"}, {"name": "planner-media"}]
    )
    data["model_profiles"].append({
        "model": "planner-other",
        "target": "planner",
        "capabilities": ["completion"],
        "context_length": 32768,
        "estimated_vram_gb": 7,
        "estimated_memory_gb": 10,
    })
    data["model_profiles"].append({
        "model": "planner-media",
        "target": "planner",
        "capabilities": ["image_generation"],
    })
    data["api_interfaces"][0]["exposed_models"] = [
        {"model": "planner-text", "target": "planner"},
        {
            "model": "planner-other",
            "target": "planner",
            "alias": "other-planner",
        },
        {"model": "planner-media", "target": "planner"},
    ]
    settings = Settings.model_validate(data)
    app = create_app(settings)
    with TestClient(app, base_url="http://testserver:21431") as client:
        discovery = client.get(
            "/playground/api/models", headers={"x-api-key": "tk"}
        )

    assert discovery.status_code == 200
    virtual = next(
        item
        for item in discovery.json()["models"]
        if item["id"] == "h3-context-ir-fake@default"
    )
    operation = virtual["operations"][0]
    choices = {
        item["value"]: item
        for item in next(
            item
            for item in operation["parameters"]
            if item["name"] == "provider"
        )["choices"]
    }
    assert choices["text-api"]["selection_kind"] == "recommended"
    assert choices["text-api"]["estimated_vram_gb"] == 4
    compatible = choices["model:other-planner"]
    assert compatible["selection_kind"] == "compatible"
    assert compatible["group"] == "自选兼容模型"
    assert compatible["modalities"] == ["text"]
    assert compatible["estimated_vram_gb"] == 7
    assert compatible["estimated_memory_gb"] == 10
    assert compatible["context_length"] == 32768
    assert "model:planner-text@planner" not in choices
    assert "model:planner-media@planner" not in choices


def test_discovery_advertises_request_scoped_external_planner_when_enabled() -> None:
    app = create_app(_settings_with_external())
    with TestClient(app, base_url="http://testserver:21431") as client:
        discovery = client.get(
            "/playground/api/models", headers={"x-api-key": "tk"}
        )

    virtual = discovery.json()["models"][0]
    operation = virtual["operations"][0]
    provider_parameter = next(
        item for item in operation["parameters"] if item["name"] == "provider"
    )
    external = next(
        item for item in provider_parameter["choices"] if item["value"] == "external"
    )
    assert external["group"] == "第三方 API"
    assert external["selection_kind"] == "external"
    assert external["backend_kind"] == "remote"
    assert operation["external_planner_api"] == {
        "models_endpoint": "/playground/api/external-models",
        "protocols": ["openai", "anthropic"],
    }


def test_external_planner_model_detection_normalizes_url_and_forwards_token() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"data": [{"id": "model-a"}, {"id": "model-b"}]},
        )

    external_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_settings_with_external())
    app.state.external_planner_http_client = external_http
    try:
        with TestClient(app, base_url="http://testserver:21431") as client:
            response = client.post(
                "/playground/api/external-models",
                headers={
                    "x-api-key": "tk",
                    "x-playground-upstream-key": "third-party-secret",
                },
                json={
                    "profile": "h3-context-ir-fake@default",
                    "protocol": "openai",
                    "base_url": "https://provider.test/api/v1",
                },
            )
    finally:
        asyncio.run(external_http.aclose())

    assert response.status_code == 200, response.text
    assert response.json() == {
        "protocol": "openai",
        "base_url": "https://provider.test/api",
        "models": ["model-a", "model-b"],
    }
    assert captured["url"] == "https://provider.test/api/v1/models"
    assert captured["headers"]["authorization"] == "Bearer third-party-secret"
    assert captured["headers"]["x-api-key"] == "third-party-secret"


def test_external_planner_detection_requires_profile_opt_in_and_token() -> None:
    disabled = create_app(_settings())
    with TestClient(disabled, base_url="http://testserver:21431") as client:
        response = client.post(
            "/playground/api/external-models",
            headers={
                "x-api-key": "tk",
                "x-playground-upstream-key": "secret",
            },
            json={
                "profile": "default",
                "protocol": "openai",
                "base_url": "https://provider.test",
            },
        )
    assert response.status_code == 403

    enabled = create_app(_settings_with_external())
    with TestClient(enabled, base_url="http://testserver:21431") as client:
        response = client.post(
            "/playground/api/external-models",
            headers={"x-api-key": "tk"},
            json={
                "profile": "default",
                "protocol": "openai",
                "base_url": "https://provider.test",
            },
        )
    assert response.status_code == 400
    assert "token is required" in response.text

    with TestClient(enabled, base_url="http://testserver:21435") as client:
        response = client.post(
            "/v1/videos/context-ir",
            headers={
                "x-api-key": "tk",
                "x-playground-upstream-key": "secret",
            },
            json={
                "model": "default",
                "prompt": "A quiet street.",
                "provider": "external",
                "external_api_protocol": "openai",
                "external_api_base_url": "https://provider.test",
                "external_api_model": "planner",
            },
        )
    assert response.status_code == 400
    assert "only available on the Playground listener" in response.text


def test_external_openai_planner_executes_with_selected_vision_model() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _planner_content(mode="i2va", duration=5.0),
                    }
                }],
                "usage": {"prompt_tokens": 120, "completion_tokens": 90},
            },
        )

    external_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_settings_with_external())
    app.state.external_planner_http_client = external_http
    try:
        with TestClient(app, base_url="http://testserver:21431") as client:
            response = client.post(
                "/v1/videos/context-ir",
                headers={
                    "x-api-key": "tk",
                    "x-playground-upstream-key": "external-key",
                },
                data={
                    "model": "h3-context-ir-fake@default",
                    "prompt": "The subject turns toward the window.",
                    "mode": "i2va",
                    "duration_seconds": "5",
                    "provider": "external",
                    "external_api_protocol": "openai",
                    "external_api_base_url": "https://gateway.test/v1",
                    "external_api_model": "vision-planner",
                    "external_api_modalities": "text,image",
                },
                files=[("image[]", ("reference.png", b"png-bytes", "image/png"))],
            )
    finally:
        asyncio.run(external_http.aclose())

    assert response.status_code == 200, response.text
    assert response.json()["provider"] == {
        "name": "external",
        "model": "vision-planner",
        "target": "external:openai",
        "modalities": ["text", "image"],
    }
    assert captured["url"] == "https://gateway.test/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer external-key"
    assert captured["body"]["model"] == "vision-planner"
    assert any(
        part.get("type") == "image_url"
        for part in captured["body"]["messages"][1]["content"]
    )


def test_external_anthropic_planner_executes_selected_model() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{
                    "type": "text",
                    "text": _planner_content(mode="t2va", duration=5.0),
                }],
                "usage": {"input_tokens": 100, "output_tokens": 70},
            },
        )

    external_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_settings_with_external())
    app.state.external_planner_http_client = external_http
    try:
        with TestClient(app, base_url="http://testserver:21431") as client:
            response = client.post(
                "/v1/videos/context-ir",
                headers={
                    "x-api-key": "tk",
                    "x-playground-upstream-key": "anthropic-key",
                },
                json={
                    "model": "default",
                    "prompt": "A runner crosses a rainy street.",
                    "provider": "external",
                    "external_api_protocol": "anthropic",
                    "external_api_base_url": "https://anthropic-gateway.test/v1/messages",
                    "external_api_model": "claude-planner",
                    "external_api_modalities": "text",
                },
            )
    finally:
        asyncio.run(external_http.aclose())

    assert response.status_code == 200, response.text
    assert response.json()["provider"]["target"] == "external:anthropic"
    assert captured["url"] == "https://anthropic-gateway.test/v1/messages"
    assert captured["headers"]["x-api-key"] == "anthropic-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-planner"
    assert captured["body"]["system"]


def test_compatible_text_planner_with_image_warns_and_does_not_receive_image() -> None:
    data = _settings().model_dump()
    data["openai_upstreams"][0]["models"].append({"name": "planner-other"})
    data["model_profiles"].append({
        "model": "planner-other",
        "target": "planner",
        "capabilities": ["completion"],
    })
    data["api_interfaces"][0]["exposed_models"] = [
        {
            "model": "planner-other",
            "target": "planner",
            "alias": "other-planner",
        }
    ]
    planner = _FakeOpenAIPlanner()
    app = create_app(Settings.model_validate(data))
    app.state.openai_clients = {"planner": planner}
    with TestClient(app, base_url="http://testserver:21431") as client:
        response = client.post(
            "/v1/videos/context-ir",
            headers={"x-api-key": "tk"},
            data={
                "model": "h3-context-ir-fake@default",
                "prompt": "The subject turns toward the window.",
                "mode": "i2va",
                "duration_seconds": "5",
                "provider": "model:other-planner",
            },
            files=[("image[]", ("reference.png", b"png-bytes", "image/png"))],
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"]["name"] == "model:other-planner"
    assert "text-only" in body["warnings"][0]
    assert planner.calls[0]["model"] == "planner-other"
    assert isinstance(planner.calls[0]["messages"][1]["content"], str)


def test_profile_can_disable_compatible_model_selection() -> None:
    data = _settings().model_dump()
    data["openai_upstreams"][0]["models"].append({"name": "planner-other"})
    data["model_profiles"].append({
        "model": "planner-other",
        "target": "planner",
        "capabilities": ["completion"],
    })
    data["api_interfaces"][0]["exposed_models"] = [
        {
            "model": "planner-other",
            "target": "planner",
            "alias": "other-planner",
        }
    ]
    data["h3_context_ir_profiles"][0]["allow_compatible_models"] = False
    app = create_app(Settings.model_validate(data))
    with TestClient(app, base_url="http://testserver:21431") as client:
        discovery = client.get(
            "/playground/api/models", headers={"x-api-key": "tk"}
        )
        response = client.post(
            "/v1/videos/context-ir",
            headers={"x-api-key": "tk"},
            json={
                "model": "h3-context-ir-fake@default",
                "prompt": "A quiet street.",
                "provider": "model:other-planner",
            },
        )

    virtual = next(
        item
        for item in discovery.json()["models"]
        if item["id"] == "h3-context-ir-fake@default"
    )
    provider_parameter = next(
        item
        for item in virtual["operations"][0]["parameters"]
        if item["name"] == "provider"
    )
    assert all(
        item["value"] != "model:other-planner"
        for item in provider_parameter["choices"]
    )
    assert response.status_code == 400
    assert "does not allow compatible model selection" in response.text


def test_invalid_provider_output_retries_then_falls_back_losslessly() -> None:
    planner = _FakeOpenAIPlanner(["not json", "still not json"])
    app = create_app(_settings())
    app.state.openai_clients = {"planner": planner}
    with TestClient(app, base_url="http://testserver:21431") as client:
        response = client.post(
            "/v1/videos/context-ir",
            headers={"x-api-key": "tk"},
            json={
                "model": "h3-context-ir-fake@default",
                "prompt": "保留这句原始需求。",
                "duration_seconds": 5,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is True
    assert body["attempts"] == 2
    assert "保留这句原始需求。" in body["content"]["prompt"]
    assert len(planner.calls) == 2


def test_video_generation_runs_context_ir_before_comfy_and_returns_revised_prompt() -> None:
    planner = _FakeOpenAIPlanner()
    comfy = _FakeComfy()
    app = create_app(_settings(attach_video=True))
    app.state.openai_clients = {"planner": planner}
    app.state.comfyui_clients = {"h3-comfy": comfy}
    with TestClient(app, base_url="http://testserver:21435") as client:
        response = client.post(
            "/v1/videos/generations",
            headers={"x-api-key": "tk"},
            json={
                "model": "h3",
                "prompt": "A runner crosses a rainy street.",
                "response_format": "b64_json",
            },
        )

    assert response.status_code == 200, response.text
    call = comfy.video_calls[0]
    assert call["prompt"].startswith("integrated_multimodal_description:")
    assert response.json()["data"][0]["revised_prompt"] == call["prompt"]
    assert len(planner.calls) == 1


def test_video_generation_can_use_request_scoped_external_planner() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _planner_content(mode="t2va", duration=5.04),
                    }
                }]
            },
        )

    external_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    comfy = _FakeComfy()
    app = create_app(_settings_with_external(attach_video=True))
    app.state.external_planner_http_client = external_http
    app.state.comfyui_clients = {"h3-comfy": comfy}
    try:
        with TestClient(app, base_url="http://testserver:21431") as client:
            response = client.post(
                "/v1/videos/generations",
                headers={
                    "x-api-key": "tk",
                    "x-playground-upstream-key": "external-key",
                },
                json={
                    "model": "h3",
                    "prompt": "A runner crosses a rainy street.",
                    "context_ir_provider": "external",
                    "external_api_protocol": "openai",
                    "external_api_base_url": "https://gateway.test/v1",
                    "external_api_model": "planner-model",
                    "external_api_modalities": "text",
                    "response_format": "b64_json",
                },
            )
    finally:
        asyncio.run(external_http.aclose())

    assert response.status_code == 200, response.text
    assert captured["body"]["model"] == "planner-model"
    assert comfy.video_calls[0]["prompt"].startswith(
        "integrated_multimodal_description:"
    )
