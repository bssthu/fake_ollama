from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_ollama.comfyui_client import ComfyUIClient, ComfyUIImage
from fake_ollama.config import Settings
from fake_ollama.server import create_app


def _settings() -> Settings:
    return Settings(
        comfyui_targets=[
            {
                "name": "z-image-comfy",
                "base_url": "http://127.0.0.1:21480",
                "model": "z-image-turbo",
                "auto_start": False,
                "default_width": 1024,
                "default_height": 1024,
                "default_steps": 8,
                "default_edit_denoise": 0.25,
                "max_batch_size": 2,
            }
        ],
        api_interfaces=[
            {
                "name": "api",
                "host": "127.0.0.1",
                "port": 21435,
                "access_tokens": ["tk"],
                "exposed_models": [
                    {
                        "model": "z-image-turbo",
                        "target": "z-image-comfy",
                        "alias": "z-image-turbo",
                    }
                ],
            }
        ],
        model_profiles=[
            {
                "model": "z-image-turbo",
                "target": "z-image-comfy",
                "capabilities": ["image_generation", "image_edit"],
                "context_length": 4096,
                "estimated_vram_gb": 16.0,
            }
        ],
    )


class _FakeComfyClient:
    idle_timeout_seconds = None

    def __init__(self) -> None:
        self.generate_calls: List[Dict[str, Any]] = []
        self.edit_calls: List[Dict[str, Any]] = []

    async def generate_image(self, **kwargs: Any) -> List[ComfyUIImage]:
        self.generate_calls.append(kwargs)
        return [
            ComfyUIImage(
                data=b"png-a",
                filename="a.png",
                subfolder="",
                image_type="output",
                mime_type="image/png",
            )
        ]

    async def edit_image(self, **kwargs: Any) -> List[ComfyUIImage]:
        self.edit_calls.append(kwargs)
        return [
            ComfyUIImage(
                data=b"png-edit",
                filename="edit.png",
                subfolder="",
                image_type="output",
                mime_type="image/png",
            )
        ]


def _client_with_fake(fake: _FakeComfyClient) -> TestClient:
    settings = _settings()
    app = create_app(settings)
    app.state.comfyui_clients = {"z-image-comfy": fake}
    return TestClient(app, base_url="http://testserver:21435")


def test_openai_image_generations_routes_to_comfyui_default_model() -> None:
    fake = _FakeComfyClient()
    client = _client_with_fake(fake)
    with client:
        resp = client.post(
            "/v1/images/generations",
            headers={"x-api-key": "tk"},
            json={
                "prompt": "a quiet product photo",
                "size": "768x512",
                "n": 2,
                "seed": 123,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["data"][0]["b64_json"] == "cG5nLWE="
    assert fake.generate_calls[0]["model"] == "z-image-turbo"
    assert fake.generate_calls[0]["prompt"] == "a quiet product photo"
    assert fake.generate_calls[0]["width"] == 768
    assert fake.generate_calls[0]["height"] == 512
    assert fake.generate_calls[0]["n"] == 2
    assert fake.generate_calls[0]["estimated_vram_gb"] == 16.0


def test_openai_models_uses_comfyui_target_profile() -> None:
    fake = _FakeComfyClient()
    client = _client_with_fake(fake)
    with client:
        resp = client.get("/v1/models", headers={"x-api-key": "tk"})

    assert resp.status_code == 200
    model = next(item for item in resp.json()["data"] if item["id"] == "z-image-turbo")
    assert model["context_length"] == 4096
    assert model["capabilities"] == ["image_generation", "image_edit"]


def test_openai_image_edits_accepts_multipart_image() -> None:
    fake = _FakeComfyClient()
    client = _client_with_fake(fake)
    with client:
        resp = client.post(
            "/v1/images/edits",
            headers={"x-api-key": "tk"},
            data={
                "model": "z-image-turbo",
                "prompt": "make it warmer",
                "size": "512x512",
            },
            files={"image": ("input.png", b"input-bytes", "image/png")},
        )

    assert resp.status_code == 200
    assert resp.json()["data"][0]["b64_json"] == "cG5nLWVkaXQ="
    assert fake.edit_calls[0]["prompt"] == "make it warmer"
    assert fake.edit_calls[0]["image_bytes"] == b"input-bytes"
    assert fake.edit_calls[0]["filename"] == "input.png"
    assert fake.edit_calls[0]["denoise"] == 0.25


def test_openai_image_edits_accepts_bracketed_image_field() -> None:
    # OpenAI's images/edits multipart convention (and the AI SDK's
    # OpenAICompatibleImageModel) names the input image field "image[]".
    fake = _FakeComfyClient()
    client = _client_with_fake(fake)
    with client:
        resp = client.post(
            "/v1/images/edits",
            headers={"x-api-key": "tk"},
            data={
                "model": "z-image-turbo",
                "prompt": "make it night",
                "size": "1024x1024",
            },
            files={"image[]": ("input.png", b"input-bytes", "image/png")},
        )

    assert resp.status_code == 200
    assert fake.edit_calls[0]["prompt"] == "make it night"
    assert fake.edit_calls[0]["image_bytes"] == b"input-bytes"
    assert fake.edit_calls[0]["filename"] == "input.png"


def _client_with_seed_mode(fake: _FakeComfyClient, *, seed_mode: str, seed: int) -> TestClient:
    settings = _settings()
    settings.comfyui_targets[0].seed_mode = seed_mode
    settings.comfyui_targets[0].seed = seed
    app = create_app(settings)
    app.state.comfyui_clients = {"z-image-comfy": fake}
    return TestClient(app, base_url="http://testserver:21435")


def _gen(client: TestClient, **extra: Any) -> None:
    body = {"prompt": "a cube", "size": "512x512"}
    body.update(extra)
    resp = client.post("/v1/images/generations", headers={"x-api-key": "tk"}, json=body)
    assert resp.status_code == 200, resp.text


def test_seed_mode_fixed_reuses_configured_seed() -> None:
    fake = _FakeComfyClient()
    with _client_with_seed_mode(fake, seed_mode="fixed", seed=777) as client:
        _gen(client)
        _gen(client)
    assert fake.generate_calls[0]["seed"] == 777
    assert fake.generate_calls[1]["seed"] == 777


def test_seed_mode_increment_advances_per_request() -> None:
    fake = _FakeComfyClient()
    with _client_with_seed_mode(fake, seed_mode="increment", seed=100) as client:
        _gen(client)            # n defaults to 1 -> 100
        _gen(client)            # -> 101
        _gen(client, n=2)       # -> 102, then counter jumps by 2
        _gen(client)            # -> 104
    assert [c["seed"] for c in fake.generate_calls] == [100, 101, 102, 104]


def test_seed_mode_random_differs_and_request_seed_overrides() -> None:
    fake = _FakeComfyClient()
    with _client_with_seed_mode(fake, seed_mode="random", seed=0) as client:
        _gen(client)
        _gen(client)
        _gen(client, seed=42)   # explicit request seed wins even in random mode
    assert fake.generate_calls[2]["seed"] == 42
    # random mode should not collapse to a constant
    assert fake.generate_calls[0]["seed"] != fake.generate_calls[1]["seed"]


def test_request_seed_overrides_fixed_mode() -> None:
    fake = _FakeComfyClient()
    with _client_with_seed_mode(fake, seed_mode="fixed", seed=5) as client:
        _gen(client, seed=999)
    assert fake.generate_calls[0]["seed"] == 999


@pytest.mark.asyncio
async def test_comfyui_client_runs_prompt_and_collects_view_image() -> None:
    seen_prompt: Optional[Dict[str, Any]] = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_prompt
        if request.method == "GET" and request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.method == "POST" and request.url.path == "/prompt":
            seen_prompt = request.read()
            return httpx.Response(200, json={"prompt_id": "pid-1"})
        if request.method == "GET" and request.url.path == "/history/pid-1":
            return httpx.Response(
                200,
                json={
                    "pid-1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "ComfyUI_00001_.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                        "status": {"status_str": "success"},
                    }
                },
            )
        if request.method == "GET" and request.url.path == "/view":
            return httpx.Response(200, content=b"image-bytes")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = ComfyUIClient(
        "http://comfy.test",
        client=httpx.AsyncClient(transport=transport),
        workflow_config={"poll_interval_seconds": 0.01},
    )

    try:
        images = await client.generate_image(
            model="z-image-turbo",
            prompt="hello",
            width=512,
            height=512,
            n=1,
            seed=7,
            steps=8,
            cfg=1.0,
            sampler_name="res_multistep",
            scheduler="simple",
            denoise=1.0,
        )
    finally:
        await client.aclose()

    assert images[0].data == b"image-bytes"
    assert seen_prompt is not None
