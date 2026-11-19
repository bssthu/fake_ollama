import json

import httpx
from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.generic_openai_client import GenericOpenAIClient
from fake_ollama.server import create_app


def test_motion_listener_routing_and_native_payload():
    exposure = {"target": "local", "model": "soma", "alias": "human"}
    settings = Settings(
        generic_openai_targets=[{"name": "local", "models": [{"name": "soma", "upstream_id": "native"}]}],
        model_profiles=[{"model": "soma", "target": "local", "capabilities": ["motion_generation"]}],
        api_interfaces=[{"name": "api", "port": 21435, "access_tokens": ["key"], "exposed_models": [exposure]}],
        ollama_interfaces=[{"name": "ollama", "port": 21434, "exposed_models": [exposure]}],
        playground_enabled=True, playground_port=21431,
    )
    forwarded = []

    def upstream(request):
        if request.method == "POST":
            forwarded.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"model": "native", "num_frames": 26})

    app = create_app(settings)
    app.state.generic_openai_clients = {"local": GenericOpenAIClient(
        base_url="http://native", client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    )}
    path = "/v1/motions/generations"
    payload = {"model": "human", "example": "walk"}
    headers = {"Authorization": "Bearer key"}
    with TestClient(app, base_url="http://127.0.0.1:21435") as client:
        assert client.post(path, json=payload).status_code == 401
        assert client.post(path, headers=headers, json={**payload, "model": "hidden"}).status_code == 404
        assert client.post(path, headers=headers, json={**payload, "target_poses": []}).status_code == 400
        assert client.post("http://127.0.0.1:21434" + path, json=payload).status_code == 404
        assert client.post("/v1/chat/completions", headers=headers, json={
            "model": "human", "messages": [{"role": "user", "content": "walk"}],
        }).status_code == 400
        entry = client.get("http://127.0.0.1:21431/playground/api/models", headers=headers).json()["models"][0]
        assert [op["id"] for op in entry["operations"]] == ["motion_generation"]
        result = client.post(path, headers=headers, json=payload)
        assert result.status_code == 200, result.text
        assert result.json() == {"model": "human", "num_frames": 26}
        assert forwarded == [(path, {"model": "native", "example": "walk"})]
