"""Playground decision capability metadata must follow the configured target."""

from fake_ollama.config import Settings
from fake_ollama.playground_support import _playground_model_entry


def _settings(**target):
    return Settings(
        decision_targets=[{
            "name": "decision-service",
            "base_url": "http://127.0.0.1:29200",
            "models": [{"name": "fixture-model"}],
            **target,
        }],
        ollama_interfaces=[],
        api_interfaces=[{
            "name": "fixture-api", "port": 29201,
            "access_tokens": ["fixture-key"],
            "exposed_models": [{"target": "decision-service", "model": "fixture-model", "alias": "public-decision"}],
        }],
    )


def test_playground_decision_discovery_exposes_distinct_dialects_without_chat():
    settings = _settings()
    entry = _playground_model_entry(settings, settings.api_interfaces[0], "public-decision")
    assert entry["capabilities"] == ["decision"]
    assert len(entry["operations"]) == 1
    operation = entry["operations"][0]
    assert operation["id"] == "decision"
    assert operation["endpoint"] == "/v1/systemone"
    assert operation["history_mode"] == "single_turn"
    assert operation["stream"] is False
    assert operation["question_types"] == ["predicate", "choice", "score"]
    assert operation["accepts_images"] is False
    assert operation["extensions"] == []
    assert operation["think_modes"] == []
    assert [(item["id"], item["endpoint"]) for item in operation["dialects"]] == [
        ("systemone", "/v1/systemone"), ("openai_decisions", "/v1/decisions"),
    ]
    assert "base_url" not in operation
    assert "auth_token" not in operation


def test_playground_decision_discovery_uses_declared_images_and_extensions():
    settings = _settings(
        runtime="startlux", endpoint_path="/v1/decisions",
        decision_capabilities={
            "question_types": ["choice", "score"], "modalities": ["text", "image"],
            "extensions": ["think_mode"], "think_modes": ["never", "auto", "always"],
        },
    )
    operation = _playground_model_entry(settings, settings.api_interfaces[0], "public-decision")["operations"][0]
    assert operation["runtime"] == "startlux"
    assert operation["accepts_images"] is True
    assert operation["image_dialect"] == "systemone"
    assert operation["image_format"] == "startlux_images"
    # The configured backend path does not change the public wire format.
    assert operation["endpoint"] == "/v1/systemone"
    assert operation["think_modes"] == ["never", "auto", "always"]
    assert operation["question_types"] == ["choice", "score"]


def test_playground_openai_native_image_metadata_does_not_infer_startlux_controls():
    settings = _settings(
        dialect="openai_decisions",
        decision_capabilities={"question_types": ["choice"], "modalities": ["text", "image"]},
    )
    operation = _playground_model_entry(settings, settings.api_interfaces[0], "public-decision")["operations"][0]
    assert operation["endpoint"] == "/v1/decisions"
    assert operation["image_dialect"] == "openai_decisions"
    assert operation["extensions"] == []
    assert operation["think_modes"] == []


def test_playground_decision_discovery_merges_cached_restrictions_without_expansion():
    settings = _settings(
        runtime="startlux",
        decision_capabilities={
            "question_types": ["choice"], "modalities": ["text", "image"],
            "extensions": ["think_mode"], "think_modes": ["never", "auto", "always"],
        },
    )

    class CachedClient:
        def capability_snapshot(self):
            return {
                "question_types": ["choice", "predicate"], "modalities": ["text"],
                "extensions": ["think_mode", "unknown"], "think_modes": ["never"],
                "version": "fixture-version", "endpoint_verified": True,
                "warnings": ["Fixture advisory"], "status": "supported",
            }

        async def probe_capabilities(self):
            raise AssertionError("Discovery must not probe a backend")

    operation = _playground_model_entry(
        settings, settings.api_interfaces[0], "public-decision",
        {"decision-service": CachedClient()},
    )["operations"][0]
    assert operation["question_types"] == ["choice"]
    assert operation["extensions"] == ["think_mode"]
    assert operation["think_modes"] == ["never"]
    assert operation["accepts_images"] is False
    assert operation["version"] == "fixture-version"
    assert operation["endpoint_verified"] is True
    assert operation["warnings"] == ["Fixture advisory"]


def test_playground_models_endpoint_uses_cached_snapshot_without_probe():
    from fastapi.testclient import TestClient
    from fake_ollama.server import create_app

    settings = _settings(
        decision_capabilities={"question_types": ["choice"], "modalities": ["text", "image"]},
    ).model_copy(update={"playground_enabled": True})
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:21431", client=("127.0.0.1", 50000)) as client:
        backend = app.state.decision_clients["decision-service"]
        snapshot = backend.capability_snapshot()
        snapshot.update(modalities=["text"], version="cached-fixture", endpoint_verified=True)
        backend.capability_snapshot = lambda: snapshot

        async def unexpected_probe(*args, **kwargs):
            raise AssertionError("Playground discovery must not contact the provider")

        backend.probe_capabilities = unexpected_probe
        response = client.get("/playground/api/models", headers={"Authorization": "Bearer fixture-key"})
    assert response.status_code == 200
    models = response.json()["models"]
    assert [model["id"] for model in models] == ["public-decision"]
    operation = models[0]["operations"][0]
    assert operation["accepts_images"] is False
    assert operation["version"] == "cached-fixture"
    assert operation["endpoint_verified"] is True
