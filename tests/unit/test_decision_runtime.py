"""Offline runtime, capability and cancellation contracts for typed decisions."""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from fake_ollama.config import ApiInterface, DecisionTarget, Settings
from fake_ollama.decision_client import DecisionClient, build_native_argv
from fake_ollama.decisions import DecisionValidationError, NativeSystemOneCodec
from fake_ollama.vram import LocalTargetResourceError


def target(**changes):
    return DecisionTarget(name="decision", base_url="http://decision.test:8010", models=[{"name": "D3", "alias": "decide", "upstream_id": "wire-d3"}], **changes)


def request(**options):
    return NativeSystemOneCodec.parse_request({"model": "wire-d3", "state": "A test input", "questions": {
        "safe": {"type": "noul", "instructions": "Is the input safe?"}}, **options})


def response():
    return {"model": "wire-d3", "answers": {"safe": {"type": "noul", "noul": 0.73}}}


def test_config_profile_is_decision_only_and_model_alias_resolves():
    source = target()
    settings = Settings(decision_targets=[source], model_profiles={"decide@decision": {"estimated_vram_gb": 4}}, api_interfaces=[ApiInterface(name="api", exposed_models=[{"target": "decision", "model": "decide"}])])
    backend, real = settings.resolve_request("decide@decision", interface_name="api")
    assert backend.protocol == "decision"
    assert backend.resolve_model(real) == "wire-d3"
    assert settings.profile_for("decide@decision").capabilities == ["decision"]
    assert settings.profile_for("decide@decision").estimated_vram_gb == 4
    assert source.capability_metadata()["think_modes"] == []
    assert source.capability_metadata()["modalities"] == ["text"]


@pytest.mark.parametrize("changes,match", [
    ({"auto_start": True, "runtime": "vllm"}, "loopback|model_path"),
    ({"runtime": "startlux", "dialect": "openai_decisions"}, "SystemOne"),
    ({"decision_capabilities": {"question_types": ["completion"]}}, "question_types"),
    ({"decision_capabilities": {"modalities": ["audio"]}}, "modalities"),
    ({"endpoint_path": "https://other.test"}, "absolute URL paths"),
    ({"decision_capabilities": {"image_format": []}}, "image_format"),
    ({"decision_capabilities": {"image_format": {}}}, "image_format"),
    ({"auto_start": True}, "start_command"),
])
def test_invalid_decision_config_fails(changes, match):
    with pytest.raises(ValidationError, match=match):
        target(**changes)


@pytest.mark.parametrize("runtime", ["llama_cpp", "vllm"])
def test_builtin_managed_launch_uses_local_argv_and_never_downloads(runtime, tmp_path):
    model = tmp_path / "model with spaces.gguf"
    model.touch() if runtime == "llama_cpp" else model.mkdir()
    source = DecisionTarget(name="local", models=["arbitrary-future-model"], base_url="http://127.0.0.1:8010",
                            runtime=runtime, auto_start=True, model_path=str(model), binary_path="llama-server")
    argv = build_native_argv(source, validate=True)
    assert str(model) in argv and "--model" in argv
    assert not any("huggingface" in arg for arg in argv)
    with pytest.raises(ValidationError, match="download"):
        DecisionTarget(**{**source.model_dump(), "start_args": ["--hf-repo=vendor/weights"]})


async def test_missing_weights_fail_on_use_without_starting(monkeypatch, tmp_path):
    source = DecisionTarget(name="local", models=["m"], runtime="llama_cpp", auto_start=True,
                            model_path=str(tmp_path / "not-downloaded.gguf"), binary_path="llama-server")
    spawn = AsyncMock()
    monkeypatch.setattr("fake_ollama.managed_http_runtime.create_managed_subprocess_exec", spawn)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as http:
        client = DecisionClient(source, client=http)
        with pytest.raises(LocalTargetResourceError, match="Download weights separately"):
            await client.decide(request())
        assert not spawn.called
        assert client._request_refs == 0
        await client.aclose()


async def test_external_native_request_probe_and_metadata_no_chat_fallback():
    calls = []
    def handler(req):
        calls.append(req)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"version": "0.0.1"}, "paths": {"/v1/systemone": {"post": {}}}})
        if req.url.path == "/version":
            return httpx.Response(200, json={"version": "v0.6.0"})
        if req.url.path == "/v1/systemone":
            assert json.loads(req.content)["state"] == "A test input"
            return httpx.Response(200, json=response())
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(runtime="llama_cpp"), client=http)
        probe = await client.probe_capabilities()
        assert probe["version"] == "v0.6.0"
        assert probe["route_advertised"] is True and probe["endpoint_verified"] is False
        result = await client.decide(request())
        assert result.answers[0].probability == 0.73
        assert result.usage is None
        assert result.metadata["endpoint_verified"] is True
        assert client.capability_snapshot()["endpoint_verified"] is True
        assert not any("chat" in call.url.path for call in calls)
        await client.aclose()


async def test_known_old_version_warns_but_allows_verified_backport():
    def handler(req):
        if req.url.path == "/version":
            return httpx.Response(200, json={"version": "0.31.0"})
        if req.method == "POST":
            return httpx.Response(200, json=response())
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(runtime="vllm"), client=http)
        result = await client.decide(request())
        assert result.answers
        assert any("v0.31.0" in w for w in client.capability_snapshot()["warnings"])
        assert client.capability_snapshot()["endpoint_verified"] is True
        await client.aclose()


async def test_unsupported_native_endpoint_actionable_error():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as http:
        client = DecisionClient(target(runtime="vllm"), client=http)
        with pytest.raises(LocalTargetResourceError, match="stable 0.31.0") as exc:
            await client.decide(request())
        assert exc.value.error_type == "decision_endpoint_unsupported"
        assert client._active == client._request_refs == 0
        assert client.capability_snapshot()["endpoint_verified"] is False
        await client.aclose()


@pytest.mark.parametrize("options", [
    {"think_mode": "auto"},
    {"extensions": {"mystery": True}},
    {"assets": {"A": {"type": "image", "format": "png", "data_base64": "AAAA"}}},
])
async def test_capability_gates_fail_before_network(options):
    def handler(req):
        raise AssertionError("capability rejection must be pre-network")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(), client=http)
        with pytest.raises(DecisionValidationError):
            await client.decide(request(**options))
        await client.aclose()


async def test_explicit_autothink_and_image_header_preserved():
    def handler(req):
        if req.method == "POST":
            assert req.headers["TypeSafe-Danger-Zone-Images"] == "true"
            assert json.loads(req.content)["think_mode"] == "auto"
            return httpx.Response(200, json=response())
        return httpx.Response(404)
    capabilities = {"modalities": ["text", "image"], "extensions": ["think_mode"], "think_modes": ["auto", "on", "off"]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(decision_capabilities=capabilities), client=http)
        await client.decide(request(think_mode="auto", assets={"A": {"type": "image", "format": "png", "data_base64": "AAAA"}}))
        await client.aclose()


async def test_cancel_queued_request_releases_lease_but_active_request_drains():
    entered = asyncio.Event()
    release = asyncio.Event()
    async def handler(req):
        if req.method == "POST":
            entered.set()
            await release.wait()
            return httpx.Response(200, json=response())
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(max_concurrent_requests=1), client=http)
        active = asyncio.create_task(client.decide(request()))
        await asyncio.wait_for(entered.wait(), 2)
        queued = asyncio.create_task(client.decide(request()))
        for _ in range(20):
            if client.queued_requests:
                break
            await asyncio.sleep(0)
        assert client.queued_requests == 1
        queued.cancel()
        active.cancel()
        await asyncio.gather(queued, return_exceptions=True)
        await asyncio.sleep(0)
        assert not active.done()
        assert client._active == client._request_refs == 1
        assert client._request_semaphore._value == 0
        release.set()
        await asyncio.gather(queued, active, return_exceptions=True)
        assert client._active == client._request_refs == client._queued == 0
        assert client._request_semaphore._value == 1
        await client.aclose()


async def test_shutdown_wakes_all_queued_decisions():
    from fake_ollama.lifecycle import BackendShuttingDownError
    entered = asyncio.Event()
    release = asyncio.Event()
    async def handler(req):
        if req.method == "POST":
            entered.set()
            await release.wait()
            return httpx.Response(200, json=response())
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(max_concurrent_requests=1), client=http)
        active = asyncio.create_task(client.decide(request()))
        await entered.wait()
        queued = asyncio.create_task(client.decide(request()))
        for _ in range(20):
            if client.queued_requests:
                break
            await asyncio.sleep(0)
        client.begin_shutdown()
        with pytest.raises(BackendShuttingDownError):
            await asyncio.wait_for(queued, 2)
        active.cancel()
        release.set()
        await asyncio.gather(active, return_exceptions=True)
        assert client._request_refs == 0
        await client.aclose()


async def test_managed_runtime_start_idle_stop_and_no_external_stop(monkeypatch, tmp_path):
    from fake_ollama import managed_http_runtime
    path = tmp_path / "model.gguf"
    path.touch()
    source = DecisionTarget(name="local", models=["m"], runtime="llama_cpp", auto_start=True,
        model_path=str(path), binary_path="llama-server", idle_timeout_seconds=1)
    client = DecisionClient(source)
    process = type("Process", (), {"returncode": None})()
    spawn = AsyncMock(return_value=process)
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr(managed_http_runtime, "create_managed_subprocess_exec", spawn)
    monkeypatch.setattr(managed_http_runtime, "terminate_process_tree", stop)
    monkeypatch.setattr(client, "_healthy", AsyncMock(side_effect=[False, False, True]))
    await client._ensure_ready()
    assert spawn.await_count == 1
    assert spawn.call_args.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    client._last_used -= 10
    await client.stop_if_idle()
    assert stop.await_count == 1
    await client.aclose()
    external = DecisionClient(target(stop_command="must-not-run"))
    command = AsyncMock(return_value=True)
    monkeypatch.setattr(managed_http_runtime, "run_stop_command", command)
    external._used_since_last_stop = True
    await external.aclose()
    assert command.await_count == 0


async def test_missing_optional_startlux_keeps_app_running_and_surfaces_child_reason(monkeypatch, tmp_path):
    from fake_ollama.server import create_app
    from fake_ollama import managed_http_runtime
    model = tmp_path / "local-model"
    model.mkdir()
    for name in ("config.json", "decision_config.json"):
        (model / name).write_text("{}")
    source = DecisionTarget(name="optional", base_url="http://127.0.0.1:19817", models=["local"],
        runtime="startlux", auto_start=True, model_path=str(model), startup_timeout_seconds=120)
    app = create_app(Settings(decision_targets=[source], dashboard_enabled=False))
    log = tmp_path / "runtime.err.log"
    reason = "StartLux setup error: The selected Python has no startlux-decide package. Set python_path to its environment's Python."
    async def spawn(*args, **kwargs):
        log.write_text(reason + "\n")
        return type("ExitedChild", (), {"returncode": 2})()
    monkeypatch.setattr(managed_http_runtime, "create_managed_subprocess_exec", spawn)
    monkeypatch.setattr(managed_http_runtime, "terminate_process_tree", AsyncMock(return_value=True))
    async with app.router.lifespan_context(app):
        client = app.state.decision_clients["optional"]
        assert client._process is None  # Lifespan/config never imports or starts optional ML.
        monkeypatch.setattr(client, "_healthy", AsyncMock(return_value=False))
        monkeypatch.setattr(client, "_stderr_log_path", lambda: log)
        with pytest.raises(httpx.ConnectError, match="no startlux-decide package"):
            await asyncio.wait_for(client.decide(request()), timeout=1)
        assert client._active == client._request_refs == 0
        assert app.state.settings.decision_targets[0].name == "optional"


async def test_decision_reload_reuse_replacement_drain_and_snapshot():
    from fake_ollama.server import create_app
    from fake_ollama.admin import _swap_settings
    from fastapi import HTTPException
    source = target()
    settings = Settings(decision_targets=[source], dashboard_enabled=False)
    app = create_app(settings)
    entered, resume = asyncio.Event(), asyncio.Event()
    seen = []
    @app.get("/api/decision-runtime-generation")
    async def generation():
        seen.append(app.state.decision_clients["decision"])
        entered.set()
        await resume.wait()
        seen.append(app.state.decision_clients["decision"])
        return {"ok": True}
    async with app.router.lifespan_context(app):
        old = app.state.decision_clients["decision"]
        await _swap_settings(app, settings.model_copy(update={"advertised_version": "another"}))
        assert app.state.decision_clients["decision"] is old
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:21434") as http:
            in_flight = asyncio.create_task(http.get("/api/decision-runtime-generation"))
            await asyncio.wait_for(entered.wait(), 2)
            changed = source.model_copy(update={"decision_capabilities": {**source.decision_capabilities, "think_modes": ["auto"]}})
            await _swap_settings(app, settings.model_copy(update={"decision_targets": [changed]}))
            new = app.state.decision_clients["decision"]
            assert new is not old
            assert not old._client.is_closed
            with pytest.raises(HTTPException, match="draining"):
                app.state.runtime_resources.check(new)
            resume.set()
            assert (await in_flight).status_code == 200
            assert seen == [old, old]
            assert old._client.is_closed
            assert app.state.runtime_resources.check(new) is new
            assert new._ownership_active
            assert not app.state.runtime_resources.retired


async def test_decision_reload_cancelled_snapshot_still_retires_old_transport():
    from fake_ollama.server import create_app
    from fake_ollama.admin import _swap_settings
    entered = asyncio.Event()
    source = target()
    settings = Settings(decision_targets=[source], dashboard_enabled=False)
    app = create_app(settings)
    @app.get("/api/decision-runtime-cancel")
    async def pause():
        entered.set()
        await asyncio.Event().wait()
    async with app.router.lifespan_context(app):
        old = app.state.decision_clients["decision"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:21434") as http:
            task = asyncio.create_task(http.get("/api/decision-runtime-cancel"))
            await asyncio.wait_for(entered.wait(), 2)
            await _swap_settings(app, settings.model_copy(update={"decision_targets": []}))
            assert not old._client.is_closed
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert old._client.is_closed
            assert not app.state.runtime_resources.retired


async def test_dashboard_does_not_double_count_shared_chat_decision_model():
    from fake_ollama.dashboard import _collect_model_snapshots
    from fake_ollama.generic_openai_client import GenericOpenAIClient
    from fake_ollama.server import create_app
    first = GenericOpenAIClient("http://same-model.test:8010", target_name="chat")
    second = DecisionClient(DecisionTarget(name="decide", base_url="http://same-model.test:8010", models=["m"]))
    first._mark_vram_reserved("m", 4)
    second._mark_vram_reserved("m", 4)
    first._mark_memory_reserved("m", 6)
    second._mark_memory_reserved("m", 6)
    app = create_app(Settings())
    app.state.generic_openai_clients = {"chat": first}
    app.state.decision_clients = {"decide": second}
    snapshots = _collect_model_snapshots(app)
    assert len(snapshots) == 1
    assert snapshots[0]["estimated_vram_gb"] == 4
    assert snapshots[0]["estimated_memory_gb"] == 6
    assert snapshots[0]["shared_target_ids"] == [first.target_id, second.target_id]
    await first.aclose()
    await second.aclose()


@pytest.mark.parametrize("resource", ["vram", "memory"])
async def test_one_decision_client_does_not_admit_different_model_using_resident_model(resource):
    from fake_ollama.vram import VramCoordinator, MemoryCoordinator
    free = [6 * 1024.0]
    async def available():
        return free[0]
    coordinator = (VramCoordinator if resource == "vram" else MemoryCoordinator)(provider=available, total_provider=available)
    client = DecisionClient(target(), **{resource + "_coordinator": coordinator})
    await getattr(client, "_ensure_" + resource)("a", 4)
    getattr(client, "_mark_" + resource + "_reserved")("a", 4)
    coordinator.discard_pending(client.target_id, "a")
    free[0] = 1
    await getattr(client, "_ensure_" + resource)("a", 4)
    with pytest.raises(LocalTargetResourceError):
        await getattr(client, "_ensure_" + resource)("b", 4)
    assert getattr(client, "has_" + resource + "_reservation")("a")
    assert not getattr(client, "has_" + resource + "_reservation")("b")
    await client.aclose()


async def test_multiple_resident_decision_models_preserve_each_reservation():
    client = DecisionClient(target())
    for model in ("a", "b"):
        client._mark_vram_reserved(model, 4)
        client._mark_memory_reserved(model, 6)
    assert client.has_vram_reservation("a") and client.has_vram_reservation("b")
    assert client.has_memory_reservation("a") and client.has_memory_reservation("b")
    assert {snapshot["model"] for snapshot in client.loaded_model_snapshots()} == {"a", "b"}
    client._clear_all_vram_state()
    assert not client._loaded_models and not client.has_vram_reservation("a")
    await client.aclose()


def test_managed_wire_model_aliases_share_identity_and_argv():
    source = DecisionTarget(name="native", runtime="vllm", base_url="http://127.0.0.1:8010", auto_start=True,
        model_path="/local/weights", models=[{"name": "public-a", "upstream_id": "wire"}, {"name": "public-b", "upstream_id": "wire"}])
    assert source.resolve_model("public-a") == source.resolve_model("public-b") == "wire"
    assert build_native_argv(source).count("wire") == 1


async def test_health_restricts_misdeclared_startlux_instant_modes_without_post():
    calls = []
    def handler(req):
        calls.append(req)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "kind": "instant", "prompt_attention": "bidirectional", "images": False})
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DecisionClient(target(runtime="startlux", decision_capabilities={"modalities": ["text", "image"],
            "extensions": ["think_mode"], "think_modes": ["auto", "always", "never"]}), client=http)
        with pytest.raises(DecisionValidationError, match="Instant"):
            await client.decide(request(think_mode="auto"))
        snapshot = client.capability_snapshot()
        assert snapshot["think_modes"] == ["never"]
        assert snapshot["modalities"] == ["text"]
        assert not any(req.method == "POST" for req in calls)
        await client.aclose()


@pytest.mark.parametrize("tree_running", [False, True])
async def test_dead_owned_process_invalidates_sibling_residency_before_admission(monkeypatch, tree_running):
    from fake_ollama.generic_openai_client import GenericOpenAIClient
    from fake_ollama.vram import VramCoordinator
    from fake_ollama import managed_http_runtime
    async def available():
        return 1.0  # No room for a fresh 4-GiB load.
    coordinator = VramCoordinator(provider=available, total_provider=available)
    chat = GenericOpenAIClient("http://dead-shared.test:8010", target_name="chat", vram_coordinator=coordinator)
    decision = DecisionClient(DecisionTarget(name="decide", base_url="http://dead-shared.test:8010", models=["wire-d3"]), vram_coordinator=coordinator)
    process = type("ExitedLauncher", (), {"returncode": 0})()
    chat._process, chat._started_by_us = process, True
    chat._mark_vram_reserved("wire-d3", 4)
    coordinator.discard_pending(chat.target_id, "wire-d3")
    chat._ownership.capture(chat)
    monkeypatch.setattr(managed_http_runtime, "process_tree_running", AsyncMock(return_value=tree_running))
    monkeypatch.setattr(managed_http_runtime, "terminate_process_tree", AsyncMock(return_value=True))
    monkeypatch.setattr("fake_ollama.llama_cpp_client.terminate_process_tree", AsyncMock(return_value=True))
    monkeypatch.setattr(decision, "_ensure_ready", AsyncMock())
    if tree_running:
        async with decision.request_lease("wire-d3", estimated_vram_gb=4):
            assert chat.has_vram_reservation("wire-d3")
    else:
        with pytest.raises(LocalTargetResourceError):
            async with decision.request_lease("wire-d3", estimated_vram_gb=4):
                pytest.fail("stale sibling residency bypassed admission")
        assert not chat.has_vram_reservation("wire-d3")
        assert not decision.has_vram_reservation("wire-d3")
    await decision.aclose()
    await chat.aclose()


async def test_startlux_assets_rejected_before_provider_can_silently_ignore_them():
    client = DecisionClient(target(runtime="startlux", decision_capabilities={"modalities": ["text", "image"]}))
    with pytest.raises(DecisionValidationError, match="top-level images"):
        await client.decide(request(assets={"A": {"type": "image", "format": "png", "data_base64": "AAAA"}}))
    await client.aclose()
