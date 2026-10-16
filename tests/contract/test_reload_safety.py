import asyncio

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from fake_ollama.admin import _swap_settings
from fake_ollama.config import OpenAIUpstream
from fake_ollama.server import create_app, _playground_interface_for
from fastapi.responses import StreamingResponse
from fake_ollama.security import playground_anonymous_allowed


async def test_unrelated_reload_reuses_anthropic_client(settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        previous = app.state.clients["default"]
        updated = settings.model_copy(update={"advertised_version": "new-version"})
        await _swap_settings(app, updated)
        assert app.state.clients["default"] is previous
        assert not previous._client.is_closed


async def test_reload_updates_separate_remote_openai_pool(settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        updated = settings.model_copy(update={"openai_upstreams": [OpenAIUpstream(name="remote", base_url="http://remote.test", models=[{"name": "m"}])]})
        await _swap_settings(app, updated)
        assert "remote" in app.state.openai_clients
        assert app.state.openai_clients["remote"].base_url == "http://remote.test"
        await _swap_settings(app, settings)
        assert "remote" not in app.state.openai_clients


async def test_reload_cannot_turn_public_playground_socket_into_anonymous(settings):
    settings = settings.model_copy(update={"playground_enabled": True, "playground_host": "0.0.0.0"})
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await _swap_settings(app, settings.model_copy(update={"playground_host": "127.0.0.1"}))
        request = Request({"type": "http", "app": app, "method": "GET", "path": "/playground/api/models",
                           "scheme": "http", "server": ("127.0.0.1", 21431), "client": ("127.0.0.1", 50000),
                           "headers": [(b"host", b"127.0.0.1:21431")]})
        with pytest.raises(HTTPException):
            _playground_interface_for(request, app.state.settings)


@pytest.mark.parametrize("host,peer,origin,server,forwarded,allowed", [
    ("127.0.0.1:21431", "127.0.0.1", None, "127.0.0.1", False, True),
    ("evil.test:21431", "127.0.0.1", None, "127.0.0.1", False, False),
    ("127.0.0.1:21431", "192.0.2.7", None, "127.0.0.1", True, False),
    ("127.0.0.1:21431", "127.0.0.1", "http://evil.test", "127.0.0.1", False, False),
    ("127.0.0.1:21431", "127.0.0.1", None, "0.0.0.0", False, False),
    ("127.0.0.1:80", "127.0.0.1", None, "127.0.0.1", False, False),
])
def test_playground_anonymous_checks_connection_and_browser(settings, host, peer, origin, server, forwarded, allowed):
    app = create_app(settings.model_copy(update={"playground_enabled": True}))
    headers = [(b"host", host.encode())]
    if origin:
        headers.append((b"origin", origin.encode()))
    if forwarded:
        headers.extend([(b"x-forwarded-for", b"127.0.0.1"), (b"x-forwarded-host", b"localhost")])
    request = Request({"type": "http", "app": app, "scheme": "http", "server": (server, 21431),
                       "client": (peer, 50000), "headers": headers})
    assert playground_anonymous_allowed(request) is allowed


@pytest.mark.parametrize("peer,origin,allowed", [
    ("192.0.2.10", "https://play.example", True),
    ("192.0.2.11", "https://play.example", False),
    ("192.0.2.10", "https://evil.example", False),
])
def test_explicit_proxy_boundary(settings, peer, origin, allowed):
    app = create_app(settings.model_copy(update={"playground_enabled": True, "playground_host": "0.0.0.0",
        "playground_trusted_proxy_ips": ["192.0.2.10"], "playground_allowed_origins": ["https://play.example"]}))
    request = Request({"type": "http", "app": app, "scheme": "http", "server": ("0.0.0.0", 21431),
        "client": (peer, 50000), "headers": [(b"host", b"play.example"), (b"origin", origin.encode())]})
    assert playground_anonymous_allowed(request) is allowed


async def test_request_stream_keeps_generation_until_complete(settings):
    app = create_app(settings)
    entered, resume = asyncio.Event(), asyncio.Event()
    seen = []

    @app.get("/api/generation-test")
    async def route():
        async def body():
            before = app.state.clients["default"]
            seen.append((app.state.settings.advertised_version, before))
            yield b"data: first\n\n"
            entered.set()
            await resume.wait()
            seen.append((app.state.settings.advertised_version, app.state.clients["default"]))
            assert not before._client.is_closed
            yield b"data: last\n\n"
        return StreamingResponse(body(), media_type="text/event-stream")

    async with app.router.lifespan_context(app):
        old = app.state.clients["default"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                base_url=f"http://127.0.0.1:{settings.ollama_interfaces[0].port}") as client:
            pending = asyncio.create_task(client.get("/api/generation-test"))
            await asyncio.wait_for(entered.wait(), 3)
            changed_up = settings.anthropic_upstreams[0].model_copy(update={"base_url": "http://changed.test"})
            await _swap_settings(app, settings.model_copy(update={"advertised_version": "second", "anthropic_upstreams": [changed_up]}))
            assert app.state.clients["default"] is not old
            assert not old._client.is_closed
            resume.set()
            response = await pending
            assert response.text == "data: first\n\ndata: last\n\n"
            assert seen == [(settings.advertised_version, old)] * 2
            assert old._client.is_closed
            assert not app.state.runtime_resources.retired


@pytest.mark.parametrize("change", ["base_url", "auth_token", "timeout_seconds", "use_system_proxy"])
async def test_remote_openai_reload_replaces_changed_transport(settings, change):
    up = OpenAIUpstream(name="remote", base_url="http://remote.test", models=[{"name": "m"}])
    settings = settings.model_copy(update={"openai_upstreams": [up]})
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        old = app.state.openai_clients["remote"]
        values = {"base_url": "http://next.test", "auth_token": "test-only", "timeout_seconds": settings.timeout_seconds + 1,
                  "use_system_proxy": not settings.use_system_proxy}
        updated = settings.model_copy(update={change: values[change]}) if change in ("timeout_seconds", "use_system_proxy") else settings.model_copy(update={"openai_upstreams": [up.model_copy(update={change: values[change]})]})
        await _swap_settings(app, updated)
        current = app.state.openai_clients["remote"]
        assert current is not old and old._client.is_closed and not current._client.is_closed
        assert current.base_url == updated.openai_upstreams[0].base_url
        assert current.auth_token == updated.openai_upstreams[0].auth_token
        assert current._timeout == updated.timeout_seconds
        assert current._trust_env == updated.use_system_proxy


async def test_atomic_save_failure_preserves_file_runtime_and_closes_prepared(settings, tmp_path, monkeypatch):
    import fake_ollama.admin as admin
    import fake_ollama.config_persistence as persistence
    path = tmp_path / "synthetic-config.json"
    path.write_text('{"old":true}', encoding="utf-8")
    created = []
    constructor = admin.AnthropicClient
    def capture(*args, **kwargs):
        client = constructor(*args, **kwargs)
        created.append(client)
        return client
    def fail(*args):
        raise PermissionError("synthetic locked destination")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        old = app.state.clients["default"]
        monkeypatch.setattr(admin, "AnthropicClient", capture)
        monkeypatch.setattr(persistence.os, "replace", fail)
        with pytest.raises(PermissionError):
            await _swap_settings(app, settings.model_copy(update={"timeout_seconds": settings.timeout_seconds + 1}), save_path=path)
        assert path.read_text() == '{"old":true}'
        assert not list(tmp_path.glob(".synthetic-config.json.*.tmp"))
        assert app.state.settings is settings and app.state.clients["default"] is old
        assert not old._client.is_closed and created[0]._client.is_closed


async def test_prepare_failure_rolls_back_all_new_clients(settings, monkeypatch):
    import fake_ollama.admin as admin
    app = create_app(settings)
    created = []
    constructor = admin.AnthropicClient
    def capture(*args, **kwargs):
        client = constructor(*args, **kwargs)
        created.append(client)
        return client
    def fail(*args, **kwargs):
        raise ValueError("synthetic construction failure")
    async with app.router.lifespan_context(app):
        old = app.state.clients["default"]
        monkeypatch.setattr(admin, "AnthropicClient", capture)
        monkeypatch.setattr(admin, "OpenAIClient", fail)
        updated = settings.model_copy(update={"timeout_seconds": settings.timeout_seconds + 1,
            "openai_upstreams": [OpenAIUpstream(name="new", base_url="http://new.test", models=[{"name": "m"}])]})
        with pytest.raises(ValueError):
            await _swap_settings(app, updated)
        assert app.state.settings is settings and not old._client.is_closed
        assert created[0]._client.is_closed and not app.state.openai_clients


async def test_concurrent_reload_serializes_prepare_and_publish(settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        one = settings.model_copy(update={"timeout_seconds": settings.timeout_seconds + 1})
        two = settings.model_copy(update={"timeout_seconds": settings.timeout_seconds + 2})
        await asyncio.gather(_swap_settings(app, one), _swap_settings(app, two))
        client = app.state.clients["default"]
        assert app.state.settings.timeout_seconds == client._timeout == two.timeout_seconds
        assert not client._client.is_closed
        assert len(app.state.runtime_resources.managed) == 1


async def test_multiple_reloads_gate_local_replacement_until_old_request_drains(settings):
    from fake_ollama.config import LlamaCppTarget
    target = LlamaCppTarget(name="local", base_url="http://127.0.0.1:21436", model="m")
    settings = settings.model_copy(update={"llama_cpp_targets": [target]})
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        manager = app.state.runtime_resources
        old = app.state.llama_cpp_clients["local"]
        manager.references[old] = 1
        for timeout in (settings.timeout_seconds + 1, settings.timeout_seconds + 2):
            await _swap_settings(app, settings.model_copy(update={"timeout_seconds": timeout}))
            current = app.state.llama_cpp_clients["local"]
            with pytest.raises(HTTPException) as error:
                manager.check(current)
            assert error.value.status_code == 503
            assert not old._client.is_closed
            assert app.state.vram_coordinator._participants[old.target_id] is old
        manager.drain_seconds = 0
        assert manager.snapshots()[0]["status"] == "waiting_for_requests"
        manager.references.pop(old)
        manager.drain_seconds = 1
        await manager.close_ready()
        assert old._client.is_closed
        assert manager.check(current) is current
        assert app.state.vram_coordinator._participants[current.target_id] is current


async def test_cleanup_failure_is_observable_and_retryable(settings):
    app = create_app(settings)
    manager = app.state.runtime_resources
    class Resource:
        attempts = 0
        async def aclose(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("synthetic cleanup failure")
    resource = Resource()
    await manager.retire([resource])
    assert manager.snapshots()[0]["status"] == "cleanup_failed"
    await manager.retry_cleanup()
    assert not manager.snapshots() and resource.attempts == 2
