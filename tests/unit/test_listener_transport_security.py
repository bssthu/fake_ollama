"""Verify the deployed ASGI stack preserves direct connection identity."""
import asyncio
import sys


def test_cli_listeners_do_not_rewrite_peer_or_scheme_from_forwarded_headers(settings, monkeypatch):
    from fake_ollama import __main__ as cli
    seen = []
    async def application(scope, receive, send):
        seen.append((scope["client"], scope["scheme"]))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    class Server:
        def __init__(self, config, callback):
            self.config = config
        async def serve(self):
            self.config.load()
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                "scheme": "http", "path": "/", "root_path": "", "query_string": b"",
                "server": (self.config.host, self.config.port), "client": ("127.0.0.1", 50000),
                "headers": [(b"host", b"localhost"), (b"x-forwarded-for", b"198.51.100.9"), (b"x-forwarded-proto", b"https")]}
            async def receive():
                return {"type": "http.request", "body": b""}
            async def send(message):
                pass
            await self.config.loaded_app(scope, receive, send)
        def run(self):
            asyncio.run(self.serve())
    monkeypatch.setattr(sys, "argv", ["fake-ollama", "--no-log-file", "--no-request-data-log"])
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "configure_request_data_logging", lambda *a: None)
    monkeypatch.setattr(cli, "load_settings", lambda **kw: settings)
    monkeypatch.setattr(cli, "create_app", lambda *a: application)
    monkeypatch.setattr(cli, "_ShutdownAwareServer", Server)
    cli.main()
    assert seen and all(value == (("127.0.0.1", 50000), "http") for value in seen)
