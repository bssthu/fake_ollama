"""CLI entrypoint: ``python -m fake_ollama``."""

from __future__ import annotations

import argparse
import asyncio
import logging
from types import FrameType
from typing import Callable

import uvicorn
from dotenv import load_dotenv

from .config import load_settings
from .server import create_app, request_shutdown

logger = logging.getLogger("fake_ollama")


class _ShutdownAwareServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, on_shutdown_requested: Callable[[], None]) -> None:
        super().__init__(config)
        self._on_shutdown_requested = on_shutdown_requested

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self._on_shutdown_requested()
        super().handle_exit(sig, frame)


def main() -> None:
    # Load .env so ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / FAKE_OLLAMA_*
    # variables are visible to the loader.
    load_dotenv()

    parser = argparse.ArgumentParser(prog="fake-ollama")
    parser.add_argument(
        "--config",
        default=None,
        help="path to config.json (default: $FAKE_OLLAMA_CONFIG or ./config.json)",
    )
    parser.add_argument("--host", default=None, help="internal bind host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="internal bind port (default from config)")
    parser.add_argument("--admin-host", default=None, help="admin bind host (default from config)")
    parser.add_argument("--admin-port", type=int, default=None, help="admin bind port (default from config)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings(config_path=args.config)
    updates = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.admin_host is not None:
        updates["admin_host"] = args.admin_host
    if args.admin_port is not None:
        updates["admin_port"] = args.admin_port
    if updates:
        data = settings.model_dump()
        data.update(updates)
        settings = type(settings)(**data)

    host = settings.host
    port = settings.port

    app = create_app(settings)

    # Internal listener: /api/* (+ /v1/* if no external listener).
    internal_cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=args.log_level,
        access_log=False,
    )

    configs = [("internal", host, port, internal_cfg)]
    if settings.external_listener_enabled:
        ext_host = settings.external_host or "127.0.0.1"
        ext_port = int(settings.external_port)  # type: ignore[arg-type]
        external_cfg = uvicorn.Config(
            app,
            host=ext_host,
            port=ext_port,
            log_level=args.log_level,
            access_log=False,
        )
        configs.append(("external", ext_host, ext_port, external_cfg))
    if settings.admin_listener_enabled:
        admin_cfg = uvicorn.Config(
            app,
            host=settings.admin_host,
            port=int(settings.admin_port),  # type: ignore[arg-type]
            log_level=args.log_level,
            access_log=False,
        )
        configs.append(("admin", settings.admin_host, int(settings.admin_port), admin_cfg))

    logger.info(
        "fake-ollama listening on %s",
        ", ".join(f"{name}={cfg_host}:{cfg_port}" for name, cfg_host, cfg_port, _ in configs),
    )

    def _request_shutdown() -> None:
        request_shutdown(app)

    if len(configs) == 1:
        _ShutdownAwareServer(configs[0][3], _request_shutdown).run()
        return

    async def _run_all() -> None:
        servers = [_ShutdownAwareServer(cfg, _request_shutdown) for _, _, _, cfg in configs]
        await asyncio.gather(*(server.serve() for server in servers))

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        _request_shutdown()
        logger.info("interrupted; exiting")


if __name__ == "__main__":  # pragma: no cover
    main()
