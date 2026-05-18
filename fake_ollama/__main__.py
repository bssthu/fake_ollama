"""CLI entrypoint: ``python -m fake_ollama``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import FrameType
from typing import Callable, Optional

import uvicorn

from .config import load_settings
from .request_data_log import (
    DEFAULT_REQUEST_DATA_LOG_FILE,
    configure_request_data_logging,
)
from .server import create_app, request_shutdown

logger = logging.getLogger("fake_ollama")
DEFAULT_LOG_FILE = Path("logs") / "fake_ollama.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _ShutdownAwareServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, on_shutdown_requested: Callable[[], None]) -> None:
        super().__init__(config)
        self._on_shutdown_requested = on_shutdown_requested

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self._on_shutdown_requested()
        super().handle_exit(sig, frame)


def _configure_logging(level_name: str, *, log_file: Optional[str]) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        level = logging.INFO

    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers, force=True)


def main() -> None:
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
    parser.add_argument("--dashboard-host", default=None, help="dashboard bind host (default from config)")
    parser.add_argument("--dashboard-port", type=int, default=None, help="dashboard bind port (default from config)")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="write application logs to this file (default: logs/fake_ollama.log)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="disable file logging and only log to stderr",
    )
    parser.add_argument(
        "--request-data-log-file",
        default=str(DEFAULT_REQUEST_DATA_LOG_FILE),
        help=(
            "write full request/response data JSONL to this file "
            "(default: logs/fake_ollama.requests.jsonl)"
        ),
    )
    parser.add_argument(
        "--no-request-data-log",
        action="store_true",
        help="disable the separate full request/response data JSONL log",
    )
    args = parser.parse_args()

    log_file = None if args.no_log_file else args.log_file
    _configure_logging(args.log_level, log_file=log_file)
    request_data_log_file = (
        None if args.no_request_data_log else args.request_data_log_file
    )
    configure_request_data_logging(request_data_log_file)

    settings = load_settings(config_path=args.config)
    logger.info(
        "logging initialised%s",
        f"; file={Path(log_file)}" if log_file else "; file=disabled",
    )
    logger.info(
        "request data logging initialised%s",
        (
            f"; file={Path(request_data_log_file)}"
            if request_data_log_file
            else "; file=disabled"
        ),
    )

    updates = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.admin_host is not None:
        updates["admin_host"] = args.admin_host
    if args.admin_port is not None:
        updates["admin_port"] = args.admin_port
    if args.dashboard_host is not None:
        updates["dashboard_host"] = args.dashboard_host
    if args.dashboard_port is not None:
        updates["dashboard_port"] = args.dashboard_port
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
        log_config=None,
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
            log_config=None,
            access_log=False,
        )
        configs.append(("external", ext_host, ext_port, external_cfg))
    if settings.admin_listener_enabled:
        admin_cfg = uvicorn.Config(
            app,
            host=settings.admin_host,
            port=int(settings.admin_port),  # type: ignore[arg-type]
            log_level=args.log_level,
            log_config=None,
            access_log=False,
        )
        configs.append(("admin", settings.admin_host, int(settings.admin_port), admin_cfg))
    if settings.dashboard_listener_enabled:
        dashboard_cfg = uvicorn.Config(
            app,
            host=settings.dashboard_host,
            port=int(settings.dashboard_port),  # type: ignore[arg-type]
            log_level=args.log_level,
            log_config=None,
            access_log=False,
        )
        configs.append(
            (
                "dashboard",
                settings.dashboard_host,
                int(settings.dashboard_port),  # type: ignore[arg-type]
                dashboard_cfg,
            )
        )

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
