"""CLI entrypoint: ``python -m fake_ollama``."""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn
from dotenv import load_dotenv

from .config import load_settings
from .server import create_app

logger = logging.getLogger("fake_ollama")


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
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings(config_path=args.config)
    host = args.host or settings.host
    port = args.port or settings.port

    app = create_app(settings)

    # Internal listener: /admin + /api/* (+ /v1/* if no external listener).
    internal_cfg = uvicorn.Config(app, host=host, port=port, log_level=args.log_level)

    if not settings.external_listener_enabled:
        uvicorn.Server(internal_cfg).run()
        return

    ext_host = settings.external_host or "127.0.0.1"
    ext_port = settings.external_port  # type: ignore[assignment]
    external_cfg = uvicorn.Config(
        app, host=ext_host, port=int(ext_port), log_level=args.log_level
    )
    logger.info(
        "fake-ollama listening on internal=%s:%s, external=%s:%s",
        host, port, ext_host, ext_port,
    )

    async def _run_both() -> None:
        s1 = uvicorn.Server(internal_cfg)
        s2 = uvicorn.Server(external_cfg)
        await asyncio.gather(s1.serve(), s2.serve())

    try:
        asyncio.run(_run_both())
    except KeyboardInterrupt:
        logger.info("interrupted; exiting")


if __name__ == "__main__":  # pragma: no cover
    main()
