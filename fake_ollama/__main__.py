"""CLI entrypoint: ``python -m fake_ollama``."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .config import get_settings
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="fake-ollama")
    parser.add_argument("--host", default=None, help="bind host (default from env)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default from env)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    app = create_app(settings)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
