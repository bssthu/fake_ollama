"""CLI entrypoint: ``python -m fake_ollama``."""

from __future__ import annotations

import argparse
import logging

import uvicorn
from dotenv import load_dotenv

from .config import load_settings
from .server import create_app


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
    parser.add_argument("--host", default=None, help="bind host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default from config)")
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
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
