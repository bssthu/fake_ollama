"""Compatibility exports and CLI for the standalone Mage-VL service."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .api import create_app
from .control import shutdown_credentials
from .engine import MageEngine
from .request import prepare_request
from .settings import AdapterSettings, AnalysisOptions, PreparedRequest, SegmentWindow
from .video import extract_segment_frames, format_timestamp, probe_duration, select_segment_windows


__all__ = [
    "AdapterSettings",
    "AnalysisOptions",
    "MageEngine",
    "PreparedRequest",
    "SegmentWindow",
    "create_app",
    "extract_segment_frames",
    "format_timestamp",
    "prepare_request",
    "probe_duration",
    "select_segment_windows",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8071)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn

    app = create_app()
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
        proxy_headers=False,
        timeout_graceful_shutdown=None,
    )
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server
    # A failed duplicate start must not replace the running instance's token.
    sock = config.bind_socket()
    try:
        with shutdown_credentials(sock.getsockname()[1]) as token:
            app.state.shutdown_token = token
            server.run(sockets=[sock])
    finally:
        app.state.shutdown_token = None
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
