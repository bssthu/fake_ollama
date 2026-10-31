"""Process-wide log privacy, inherited by gateway-owned Python workers."""

import os


PRIVACY_MODE_ENV = "FAKE_OLLAMA_PRIVACY_MODE"


def privacy_mode_enabled() -> bool:
    return os.environ.get(PRIVACY_MODE_ENV) == "1"


def configure_privacy_mode(enabled: bool) -> None:
    if enabled:
        os.environ[PRIVACY_MODE_ENV] = "1"
    else:
        os.environ.pop(PRIVACY_MODE_ENV, None)
