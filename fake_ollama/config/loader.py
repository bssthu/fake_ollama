"""Filesystem and environment loading for validated application settings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Settings


CONFIG_ENV_VAR = "FAKE_OLLAMA_CONFIG"
DEFAULT_CONFIG_PATH = Path("config.json")


def _resolve_config_path(explicit: Optional[str | Path]) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path)
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    return json.loads(raw)


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    resolved = _resolve_config_path(config_path)
    settings = Settings(**_read_json(resolved))
    if resolved is not None:
        settings = settings.model_copy(update={"config_path": str(resolved)})
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
