"""Lifecycle-managed client for generic OpenAI-compatible model servers.

This covers vLLM, SGLang, TGI, and small adapter services that expose the
OpenAI Chat Completions surface but still need fake_ollama to own start/stop,
health checks, queueing, and resource coordination.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .llama_cpp_client import LlamaCppClient


class GenericOpenAIClient(LlamaCppClient):
    """A generic OpenAI-compatible lifecycle client.

    The request/streaming protocol is the same as llama.cpp's OpenAI server,
    so this intentionally reuses the battle-tested lifecycle implementation
    while giving the target a generic identity and log file namespace.
    """

    def __init__(self, *args, target_name: str = "generic_openai", **kwargs) -> None:
        super().__init__(
            *args,
            target_name=target_name,
            target_prefix="generic_openai",
            target_log_label="generic OpenAI",
            **kwargs,
        )

    def _stderr_log_path(self) -> Optional[Path]:
        parsed = urlparse(self._base)
        port = parsed.port
        host = (parsed.hostname or "").strip()
        safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", self.target_id).strip("_")
        if port:
            stem = f"generic-openai-{port}"
        elif safe_target:
            stem = f"generic-openai-{safe_target}"
        else:
            return None
        if host and host not in {"127.0.0.1", "localhost", "::1"}:
            safe_host = re.sub(r"[^A-Za-z0-9._-]+", "_", host)
            stem = f"{stem}-{safe_host}"
        return Path("logs") / f"{stem}.err.log"
