"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


# Default capabilities advertised when a model has no explicit profile.
# Anthropic models support all three.
DEFAULT_CAPABILITIES: List[str] = ["completion", "tools", "vision"]
# Default context window when no profile is supplied. Conservative – clients
# (including GitHub Copilot) display this value, so make it match Anthropic
# Claude 3.5 / 4 family which is 200k tokens.
DEFAULT_CONTEXT_LENGTH: int = 200_000


# Thinking modes for reasoning models (DeepSeek-V3.2, Claude 3.7+):
#   "auto"     - do not touch the upstream `thinking` field; let the client
#                or upstream default decide.
#   "enabled"  - inject `thinking: {type:"enabled", budget_tokens:N}` if
#                the client did not already specify one.
#   "disabled" - inject `thinking: {type:"disabled"}` (force off).
VALID_THINKING_MODES = ("auto", "enabled", "disabled")
DEFAULT_THINKING_BUDGET_TOKENS: int = 1024


@dataclass(frozen=True)
class ModelProfile:
    """Per-model metadata: advertised capabilities and limits."""

    capabilities: List[str]
    context_length: int
    max_output_tokens: Optional[int] = None
    # Reasoning / thinking controls. See VALID_THINKING_MODES.
    thinking_mode: str = "auto"
    thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS
    # Whether to surface upstream `thinking` blocks back to the client. When
    # True we wrap them in <think>...</think> in the output text (and also
    # emit OpenAI-style `reasoning_content` deltas on /v1/chat/completions).
    show_thinking: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelProfile":
        caps = data.get("capabilities")
        if not isinstance(caps, list) or not caps:
            caps = list(DEFAULT_CAPABILITIES)
        ctx = data.get("context_length") or data.get("num_ctx") or DEFAULT_CONTEXT_LENGTH
        out = data.get("max_output_tokens") or data.get("max_tokens")
        thinking = str(data.get("thinking", data.get("thinking_mode", "auto"))).lower()
        if thinking not in VALID_THINKING_MODES:
            thinking = "auto"
        budget = data.get("thinking_budget_tokens") or data.get("thinking_budget") or DEFAULT_THINKING_BUDGET_TOKENS
        show = data.get("show_thinking")
        return cls(
            capabilities=[str(c) for c in caps],
            context_length=int(ctx),
            max_output_tokens=int(out) if out else None,
            thinking_mode=thinking,
            thinking_budget_tokens=int(budget),
            show_thinking=True if show is None else bool(show),
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_base_url: str = Field(..., alias="ANTHROPIC_BASE_URL")
    anthropic_auth_token: str = Field(..., alias="ANTHROPIC_AUTH_TOKEN")

    host: str = Field("127.0.0.1", alias="FAKE_OLLAMA_HOST")
    port: int = Field(21434, alias="FAKE_OLLAMA_PORT")

    # Version string returned by /api/version. Some clients refuse to talk to
    # servers older than a minimum Ollama version, so we advertise a recent
    # one by default. Override via FAKE_OLLAMA_ADVERTISED_VERSION if needed.
    advertised_version: str = Field("0.6.4", alias="FAKE_OLLAMA_ADVERTISED_VERSION")

    default_max_tokens: int = Field(4096, alias="FAKE_OLLAMA_DEFAULT_MAX_TOKENS")
    timeout_seconds: float = Field(300.0, alias="FAKE_OLLAMA_TIMEOUT")
    # If False (default), httpx ignores HTTP_PROXY/HTTPS_PROXY/system proxy
    # when calling the upstream. Useful on machines where Clash/V2Ray etc.
    # would otherwise hijack the request to a local proxy.
    use_system_proxy: bool = Field(False, alias="FAKE_OLLAMA_USE_SYSTEM_PROXY")

    # If True (default), reject requests whose estimated input + max_tokens
    # exceed the configured context_length for the target model. This is a
    # cheap guardrail to avoid huge upstream bills from runaway prompts.
    enforce_context_limit: bool = Field(True, alias="FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT")

    models: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["claude-3-5-sonnet-20241022"],
        alias="FAKE_OLLAMA_MODELS",
    )
    model_map: Annotated[Dict[str, str], NoDecode] = Field(
        default_factory=dict,
        alias="FAKE_OLLAMA_MODEL_MAP",
    )

    # JSON object: { "<ollama-name>": { "capabilities": [...],
    #                                   "context_length": 200000,
    #                                   "max_output_tokens": 8192 }, ... }
    # Unspecified models fall back to defaults.
    model_profiles: Annotated[Dict[str, Dict[str, Any]], NoDecode] = Field(
        default_factory=dict,
        alias="FAKE_OLLAMA_MODEL_PROFILES",
    )

    @field_validator("models", mode="before")
    @classmethod
    def _split_models(cls, value):
        if value is None or value == "":
            return ["claude-3-5-sonnet-20241022"]
        if isinstance(value, str):
            return [m.strip() for m in value.split(",") if m.strip()]
        return value

    @field_validator("model_map", mode="before")
    @classmethod
    def _parse_model_map(cls, value):
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("model_profiles", mode="before")
    @classmethod
    def _parse_model_profiles(cls, value):
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            return json.loads(value)
        return value

    @property
    def upstream_url(self) -> str:
        return self.anthropic_base_url.rstrip("/")

    def resolve_model(self, ollama_name: str) -> str:
        """Map an Ollama-style model name to the upstream model id."""
        if ollama_name in self.model_map:
            return self.model_map[ollama_name]
        # Strip optional ":tag" suffix that Ollama clients sometimes append.
        if ":" in ollama_name:
            base = ollama_name.split(":", 1)[0]
            if base in self.model_map:
                return self.model_map[base]
            return base
        return ollama_name

    def profile_for(self, ollama_name: str) -> ModelProfile:
        """Return the ModelProfile for a given Ollama-style model name."""
        raw = self.model_profiles.get(ollama_name)
        if raw is None and ":" in ollama_name:
            raw = self.model_profiles.get(ollama_name.split(":", 1)[0])
        if raw is None:
            return ModelProfile(
                capabilities=list(DEFAULT_CAPABILITIES),
                context_length=DEFAULT_CONTEXT_LENGTH,
                max_output_tokens=None,
            )
        return ModelProfile.from_dict(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Token estimation (rough – used only for the cost guardrail)
# ---------------------------------------------------------------------------


def estimate_tokens_from_anthropic_payload(body: Dict[str, Any]) -> int:
    """Rough token estimate of an Anthropic /v1/messages request body.

    Uses ~3 chars per token as a conservative (over-estimating) heuristic so
    that the guardrail trips slightly early rather than slightly late. Images
    and tool_result blocks add a fixed overhead.
    """
    chars = 0
    images = 0
    sys = body.get("system")
    if isinstance(sys, str):
        chars += len(sys)
    elif isinstance(sys, list):
        for block in sys:
            if isinstance(block, dict) and block.get("type") == "text":
                chars += len(block.get("text", ""))
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        if not isinstance(content, Iterable):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                chars += len(block.get("text", ""))
            elif btype == "image":
                images += 1
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str):
                    chars += len(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            chars += len(sub.get("text", ""))
            elif btype == "tool_use":
                chars += len(json.dumps(block.get("input") or {}))
    # rough overhead per message turn
    overhead = 4 * len(body.get("messages") or [])
    # ~1500 tokens per image is a safe upper bound for Claude vision
    return math.ceil(chars / 3) + overhead + images * 1500

