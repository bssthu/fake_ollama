"""Application configuration.

Configuration is layered, in order of increasing priority:

  1. Defaults (hard-coded in this module)
  2. ``config.json`` (path overridable via ``--config`` CLI flag or
     ``FAKE_OLLAMA_CONFIG`` env var; default ``./config.json``)
  3. Environment variables (``FAKE_OLLAMA_*`` and the legacy single-upstream
     ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` pair)

The structured ``upstreams`` and ``model_profiles`` sections are best edited
in ``config.json``. Secrets can be kept out of the JSON file by leaving the
``auth_token`` placeholder there and overriding via env var.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import logging

from pydantic import BaseModel, Field, field_validator, model_validator

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CAPABILITIES: List[str] = ["completion", "tools", "vision"]
DEFAULT_CONTEXT_LENGTH: int = 200_000
DEFAULT_THINKING_BUDGET_TOKENS: int = 1024
VALID_THINKING_MODES = ("auto", "enabled", "disabled")

CONFIG_ENV_VAR = "FAKE_OLLAMA_CONFIG"
DEFAULT_CONFIG_PATH = Path("config.json")
LEGACY_UPSTREAM_NAME = "default"


# ---------------------------------------------------------------------------
# Per-model profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    capabilities: List[str]
    context_length: int
    max_output_tokens: Optional[int] = None
    thinking_mode: str = "auto"
    thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS
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


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


class Upstream(BaseModel):
    """A single Anthropic-compatible upstream endpoint."""

    name: str
    base_url: str
    auth_token: str = ""
    # Display names this upstream serves. The union across upstreams (with
    # order preserved and duplicates dropped, first occurrence wins) is what
    # /api/tags reports.
    models: List[str] = Field(default_factory=list)
    # Display name -> upstream-side model id. Falls through to the display
    # name itself when not present.
    model_map: Dict[str, str] = Field(default_factory=dict)
    # Subset of ``models`` that should be visible on the EXTERNAL
    # reverse-proxy surface (/v1/models, /v1/messages passthrough).
    # Semantics:
    #   None  -> expose all entries in ``models`` (legacy / default).
    #   []    -> expose nothing from this upstream externally.
    #   [...] -> expose only the listed display names.
    # ollama_targets are unaffected (they exist solely for external use).
    expose_external: Optional[List[str]] = None

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolve_model(self, display_name: str) -> str:
        if display_name in self.model_map:
            return self.model_map[display_name]
        if ":" in display_name:
            base = display_name.split(":", 1)[0]
            if base in self.model_map:
                return self.model_map[base]
            return base
        return display_name

    def serves(self, display_name: str) -> bool:
        if display_name in self.models:
            return True
        if ":" in display_name and display_name.split(":", 1)[0] in self.models:
            return True
        return False


class OllamaTarget(BaseModel):
    """A local-side Ollama-compatible server we expose as Anthropic API.

    Used by the reverse proxy ``POST /v1/messages`` endpoint: when an
    incoming request's ``model`` is served by an OllamaTarget, fake-ollama
    converts the request to Ollama's ``/api/chat`` format, calls the target,
    and converts the response back to the Anthropic Messages format.

    Access control for the reverse-proxy surface lives at the Settings level
    (``external_access_tokens``); targets themselves no longer carry tokens.
    """

    name: str
    base_url: str = "http://127.0.0.1:11434"
    # Anthropic-side display names this target serves.
    models: List[str] = Field(default_factory=list)
    # Anthropic-side display name -> Ollama-side model id.
    model_map: Dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}  # tolerate legacy ``api_token`` field

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolve_model(self, display_name: str) -> str:
        if display_name in self.model_map:
            return self.model_map[display_name]
        return display_name

    def serves(self, display_name: str) -> bool:
        return display_name in self.models


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    # ---- Internal listener (admin UI + Ollama-compatible /api/*) --------
    host: str = "127.0.0.1"
    port: int = 21434
    # ---- External listener (reverse-proxy /v1/*; optional) --------------
    # When ``external_port`` is set, /v1/* moves off the internal listener
    # and is served on this separate (host, port). When unset, /v1/* stays
    # on the internal listener (single-port mode).
    external_host: Optional[str] = None
    external_port: Optional[int] = None
    # Tokens accepted on /v1/messages and /v1/models (x-api-key or Bearer).
    # Required when an external listener is configured. Optional otherwise:
    # if non-empty, /v1/* on the internal listener also requires auth.
    external_access_tokens: List[str] = Field(default_factory=list)

    advertised_version: str = "0.6.4"
    default_max_tokens: int = 4096
    timeout_seconds: float = 300.0
    use_system_proxy: bool = False
    enforce_context_limit: bool = True
    upstreams: List[Upstream] = Field(default_factory=list)
    ollama_targets: List[OllamaTarget] = Field(default_factory=list)
    model_profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Web admin UI (mounted at /admin). Set to false to disable entirely.
    admin_enabled: bool = True

    # Where the JSON config came from (empty string if no file was used).
    config_path: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if not self.upstreams:
            raise ValueError(
                "At least one upstream is required. Either set ANTHROPIC_BASE_URL "
                "and ANTHROPIC_AUTH_TOKEN, or define an `upstreams` array in "
                "config.json."
            )
        names = [u.name for u in self.upstreams]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate upstream names: {names}")
        target_names = [t.name for t in self.ollama_targets]
        if len(set(target_names)) != len(target_names):
            raise ValueError(f"Duplicate ollama_target names: {target_names}")
        # Normalize tokens: drop blanks and dedupe.
        seen: Dict[str, None] = {}
        for tk in self.external_access_tokens:
            if tk and tk not in seen:
                seen[tk] = None
        object.__setattr__(self, "external_access_tokens", list(seen.keys()))
        # External listener requires at least one access token.
        if self.external_port is not None and not self.external_access_tokens:
            _LOG.warning(
                "external_port=%s is set but external_access_tokens is empty; "
                "/v1/* will refuse all requests until you add at least one "
                "token (Web UI: External → Generate).",
                self.external_port,
            )
        # Tokens configured but no ollama_target is unusual but allowed (the
        # tokens then only gate the upstream-passthrough side of /v1/*).
        return self

    # -- External-listener helpers ---------------------------------------

    @property
    def external_listener_enabled(self) -> bool:
        return self.external_port is not None

    @property
    def reverse_proxy_models(self) -> List[str]:
        """Union of model display names served by ``ollama_targets``."""
        seen: Dict[str, None] = {}
        for t in self.ollama_targets:
            for m in t.models:
                if m not in seen:
                    seen[m] = None
        return list(seen.keys())

    def is_valid_external_token(self, token: str) -> bool:
        """True iff ``token`` is in ``external_access_tokens``."""
        if not token:
            return False
        return token in self.external_access_tokens

    @property
    def externally_exposed_upstream_models(self) -> List[str]:
        """Upstream model names visible on /v1/* (subject to expose_external)."""
        seen: Dict[str, None] = {}
        for up in self.upstreams:
            allowed = up.models if up.expose_external is None else [
                m for m in up.models if m in set(up.expose_external)
            ]
            for m in allowed:
                if m not in seen:
                    seen[m] = None
        return list(seen.keys())

    def is_externally_exposed(self, display_name: str) -> bool:
        """True iff a model is reachable on the /v1/* reverse-proxy surface.

        ollama_targets are always exposed; upstream models follow each
        upstream's ``expose_external`` whitelist.
        """
        for t in self.ollama_targets:
            if t.serves(display_name):
                return True
        for up in self.upstreams:
            if up.expose_external is None:
                if display_name in up.models:
                    return True
            else:
                if display_name in up.expose_external and display_name in up.models:
                    return True
        return False

    @property
    def auth_required_for_v1(self) -> bool:
        """Whether /v1/messages and /v1/models require a token.

        True when any access token is configured OR an external listener is
        enabled (in which case auth is mandatory regardless of token list,
        though an empty list means no token can pass — effectively closed).
        """
        return bool(self.external_access_tokens) or self.external_listener_enabled

    # -- Backwards-compatible aggregated views ---------------------------

    @property
    def models(self) -> List[str]:
        """Union of all upstream models, dedup, order preserved."""
        seen: Dict[str, None] = {}
        for up in self.upstreams:
            for name in up.models:
                if name not in seen:
                    seen[name] = None
        return list(seen.keys())

    @property
    def upstream_url(self) -> str:
        """First upstream's base_url. Kept for backwards compatibility."""
        return self.upstreams[0].base_url if self.upstreams else ""

    @property
    def anthropic_auth_token(self) -> str:
        """First upstream's token. Kept for backwards compatibility."""
        return self.upstreams[0].auth_token if self.upstreams else ""

    # -- Routing helpers -------------------------------------------------

    def upstream_for_model(self, display_name: str) -> Upstream:
        """Return the upstream that should serve the given display name.

        Falls back to the first upstream when no explicit match exists, so
        unknown model names still get a sensible default route.
        """
        for up in self.upstreams:
            if up.serves(display_name):
                return up
        return self.upstreams[0]

    def upstream_name_for(self, display_name: str) -> str:
        return self.upstream_for_model(display_name).name

    def resolve_model(self, display_name: str) -> str:
        return self.upstream_for_model(display_name).resolve_model(display_name)

    # -- Reverse-proxy routing -------------------------------------------

    def ollama_target_for(self, display_name: str):
        """Return the OllamaTarget that should serve the given display name,
        or ``None`` if no target serves it."""
        for t in self.ollama_targets:
            if t.serves(display_name):
                return t
        return None

    def profile_for(self, display_name: str) -> ModelProfile:
        raw = self.model_profiles.get(display_name)
        if raw is None and ":" in display_name:
            raw = self.model_profiles.get(display_name.split(":", 1)[0])
        if raw is None:
            return ModelProfile(
                capabilities=list(DEFAULT_CAPABILITIES),
                context_length=DEFAULT_CONTEXT_LENGTH,
                max_output_tokens=None,
            )
        return ModelProfile.from_dict(raw)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


_ENV_SCALARS: Dict[str, tuple] = {
    "FAKE_OLLAMA_HOST": ("host", str),
    "FAKE_OLLAMA_PORT": ("port", int),
    "FAKE_OLLAMA_EXTERNAL_HOST": ("external_host", str),
    "FAKE_OLLAMA_EXTERNAL_PORT": ("external_port", int),
    "FAKE_OLLAMA_ADVERTISED_VERSION": ("advertised_version", str),
    "FAKE_OLLAMA_DEFAULT_MAX_TOKENS": ("default_max_tokens", int),
    "FAKE_OLLAMA_TIMEOUT": ("timeout_seconds", float),
    "FAKE_OLLAMA_USE_SYSTEM_PROXY": ("use_system_proxy", _parse_bool),
    "FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT": ("enforce_context_limit", _parse_bool),
}


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


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    # Scalar overrides
    for env_key, (field, caster) in _ENV_SCALARS.items():
        if env_key in os.environ:
            try:
                data[field] = caster(os.environ[env_key])
            except (TypeError, ValueError):
                continue

    # CSV-list override for external_access_tokens.
    raw_tokens = os.getenv("FAKE_OLLAMA_EXTERNAL_ACCESS_TOKENS")
    if raw_tokens is not None:
        data["external_access_tokens"] = [
            t.strip() for t in raw_tokens.split(",") if t.strip()
        ]

    # model_profiles via FAKE_OLLAMA_MODEL_PROFILES (JSON)
    raw_profiles = os.getenv("FAKE_OLLAMA_MODEL_PROFILES")
    if raw_profiles:
        try:
            data["model_profiles"] = json.loads(raw_profiles)
        except json.JSONDecodeError:
            pass

    # Legacy single-upstream env vars. When set, they create or override the
    # upstream named "default".
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    auth = os.getenv("ANTHROPIC_AUTH_TOKEN")
    if base_url and auth:
        legacy_models_str = os.getenv("FAKE_OLLAMA_MODELS")
        legacy_map_str = os.getenv("FAKE_OLLAMA_MODEL_MAP")
        legacy_models = (
            [m.strip() for m in legacy_models_str.split(",") if m.strip()]
            if legacy_models_str
            else []
        )
        legacy_map: Dict[str, str] = {}
        if legacy_map_str:
            try:
                legacy_map = json.loads(legacy_map_str)
            except json.JSONDecodeError:
                legacy_map = {}
        env_up = {
            "name": LEGACY_UPSTREAM_NAME,
            "base_url": base_url,
            "auth_token": auth,
            "models": legacy_models,
            "model_map": legacy_map,
        }
        upstreams = list(data.get("upstreams") or [])
        for i, up in enumerate(upstreams):
            if up.get("name") == LEGACY_UPSTREAM_NAME:
                merged = {**up, **env_up}
                if not legacy_models and up.get("models"):
                    merged["models"] = up["models"]
                if not legacy_map and up.get("model_map"):
                    merged["model_map"] = up["model_map"]
                upstreams[i] = merged
                break
        else:
            upstreams.insert(0, env_up)
        data["upstreams"] = upstreams

    return data


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Build a Settings object from JSON config + env vars."""
    resolved = _resolve_config_path(config_path)
    data = _read_json(resolved)
    data = _apply_env_overrides(data)
    data = _migrate_legacy_target_tokens(data)
    settings = Settings(**data)
    if resolved is not None:
        settings = settings.model_copy(update={"config_path": str(resolved)})
    return settings


def _migrate_legacy_target_tokens(data: Dict[str, Any]) -> Dict[str, Any]:
    """Hoist legacy per-target ``api_token`` into ``external_access_tokens``.

    Older config.json files carried an ``api_token`` on each ollama_target.
    The new model centralises the access-token list on the Settings; we keep
    backward compat by lifting any non-empty per-target tokens into the
    central list (deduped) and then dropping the legacy field so the
    OllamaTarget validator does not warn about it.
    """
    targets = data.get("ollama_targets")
    if not isinstance(targets, list):
        return data
    existing = list(data.get("external_access_tokens") or [])
    seen = {t for t in existing if isinstance(t, str)}
    migrated = False
    for tgt in targets:
        if not isinstance(tgt, dict):
            continue
        tok = tgt.pop("api_token", None)
        if isinstance(tok, str) and tok and tok not in seen:
            existing.append(tok)
            seen.add(tok)
            migrated = True
    if migrated:
        _LOG.warning(
            "migrated legacy ollama_target.api_token into external_access_tokens; "
            "please re-save config from the Web UI to make this permanent."
        )
        data["external_access_tokens"] = existing
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


# ---------------------------------------------------------------------------
# Token estimation (rough – used only for the cost guardrail)
# ---------------------------------------------------------------------------


def estimate_tokens_from_anthropic_payload(body: Dict[str, Any]) -> int:
    """Rough token estimate of an Anthropic /v1/messages request body."""
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
    overhead = 4 * len(body.get("messages") or [])
    return math.ceil(chars / 3) + overhead + images * 1500
