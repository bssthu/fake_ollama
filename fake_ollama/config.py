"""Application configuration (v2 schema).

This module models the new configuration layout used since the
"multi-interface + per-source aliases" refactor:

* Sources (``anthropic_upstreams``, ``openai_upstreams``, ``ollama_targets``,
  ``llama_cpp_targets``) declare models as objects: ``{name, alias?,
  upstream_id?}``. ``alias`` is the source-level display name; ``upstream_id``
  is the wire-side identifier sent to the backend. The composite source id
  is ``<alias_or_name>@<source_name>``.

* Interfaces (``ollama_interfaces`` for /api/*, ``api_interfaces`` for
  /v1/messages + /v1/chat/completions + /v1/models) are arrays. Each
  interface has its own ``host``, ``port``, ``access_tokens``, and
  ``exposed_models``. ``exposed_models[*]`` references a (target, model)
  pair from any source and optionally renames it via ``alias`` for clients.

* The legacy single ``host`` / ``port`` / ``external_*`` /
  ``internal_exposed_models`` / ``external_exposed_models`` fields are
  removed. Loaders raise on these keys instead of silently migrating.

Module-level helpers:

* ``Settings.resolve_request(public_id, interface_name)`` looks up the
  client-facing model id within a specific interface and returns
  ``(backend, real_model)``.
* ``Settings.detect_upstream_cycles()`` walks every upstream's
  ``base_url`` against every interface's bind address and raises if any
  upstream points back at this process.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import os
import shlex
import socket
import subprocess
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

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

# Header used to detect runtime cycles when a fake_ollama process talks to
# another fake_ollama process through one of its own upstreams. Routing
# stamps it on every outbound request and rejects inbound requests that
# carry the same instance marker.
FORWARDED_BY_HEADER = "x-fake-ollama-forwarded-by"

# Per-process identifier appended to the forwarded-by header on every
# outbound request. Generated once at import time. Tests that need a
# stable value can monkey-patch ``fake_ollama.config.INSTANCE_ID``.
INSTANCE_ID: str = uuid.uuid4().hex

# Tracks the forwarded-by chain of the currently-handled inbound request
# so each outbound HTTP client can stamp the right value without having
# its signature changed.
_inbound_forwarded_chain: ContextVar[Tuple[str, ...]] = ContextVar(
    "fake_ollama_forwarded_chain", default=()
)


def set_inbound_forwarded_chain(chain: Tuple[str, ...]):
    """Record the forwarded-by chain parsed from the inbound request."""
    return _inbound_forwarded_chain.set(tuple(chain))


def reset_inbound_forwarded_chain(token) -> None:
    _inbound_forwarded_chain.reset(token)


def current_inbound_forwarded_chain() -> Tuple[str, ...]:
    return _inbound_forwarded_chain.get()


def outbound_forwarded_chain() -> str:
    """Header value for an outbound request, including this process's id."""
    chain = list(_inbound_forwarded_chain.get())
    if INSTANCE_ID not in chain:
        chain.append(INSTANCE_ID)
    return ",".join(chain)


def outbound_cycle_headers() -> Dict[str, str]:
    """Convenience: ``{FORWARDED_BY_HEADER: ...}`` for client ``_headers``."""
    return {FORWARDED_BY_HEADER: outbound_forwarded_chain()}


def parse_forwarded_chain(raw: str) -> Tuple[str, ...]:
    """Split a comma-separated forwarded-by header value into a tuple."""
    if not raw:
        return ()
    return tuple(tok.strip() for tok in raw.split(",") if tok.strip())


def _normalize_public_id(pid: str) -> str:
    """Normalize a model id for cycle comparison.

    Ollama treats ``foo`` and ``foo:latest`` as the same model. The cycle
    detector folds them so an exposure named ``qwen3`` cannot sneak past
    by being requested as ``qwen3:latest`` on the next hop.
    """
    if not pid:
        return pid
    if "@" in pid:
        # Composite ``display@target`` ids: normalise the display half.
        disp, _, target = pid.rpartition("@")
        return f"{_normalize_public_id(disp)}@{target}"
    if pid.endswith(":latest"):
        return pid[: -len(":latest")]
    return pid


# Sentinel returned by the cycle detector's host resolver to flag that an
# upstream's base_url points at one of *our* listeners that does not serve
# model traffic (admin / dashboard). Treated as a hard error.
_MISROUTED_LISTENER = object()


def _model_base(name: str) -> str:
    """Return the Ollama-style tagless model name (``foo:tag`` -> ``foo``)."""
    return name.split(":", 1)[0] if ":" in name else name


def _shell_quote(value: str) -> str:
    """Quote ``value`` for the platform's default shell.

    ``create_subprocess_shell`` uses ``cmd.exe`` on Windows, which does not
    treat single quotes as quoting characters — so ``shlex.quote`` (which
    emits ``'...'``) silently breaks every path that contains a backslash
    or space. ``subprocess.list2cmdline`` produces the right ``"..."``
    form on Windows. On POSIX we keep ``shlex.quote``.
    """
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _find_configured_model(display_name: str, candidates: Iterable[str]) -> Optional[str]:
    """Match a client-supplied name against ``candidates`` Ollama-style.

    ``foo`` and ``foo:latest`` are equivalent; an explicit tag must match
    exactly. ``foo`` will not match ``foo:q4_K_M`` etc.
    """
    candidates = list(candidates)
    if display_name in candidates:
        return display_name
    base = _model_base(display_name)
    if ":" in display_name:
        tag = display_name.split(":", 1)[1]
        if tag == "latest" and base in candidates:
            return base
        return None
    latest = f"{base}:latest"
    if latest in candidates:
        return latest
    return None


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
    estimated_vram_gb: Optional[float] = None

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
        budget = (
            data.get("thinking_budget_tokens")
            or data.get("thinking_budget")
            or DEFAULT_THINKING_BUDGET_TOKENS
        )
        show = data.get("show_thinking")
        raw_vram = data.get("estimated_vram_gb")
        estimated_vram_gb: Optional[float] = None
        if raw_vram not in (None, ""):
            try:
                parsed = float(raw_vram)
                if parsed > 0:
                    estimated_vram_gb = parsed
            except (TypeError, ValueError):
                estimated_vram_gb = None
        return cls(
            capabilities=[str(c) for c in caps],
            context_length=int(ctx),
            max_output_tokens=int(out) if out else None,
            thinking_mode=thinking,
            thinking_budget_tokens=int(budget),
            show_thinking=True if show is None else bool(show),
            estimated_vram_gb=estimated_vram_gb,
        )


# ---------------------------------------------------------------------------
# Model entry (per-source)
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    """One model advertised by a source (upstream or local target).

    * ``name`` — required; the model id the source itself knows about.
    * ``alias`` — optional; renames the model for fake_ollama's purposes.
      When set, the composite source id becomes ``<alias>@<source>`` and
      every downstream reference (interface exposure, profiles, routing)
      should use the alias instead of ``name``.
    * ``upstream_id`` — optional; the actual wire-side id sent to the
      backend. Falls back to ``name``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    alias: Optional[str] = None
    upstream_id: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("model entry 'name' must be non-empty")
        if "@" in v:
            raise ValueError(
                "model entry 'name' must not contain '@'; '@' is reserved "
                "for composite model@target ids"
            )
        return v

    @field_validator("alias")
    @classmethod
    def _alias_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if "@" in v:
            raise ValueError(
                "model entry 'alias' must not contain '@'; '@' is reserved "
                "for composite model@target ids"
            )
        return v

    @field_validator("upstream_id")
    @classmethod
    def _upstream_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @property
    def display(self) -> str:
        """The name used in the composite ``<display>@<source>`` id."""
        return self.alias or self.name

    @property
    def wire_id(self) -> str:
        """The id actually sent to the backend."""
        return self.upstream_id or self.name


def _coerce_model_entries(value: Any) -> List[Dict[str, Any]]:
    """Normalise a ``models`` field into a list of ModelEntry-shaped dicts.

    Accepts:
    * a list of strings (bare model names; alias/upstream_id default to None)
    * a list of dicts (already in the canonical shape)
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("'models' must be a list")
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            out.append({"name": item})
        elif isinstance(item, dict):
            out.append(dict(item))
        elif isinstance(item, ModelEntry):
            out.append(item.model_dump())
        else:
            raise ValueError(
                f"each models[] entry must be a string or object, got {type(item).__name__}"
            )
    return out


def _validate_unique_displays(entries: List[ModelEntry], context: str) -> None:
    seen: Dict[str, None] = {}
    dupes: List[str] = []
    for e in entries:
        d = e.display
        if d in seen:
            dupes.append(d)
        else:
            seen[d] = None
    if dupes:
        raise ValueError(
            f"duplicate model display names {sorted(set(dupes))} in {context}; "
            f"each entry must have a unique alias_or_name within its source"
        )


# ---------------------------------------------------------------------------
# Source classes
# ---------------------------------------------------------------------------


class _SourceBase(BaseModel):
    """Shared lookup helpers for upstream-style sources."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    auth_token: str = ""
    models: List[ModelEntry] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("source 'name' must be non-empty")
        if "@" in v:
            raise ValueError(
                f"source name {v!r} contains '@'; '@' is reserved as the "
                "model/target separator in composite ids"
            )
        return v

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return (v or "").rstrip("/")

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models(cls, v: Any) -> Any:
        return _coerce_model_entries(v)

    @model_validator(mode="after")
    def _check_unique(self) -> "_SourceBase":
        _validate_unique_displays(self.models, f"source {self.name!r}")
        return self

    # -- public-ish view ------------------------------------------------

    @property
    def display_models(self) -> List[str]:
        """Display names this source advertises (alias_or_name per entry)."""
        return [e.display for e in self.models]

    def entry_for(self, display_name: str) -> Optional[ModelEntry]:
        """Return the ModelEntry matching ``display_name`` (Ollama-style)."""
        match = _find_configured_model(display_name, self.display_models)
        if match is None:
            return None
        for e in self.models:
            if e.display == match:
                return e
        return None

    def matching_model(self, display_name: str) -> Optional[str]:
        """Display name that matches ``display_name`` (or None)."""
        entry = self.entry_for(display_name)
        return entry.display if entry is not None else None

    def resolve_model(self, display_name: str) -> str:
        """Map ``display_name`` to the backend wire-side id."""
        entry = self.entry_for(display_name)
        if entry is None:
            return display_name
        return entry.wire_id

    def serves(self, display_name: str) -> bool:
        return self.entry_for(display_name) is not None


class AnthropicUpstream(_SourceBase):
    """A single Anthropic Messages-compatible upstream endpoint."""


# Back-compat alias for any external imports that still say ``Upstream``.
Upstream = AnthropicUpstream


class OpenAIUpstream(_SourceBase):
    """An OpenAI Chat Completions-compatible upstream endpoint."""


class OllamaTarget(_SourceBase):
    """A local-side Ollama-compatible server we proxy ``/v1/*`` to."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:11434"
    auth_token: str = ""
    auto_start: bool = False
    start_command: Optional[str] = None
    stop_command: Optional[str] = None
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: float = 60.0
    health_path: str = "/api/version"
    cwd: Optional[str] = None

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: str) -> str:
        if not v:
            return "/api/version"
        return v if v.startswith("/") else "/" + v


class LlamaCppDefaults(BaseModel):
    """Defaults inherited by each llama.cpp target unless overridden."""

    model_config = ConfigDict(extra="forbid")

    auto_start: bool = False
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: float = 120.0
    health_path: str = "/health"
    cwd: Optional[str] = None
    binary_path: Optional[str] = None
    runtime_root: Optional[str] = None
    gpu_layers: Optional[int] = None
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    flash_attn: Optional[bool] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    extra_args: Optional[str] = None
    # Concurrency / timeout knobs honoured by LlamaCppClient.
    # ``max_concurrent_requests``:
    #   * ``None`` -> no fake_ollama-level queue; requests pass straight
    #     through to the upstream llama.cpp server, which manages its own
    #     ``--parallel`` slots.  This is the default.
    #   * ``0``    -> explicitly disable the gate (same as ``None``).
    #   * ``>0``   -> cap simultaneous upstream requests, queue the rest.
    # ``request_read_timeout_seconds``:
    #   * ``None`` -> inherit ``timeout_seconds``.
    #   * ``<=0``  -> disable the read timeout entirely (httpx ``None``).
    #   * ``>0``   -> override only the read leg of the httpx timeout.
    max_concurrent_requests: Optional[int] = None
    request_read_timeout_seconds: Optional[float] = None

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: str) -> str:
        if not v:
            return "/health"
        return v if v.startswith("/") else "/" + v


class LlamaCppTarget(BaseModel):
    """One llama.cpp server process exposed through fake-ollama.

    Each ``llama_cpp_targets`` entry represents exactly one model, one
    server process, one port, and one set of lifecycle commands.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    base_url: str = "http://127.0.0.1:8080"
    auth_token: str = ""

    # Single model the process loads.
    model: str = ""
    # Source-level display alias: renames the model in the composite id.
    alias: Optional[str] = None
    # Wire-side id sent to llama.cpp (--alias and the OpenAI ``model``
    # field returned by the upstream).
    upstream_id: Optional[str] = None

    # Optional lifecycle management.
    auto_start: Optional[bool] = None
    start_command: Optional[str] = None
    stop_command: Optional[str] = None
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: Optional[float] = None
    health_path: Optional[str] = None
    cwd: Optional[str] = None

    # Synthesised start_command parameters.
    binary_path: Optional[str] = None
    runtime_root: Optional[str] = None
    model_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    gpu_layers: Optional[int] = None
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    flash_attn: Optional[bool] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    extra_args: Optional[str] = None
    max_concurrent_requests: Optional[int] = None
    request_read_timeout_seconds: Optional[float] = None

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return (v or "").rstrip("/")

    @field_validator("alias")
    @classmethod
    def _alias_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if "@" in v:
            raise ValueError("llama_cpp_targets[*].alias must not contain '@'")
        return v

    @field_validator("upstream_id")
    @classmethod
    def _upstream_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not v:
            return "/health"
        return v if v.startswith("/") else "/" + v

    @model_validator(mode="after")
    def _post_init(self) -> "LlamaCppTarget":
        if not self.model:
            raise ValueError(
                "Each llama_cpp_target must declare a non-empty `model`. "
                "Each llama.cpp server process loads one model; configure "
                "additional models as additional llama_cpp_targets."
            )
        if not self.name:
            object.__setattr__(self, "name", self.alias or self.model)
        if "@" in self.name:
            raise ValueError(
                f"llama_cpp_target name {self.name!r} contains '@'; reserved separator"
            )
        return self

    # KV-cache types that require flash-attention to run. Anything in this
    # set as ``-ctk`` / ``-ctv`` will crash llama-server's CUDA attention
    # kernel on the first request unless ``-fa on`` is also passed.
    _QUANTIZED_KV_CACHE_TYPES = frozenset(
        {"q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
    )

    @staticmethod
    def _normalised_cache_type(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip().lower()
        return s or None

    @staticmethod
    def _effective_cache_type(value: Optional[str]) -> str:
        """Resolve a cache type to what llama-server actually uses.

        ``None``, ``""`` and ``"f16"`` all map to ``"f16"`` because that
        is llama-server's default when ``-ctk`` / ``-ctv`` is omitted.
        Anything else is returned in lower-case so equality checks are
        case-insensitive.
        """
        s = (value or "").strip().lower()
        return s or "f16"

    def validate_kv_cache_types_match(self) -> None:
        """Reject asymmetric K/V cache configurations.

        llama.cpp's CUDA flash-attention kernel only has fast, well-tuned
        paths for matched K/V cache types. Mismatched pairs (e.g. K=f16 /
        V=q8_0) fall back to a generic path that runs much slower and
        leaves the GPU under-utilised. A missing field defaults to f16, so
        a config like ``cache_type_k=""`` + ``cache_type_v="q8_0"`` is
        still asymmetric and is rejected here.
        """
        k_eff = self._effective_cache_type(self.cache_type_k)
        v_eff = self._effective_cache_type(self.cache_type_v)
        if k_eff == v_eff:
            return
        raise ValueError(
            f"llama_cpp_target {self.name!r}: cache_type_k={self.cache_type_k!r} "
            f"and cache_type_v={self.cache_type_v!r} resolve to different types "
            f"({k_eff} vs {v_eff}). llama.cpp's CUDA flash-attention kernel only "
            f"has fast paths for matched K/V; mismatches fall back to a slow "
            f"generic path that drops GPU utilisation. Set both to the same value, "
            f"or leave both unset to keep the default f16."
        )

    def validate_kv_cache_requires_flash_attn(self) -> None:
        """Reject KV-cache configs that crash llama-server at runtime.

        llama.cpp requires flash-attention (``-fa on``) whenever the K or V
        KV-cache is anything other than the default f16. Without flash
        attention, the CUDA attention kernel cannot read a quantized cache
        and the server aborts with a memory access violation (Windows exit
        ``3221225477`` / ``0xC0000005``) on the first request. This check
        must be run *after* defaults are applied, since ``flash_attn`` can
        come from ``llama_cpp_defaults``.
        """
        k = self._normalised_cache_type(self.cache_type_k)
        v = self._normalised_cache_type(self.cache_type_v)
        if k is None and v is None:
            return
        if self.flash_attn:
            return
        issues: List[str] = []
        for label, val in (("cache_type_k", k), ("cache_type_v", v)):
            if val in self._QUANTIZED_KV_CACHE_TYPES:
                issues.append(f"{label}={val!r}")
        if not issues and k is not None and v is not None and k != v:
            issues.append(f"cache_type_k={k!r} != cache_type_v={v!r}")
        if not issues:
            return
        raise ValueError(
            f"llama_cpp_target {self.name!r}: {', '.join(issues)} requires "
            f"flash_attn=true (-fa on). llama-server's CUDA attention kernel "
            f"cannot read a quantized KV-cache without flash-attention and "
            f"crashes on the first request (Windows exit 3221225477 / "
            f"0xC0000005). Set \"flash_attn\": true on the target or in "
            f"llama_cpp_defaults."
        )

    # -- presentation helpers ------------------------------------------

    @property
    def display(self) -> str:
        return self.alias or self.model

    @property
    def effective_max_concurrent_requests(self) -> Optional[int]:
        """Resolved concurrency gate honoured by ``LlamaCppClient``.

        Only an explicit ``max_concurrent_requests`` installs a fake_ollama-
        level queue.  We deliberately do NOT inherit from ``parallel``:

        * ``parallel`` controls llama.cpp's own slot count and HTTP-level
          queue.  When several requests pile up at the upstream the server
          can pre-tokenise / prefill the next prompt while the current one
          is still decoding, so slot transitions are seamless.
        * A proxy-side cap (this value) is strictly serial: request B does
          not even start its HTTP call to llama.cpp until request A's stream
          fully closes, which adds one full round-trip + tokenisation latency
          per transition and makes ``cap=1`` markedly slower than the
          ``cap=None`` baseline even though both serialise the slot.

        ``0`` is preserved as the explicit "no gate" sentinel and is
        semantically identical to ``None``.
        """
        return self.max_concurrent_requests

    @property
    def models(self) -> List[ModelEntry]:
        """Mimic the multi-entry ``models`` shape used by other sources."""
        return [ModelEntry(name=self.model, alias=self.alias, upstream_id=self.upstream_id)]

    @property
    def display_models(self) -> List[str]:
        return [self.display]

    def entry_for(self, display_name: str) -> Optional[ModelEntry]:
        for candidate in (self.display, self.model):
            match = _find_configured_model(display_name, [candidate])
            if match is not None:
                return self.models[0]
        return None

    def matching_model(self, display_name: str) -> Optional[str]:
        entry = self.entry_for(display_name)
        return entry.display if entry is not None else None

    def resolve_model(self, display_name: str) -> str:
        entry = self.entry_for(display_name)
        if entry is None:
            return display_name
        return entry.wire_id

    def serves(self, display_name: str) -> bool:
        return self.entry_for(display_name) is not None

    def with_defaults(self, defaults: LlamaCppDefaults) -> "LlamaCppTarget":
        def pick(field: str) -> Any:
            cur = getattr(self, field)
            return cur if cur is not None else getattr(defaults, field)

        return self.model_copy(
            update={
                "auto_start": (
                    self.auto_start if self.auto_start is not None else defaults.auto_start
                ),
                "idle_timeout_seconds": pick("idle_timeout_seconds"),
                "startup_timeout_seconds": (
                    self.startup_timeout_seconds
                    if self.startup_timeout_seconds is not None
                    else defaults.startup_timeout_seconds
                ),
                "health_path": self.health_path or defaults.health_path,
                "cwd": self.cwd if self.cwd is not None else defaults.cwd,
                "binary_path": pick("binary_path"),
                "runtime_root": pick("runtime_root"),
                "gpu_layers": pick("gpu_layers"),
                "ctx_size": pick("ctx_size"),
                "parallel": pick("parallel"),
                "batch_size": pick("batch_size"),
                "ubatch_size": pick("ubatch_size"),
                "flash_attn": pick("flash_attn"),
                "cache_type_k": pick("cache_type_k"),
                "cache_type_v": pick("cache_type_v"),
                "extra_args": pick("extra_args"),
                "max_concurrent_requests": pick("max_concurrent_requests"),
                "request_read_timeout_seconds": pick("request_read_timeout_seconds"),
            }
        )

    def synthesize_start_command(self) -> Optional[str]:
        if self.start_command:
            return self.start_command
        argv = self.synthesize_start_argv()
        if argv is None:
            return None
        return " ".join(_shell_quote(a) for a in argv)

    def synthesize_start_argv(self) -> Optional[List[str]]:
        if self.start_command:
            return None
        if not self.model_path:
            return None
        parsed = urlparse(self.base_url or "")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        binary = self._resolve_binary(self.binary_path)
        argv: List[str] = [
            binary,
            "--host", host,
            "--port", str(port),
            "--model", self.model_path,
        ]
        if self.mmproj_path:
            argv += ["--mmproj", self.mmproj_path]
        if self.gpu_layers is not None:
            argv += ["-ngl", str(self.gpu_layers)]
        if self.ctx_size:
            argv += ["--ctx-size", str(self.ctx_size)]
        if self.parallel:
            argv += ["--parallel", str(self.parallel)]
        if self.batch_size:
            argv += ["-b", str(self.batch_size)]
        if self.ubatch_size:
            argv += ["-ub", str(self.ubatch_size)]
        if self.flash_attn:
            argv += ["-fa", "on"]
        if self.cache_type_k:
            argv += ["-ctk", str(self.cache_type_k)]
        if self.cache_type_v:
            argv += ["-ctv", str(self.cache_type_v)]
        if self.upstream_id:
            argv += ["--alias", self.upstream_id]
        if self.auth_token:
            argv += ["--api-key", self.auth_token]
        if self.extra_args:
            extra = self.extra_args.strip()
            if extra:
                argv += shlex.split(extra, posix=(os.name != "nt"))
        return argv

    def effective_env(
        self, base_env: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, str]]:
        if not self.runtime_root and not self.binary_path:
            return None
        env = dict(base_env if base_env is not None else os.environ)
        sep = os.pathsep
        prepend: list[str] = []
        binary = self._resolve_binary(self.binary_path)
        if binary:
            try:
                bin_dir = str(Path(binary).resolve().parent)
                if bin_dir and Path(bin_dir).is_dir():
                    prepend.append(bin_dir)
            except (OSError, ValueError):
                pass
        if self.runtime_root:
            try:
                rt = str(Path(self.runtime_root).resolve())
                if Path(rt).is_dir():
                    prepend.append(rt)
            except (OSError, ValueError):
                prepend.append(self.runtime_root)
        if not prepend:
            return None
        existing = env.get("PATH", "")
        env["PATH"] = sep.join(prepend + ([existing] if existing else []))
        return env

    @staticmethod
    def _resolve_binary(binary_path: Optional[str]) -> str:
        if not binary_path:
            return "llama-server"
        try:
            p = Path(binary_path)
        except (OSError, ValueError):
            return binary_path
        if p.is_dir():
            for name in ("llama-server.exe", "llama-server"):
                candidate = p / name
                if candidate.exists():
                    return str(candidate)
        return binary_path


class ComfyUITarget(BaseModel):
    """One ComfyUI workflow-backed image model exposed through fake-ollama."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    base_url: str = "http://127.0.0.1:8188"
    auth_token: str = ""

    # Single public source-level model served by this target.
    model: str = "z-image-turbo"
    alias: Optional[str] = None
    upstream_id: Optional[str] = None

    # Optional lifecycle management for the ComfyUI server process.
    auto_start: bool = False
    start_command: Optional[str] = None
    stop_command: Optional[str] = None
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: float = 120.0
    health_path: str = "/system_stats"
    cwd: Optional[str] = None

    # Declarative workflow preset (see ``comfyui_presets``). The default
    # ``z_image_turbo`` reproduces the legacy behaviour from the per-field node
    # ids/model files below; ``qwen_image_edit_aio`` and ``sensenova_u1`` ship
    # bundled workflows + bindings. Set ``bindings`` / ``static_inputs`` to
    # override a preset (or describe an arbitrary workflow) without touching the
    # client; both are keyed by mode, e.g. {"t2i": {...}, "i2i": {...}}.
    preset: str = "z_image_turbo"
    bindings: Optional[Dict[str, Any]] = None
    static_inputs: Optional[Dict[str, Any]] = None

    # Workflow files. When omitted, the preset's bundled API workflows are used.
    text_to_image_workflow_path: Optional[str] = None
    image_to_image_workflow_path: Optional[str] = None

    # Z-Image-Turbo default model files and sampling parameters.
    diffusion_model: str = "z-image-turbo-fp8-e4m3fn.safetensors"
    diffusion_weight_dtype: str = "default"
    text_encoder_model: str = "qwen_3_4b_fp4_mixed.safetensors"
    text_encoder_type: str = "lumina2"
    text_encoder_device: str = "default"
    vae_model: str = "ae.safetensors"
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 8
    default_cfg: float = 1.0
    default_sampler_name: str = "res_multistep"
    default_scheduler: str = "simple"
    default_denoise: float = 1.0
    default_edit_denoise: float = 0.25
    default_shift: float = 3.0
    # Random seed control for the KSampler node. seed_mode:
    #   "random"    -> a fresh random seed per request (default)
    #   "fixed"     -> always use `seed`
    #   "increment" -> start at `seed`, advance by the image count per request
    # An explicit `seed` in the request body always overrides this.
    seed_mode: str = "random"
    seed: int = 0
    max_batch_size: int = 4
    output_prefix: str = "fake_ollama/z-image-turbo"
    image_upscale_method: str = "lanczos"
    image_crop: str = "center"
    prompt_timeout_seconds: float = 600.0
    poll_interval_seconds: float = 0.5

    # Node ids for the bundled workflow shape. Operators can override these
    # when pointing at a custom API-format workflow that keeps the same inputs.
    unet_node_id: str = "28"
    clip_node_id: str = "30"
    vae_node_id: str = "29"
    prompt_node_id: str = "27"
    latent_node_id: str = "13"
    sampling_node_id: str = "11"
    ksampler_node_id: str = "3"
    save_image_node_id: str = "9"
    load_image_node_id: str = "12"
    image_scale_node_id: str = "14"

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return (v or "").rstrip("/")

    @field_validator("alias")
    @classmethod
    def _alias_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if "@" in v:
            raise ValueError("comfyui_targets[*].alias must not contain '@'")
        return v

    @field_validator("upstream_id")
    @classmethod
    def _upstream_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: str) -> str:
        if not v:
            return "/system_stats"
        return v if v.startswith("/") else "/" + v

    @field_validator(
        "default_width",
        "default_height",
        "default_steps",
        "max_batch_size",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError("ComfyUI image dimensions/steps/batch size must be positive")
        return int(v)

    @field_validator(
        "default_cfg",
        "default_denoise",
        "default_edit_denoise",
        "default_shift",
        "startup_timeout_seconds",
        "prompt_timeout_seconds",
        "poll_interval_seconds",
    )
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if float(v) < 0:
            raise ValueError("ComfyUI numeric workflow settings must be non-negative")
        return float(v)

    @field_validator("seed_mode")
    @classmethod
    def _valid_seed_mode(cls, v: str) -> str:
        v = (v or "random").strip().lower()
        if v not in ("random", "fixed", "increment"):
            raise ValueError(
                "comfyui_targets[*].seed_mode must be one of random|fixed|increment"
            )
        return v

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("comfyui_targets[*].seed must be >= 0")
        return int(v)

    @field_validator("preset")
    @classmethod
    def _valid_preset(cls, v: str) -> str:
        from .comfyui_presets import is_known_preset, preset_names

        v = (v or "z_image_turbo").strip()
        if not is_known_preset(v):
            raise ValueError(
                f"comfyui_targets[*].preset {v!r} is unknown; "
                f"known presets: {preset_names()}"
            )
        return v

    @model_validator(mode="after")
    def _post_init(self) -> "ComfyUITarget":
        if not self.model:
            raise ValueError("Each comfyui_target must declare a non-empty `model`.")
        if not self.name:
            object.__setattr__(self, "name", self.alias or self.model)
        if "@" in self.name:
            raise ValueError(
                f"comfyui_target name {self.name!r} contains '@'; reserved separator"
            )
        return self

    @property
    def display(self) -> str:
        return self.alias or self.model

    @property
    def models(self) -> List[ModelEntry]:
        return [ModelEntry(name=self.model, alias=self.alias, upstream_id=self.upstream_id)]

    @property
    def display_models(self) -> List[str]:
        return [self.display]

    def entry_for(self, display_name: str) -> Optional[ModelEntry]:
        for candidate in (self.display, self.model):
            match = _find_configured_model(display_name, [candidate])
            if match is not None:
                return self.models[0]
        return None

    def matching_model(self, display_name: str) -> Optional[str]:
        entry = self.entry_for(display_name)
        return entry.display if entry is not None else None

    def resolve_model(self, display_name: str) -> str:
        entry = self.entry_for(display_name)
        if entry is None:
            return display_name
        return entry.wire_id

    def serves(self, display_name: str) -> bool:
        return self.entry_for(display_name) is not None

    def workflow_config(self) -> Dict[str, Any]:
        keys = [
            "preset",
            "bindings",
            "static_inputs",
            "text_to_image_workflow_path",
            "image_to_image_workflow_path",
            "diffusion_model",
            "diffusion_weight_dtype",
            "text_encoder_model",
            "text_encoder_type",
            "text_encoder_device",
            "vae_model",
            "default_shift",
            "output_prefix",
            "image_upscale_method",
            "image_crop",
            "prompt_timeout_seconds",
            "poll_interval_seconds",
            "unet_node_id",
            "clip_node_id",
            "vae_node_id",
            "prompt_node_id",
            "latent_node_id",
            "sampling_node_id",
            "ksampler_node_id",
            "save_image_node_id",
            "load_image_node_id",
            "image_scale_node_id",
        ]
        return {key: getattr(self, key) for key in keys}


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class ExposureEntry(BaseModel):
    """One entry in an interface's ``exposed_models`` list.

    References a ``(target, model)`` pair (where ``model`` is the
    source-level display name, i.e. alias-or-name). Optional ``alias``
    overrides the client-facing public id; without it the public id is
    the composite ``<model>@<target>`` string.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    target: str
    alias: Optional[str] = None

    @field_validator("model", "target")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("exposure entry 'model' and 'target' must be non-empty")
        return v

    @field_validator("alias")
    @classmethod
    def _alias_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @property
    def composite_id(self) -> str:
        """The pre-alias ``model@target`` identifier."""
        return f"{self.model}@{self.target}"

    @property
    def public_id(self) -> str:
        """Client-facing id on the owning interface."""
        return self.alias or self.composite_id


def _coerce_exposure_entries(value: Any) -> List[Dict[str, Any]]:
    """Normalise an ``exposed_models`` field into ExposureEntry dicts."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("'exposed_models' must be a list")
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
        elif isinstance(item, ExposureEntry):
            out.append(item.model_dump())
        elif isinstance(item, str):
            # Allow the shorthand "model@target" string.
            if "@" not in item:
                raise ValueError(
                    f"exposed_models string entry {item!r} must be 'model@target'"
                )
            model, _, target = item.rpartition("@")
            if not model or not target:
                raise ValueError(
                    f"exposed_models string entry {item!r} is not a valid 'model@target'"
                )
            out.append({"model": model, "target": target})
        else:
            raise ValueError(
                f"each exposed_models[] entry must be an object or 'model@target' "
                f"string, got {type(item).__name__}"
            )
    return out


class _InterfaceBase(BaseModel):
    """Shared interface fields (Ollama-compatible vs API-compatible)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    host: str = "127.0.0.1"
    port: int
    access_tokens: List[str] = Field(default_factory=list)
    exposed_models: List[ExposureEntry] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("interface 'name' must be non-empty")
        return v

    @field_validator("access_tokens")
    @classmethod
    def _dedupe_tokens(cls, v: List[str]) -> List[str]:
        seen: Dict[str, None] = {}
        for tk in v:
            if tk and tk not in seen:
                seen[tk] = None
        return list(seen.keys())

    @field_validator("exposed_models", mode="before")
    @classmethod
    def _coerce_exposed(cls, v: Any) -> Any:
        return _coerce_exposure_entries(v)

    @model_validator(mode="after")
    def _check_unique_public_ids(self) -> "_InterfaceBase":
        seen: Dict[str, ExposureEntry] = {}
        dupes: List[str] = []
        for e in self.exposed_models:
            pid = e.public_id
            if pid in seen:
                dupes.append(pid)
            else:
                seen[pid] = e
        if dupes:
            raise ValueError(
                f"interface {self.name!r}: duplicate public model ids "
                f"{sorted(set(dupes))} in exposed_models. Each entry must produce "
                f"a unique client-facing id (alias if set, else model@target)."
            )
        return self

    @property
    def auth_required(self) -> bool:
        return bool(self.access_tokens)

    def is_valid_token(self, token: str) -> bool:
        if not token:
            return False
        return token in self.access_tokens

    def exposure_for_public_id(self, public_id: str) -> Optional[ExposureEntry]:
        for e in self.exposed_models:
            if e.public_id == public_id:
                return e
        # Tagless Ollama-style match: client "foo@target" should also match
        # exposed "foo:latest@target" (and vice versa) when the entry has no
        # explicit alias. Split on '@' so the tagless matcher only sees the
        # display-name half.
        if "@" in public_id:
            req_display, _, req_target = public_id.rpartition("@")
            same_target = [
                e for e in self.exposed_models
                if not e.alias and e.target == req_target
            ]
            displays = [e.model for e in same_target]
            match = _find_configured_model(req_display, displays)
            if match is not None:
                for e in same_target:
                    if e.model == match:
                        return e
        return None

    def public_ids(self) -> List[str]:
        return [e.public_id for e in self.exposed_models]


class OllamaInterface(_InterfaceBase):
    """An Ollama-compatible listener (/api/* and /v1/chat/completions)."""

    port: int = 21434


class ApiInterface(_InterfaceBase):
    """An OpenAI / Anthropic-compatible listener (/v1/*)."""

    port: int = 21435


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


_REMOVED_TOP_LEVEL_KEYS = {
    "host": "ollama_interfaces[*].host",
    "port": "ollama_interfaces[*].port",
    "external_host": "api_interfaces[*].host",
    "external_port": "api_interfaces[*].port",
    "external_access_tokens": "api_interfaces[*].access_tokens",
    "internal_exposed_models": "ollama_interfaces[*].exposed_models",
    "external_exposed_models": "api_interfaces[*].exposed_models",
    "upstreams": "anthropic_upstreams",
}


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # -- Sources ---------------------------------------------------------
    anthropic_upstreams: List[AnthropicUpstream] = Field(default_factory=list)
    openai_upstreams: List[OpenAIUpstream] = Field(default_factory=list)
    ollama_targets: List[OllamaTarget] = Field(default_factory=list)
    llama_cpp_defaults: LlamaCppDefaults = Field(default_factory=LlamaCppDefaults)
    llama_cpp_targets: List[LlamaCppTarget] = Field(default_factory=list)
    comfyui_targets: List[ComfyUITarget] = Field(default_factory=list)

    # -- Interfaces ------------------------------------------------------
    ollama_interfaces: List[OllamaInterface] = Field(
        default_factory=lambda: [OllamaInterface(name="ollama", port=21434)]
    )
    api_interfaces: List[ApiInterface] = Field(default_factory=list)

    # -- Runtime knobs ---------------------------------------------------
    advertised_version: str = "0.6.4"
    default_max_tokens: int = 4096
    timeout_seconds: float = 300.0
    use_system_proxy: bool = False
    enforce_context_limit: bool = True
    model_profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # -- Admin UI listener ----------------------------------------------
    admin_enabled: bool = True
    admin_host: str = "127.0.0.1"
    admin_port: Optional[int] = 21433

    # -- Dashboard listener ---------------------------------------------
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: Optional[int] = 21432
    dashboard_sample_interval_seconds: float = 10.0
    dashboard_retention_seconds: float = 7 * 24 * 60 * 60
    dashboard_data_path: Optional[str] = "logs/dashboard_history.json"
    dashboard_model_reclaim_enabled: bool = False
    # Manual close-button uses a *separate*, looser idle threshold than the
    # automatic LRU reclaim path (VRAM_IDLE_RECLAIM_SECONDS = 60s). The
    # dashboard button is a user-driven decision so we trust the operator
    # after only a few seconds of idle time.
    dashboard_reclaim_idle_seconds: float = 20.0
    vram_low_free_reclaim_enabled: bool = True
    vram_low_free_threshold_mib: float = 200.0

    # -- Meta ------------------------------------------------------------
    config_path: str = ""

    @field_validator("model_profiles", mode="before")
    @classmethod
    def _coerce_model_profiles(cls, value: Any) -> Any:
        """Accept the new list-of-entries shape as well as the legacy dict.

        New canonical form (mirrors ``exposed_models``)::

            "model_profiles": [
              {"model": "qwen3", "context_length": 65536, ...},
              {"model": "deepseek-r1", "target": "deepseek", ...}
            ]

        Each entry's key is ``model`` (when no target) or
        ``model@target``. Older configs that still use a dict keyed by
        composite id keep working.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if not isinstance(value, list):
            raise ValueError(
                "'model_profiles' must be a list of {model, target?, ...} "
                "entries (or the legacy dict form)"
            )
        out: Dict[str, Dict[str, Any]] = {}
        for idx, raw in enumerate(value):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"model_profiles[{idx}] must be an object, got {type(raw).__name__}"
                )
            item = dict(raw)
            model = item.pop("model", None)
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"model_profiles[{idx}]: 'model' is required and must be a "
                    f"non-empty string"
                )
            target = item.pop("target", None)
            if target is not None and not isinstance(target, str):
                raise ValueError(
                    f"model_profiles[{idx}]: 'target' must be a string if set"
                )
            target = (target or "").strip()
            key = f"{model.strip()}@{target}" if target else model.strip()
            if key in out:
                raise ValueError(
                    f"model_profiles: duplicate entry for {key!r}"
                )
            out[key] = item
        return out

    @field_serializer("model_profiles")
    def _serialize_model_profiles(
        self, value: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Emit ``model_profiles`` as a list so the config round-trips in the
        new canonical shape after ``model_dump()`` / save-to-disk."""
        out: List[Dict[str, Any]] = []
        for composite, body in value.items():
            model, sep, target = composite.partition("@")
            entry: Dict[str, Any] = {"model": model}
            if sep and target:
                entry["target"] = target
            if isinstance(body, dict):
                for k, v in body.items():
                    entry[k] = v
            out.append(entry)
        return out

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_top_level(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = {
            k: v
            for k, v in data.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
        bad = [k for k in data if k in _REMOVED_TOP_LEVEL_KEYS]
        if bad:
            hints = ", ".join(f"{k} -> {_REMOVED_TOP_LEVEL_KEYS[k]}" for k in bad)
            raise ValueError(
                f"unsupported legacy top-level config keys: {bad}. Migrate manually: {hints}"
            )
        return data

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        # 1. Source name uniqueness (global, across all source kinds).
        all_source_names: List[str] = []
        for src in self.all_sources():
            all_source_names.append(src.name)
        dupes = sorted({n for n in all_source_names if all_source_names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"Duplicate source names across anthropic_upstreams/openai_upstreams/"
                f"ollama_targets/llama_cpp_targets/comfyui_targets: {dupes}. Source names are used as "
                f"the right side of composite model ids and must be globally unique."
            )

        # 2. Interface name uniqueness within each list.
        for label, items in (
            ("ollama_interfaces", self.ollama_interfaces),
            ("api_interfaces", self.api_interfaces),
        ):
            names = [it.name for it in items]
            inter_dupes = sorted({n for n in names if names.count(n) > 1})
            if inter_dupes:
                raise ValueError(
                    f"Duplicate interface names in {label}: {inter_dupes}"
                )

        # 3. Interface ports distinct across every listener fake_ollama owns.
        port_owners: Dict[int, str] = {}
        for it in self.ollama_interfaces:
            tag = f"ollama_interface[{it.name}]"
            if it.port in port_owners:
                raise ValueError(
                    f"{tag} port={it.port} conflicts with {port_owners[it.port]}"
                )
            port_owners[it.port] = tag
        for it in self.api_interfaces:
            tag = f"api_interface[{it.name}]"
            if it.port in port_owners:
                raise ValueError(
                    f"{tag} port={it.port} conflicts with {port_owners[it.port]}"
                )
            port_owners[it.port] = tag
        if self.admin_enabled and self.admin_port is not None:
            if self.admin_port in port_owners:
                raise ValueError(
                    f"admin_port={self.admin_port} conflicts with {port_owners[self.admin_port]}"
                )
            port_owners[self.admin_port] = "admin"
        if self.dashboard_enabled and self.dashboard_port is not None:
            if self.dashboard_port in port_owners:
                raise ValueError(
                    f"dashboard_port={self.dashboard_port} conflicts with "
                    f"{port_owners[self.dashboard_port]}"
                )
            port_owners[self.dashboard_port] = "dashboard"

        # 4. Duplicate base_url within llama_cpp_targets — each llama.cpp
        # server process must listen on a distinct address.
        seen_lc_urls: Dict[str, str] = {}
        for tgt in self.llama_cpp_targets:
            url = tgt.base_url.rstrip("/")
            if url in seen_lc_urls:
                raise ValueError(
                    f"llama_cpp_target {tgt.name!r} has the same base_url {url!r} as "
                    f"{seen_lc_urls[url]!r}. Each llama.cpp server process must listen "
                    f"on a distinct address."
                )
            seen_lc_urls[url] = tgt.name

        # 4b. KV-cache / flash-attention consistency, evaluated after
        # ``llama_cpp_defaults`` is folded in so a target that inherits
        # ``flash_attn`` from defaults is accepted.
        for tgt in self.llama_cpp_targets:
            effective = self.effective_llama_cpp_target(tgt)
            effective.validate_kv_cache_requires_flash_attn()
            effective.validate_kv_cache_types_match()

        # 5. Exposure entries reference real (target, model) pairs.
        all_composite_ids = set(self.all_source_composite_ids())
        for it in list(self.ollama_interfaces) + list(self.api_interfaces):
            for e in it.exposed_models:
                cid = e.composite_id
                if cid not in all_composite_ids:
                    raise ValueError(
                        f"interface {it.name!r}: exposed_models references unknown "
                        f"target/model {cid!r}. Available composite ids: "
                        f"{sorted(all_composite_ids)}"
                    )

        # 6. Cycle detection: any upstream whose base_url points back at any
        # interface this process listens on means /v1/messages would
        # recurse forever.
        self.detect_upstream_cycles()
        return self

    # -- Source enumeration ---------------------------------------------

    def all_sources(self) -> List[Any]:
        """All configured sources in routing order (local first, then remote)."""
        return [
            *self.ollama_targets,
            *self.llama_cpp_targets,
            *self.comfyui_targets,
            *self.openai_upstreams,
            *self.anthropic_upstreams,
        ]

    def all_source_composite_ids(self) -> List[str]:
        """Every ``<display>@<source>`` id any source advertises."""
        out: List[str] = []
        seen: Dict[str, None] = {}
        for src in self.all_sources():
            for entry in src.models:
                cid = f"{entry.display}@{src.name}"
                if cid not in seen:
                    seen[cid] = None
                    out.append(cid)
        return out

    # -- Cycle detection -------------------------------------------------

    def own_bind_endpoints(self) -> List[Tuple[str, int]]:
        """Return (host, port) tuples for every listener this process owns."""
        endpoints: List[Tuple[str, int]] = []
        for it in self.ollama_interfaces:
            endpoints.append((it.host, it.port))
        for it in self.api_interfaces:
            endpoints.append((it.host, it.port))
        if self.admin_enabled and self.admin_port is not None:
            endpoints.append((self.admin_host, self.admin_port))
        if self.dashboard_enabled and self.dashboard_port is not None:
            endpoints.append((self.dashboard_host, self.dashboard_port))
        return endpoints

    def detect_upstream_cycles(self) -> None:
        """Validate same-process forwarding chains by walking the model graph.

        The graph nodes are ``(interface_name, normalized_public_id)``
        pairs (Ollama-style ``:latest`` is folded so ``foo`` and
        ``foo:latest`` collide). An edge ``A -> B`` exists when interface
        ``A``'s exposure resolves to a source whose ``base_url`` points
        back at one of this process's own listeners — i.e. the request
        would re-enter us at interface ``B`` carrying the next hop's
        wire-side model id.

        We then run a DFS:

        * Any cycle raises ``ValueError`` (the request would loop
          forever).
        * Same-process back-edges that do **not** form a cycle (case 1
          in the README — different aliases at every hop) are logged as
          a WARNING and allowed; they describe a legitimate local
          fan-out for testing.

        The check intentionally does **not** resolve DNS. It only fires
        when an upstream URL has a literal host that matches (and a
        port that matches) one of this process's own bind endpoints.
        """
        own = self.own_bind_endpoints()
        if not own:
            return
        loopback_hosts = {"127.0.0.1", "0.0.0.0", "localhost", "::1", "::"}
        local_hosts = set(loopback_hosts) | _local_host_aliases()

        # Map each model-serving listener port to the set of host names
        # that resolve to "ourselves" plus the interface that owns it.
        accept_by_port: Dict[int, set[str]] = {}
        iface_by_port: Dict[int, _InterfaceBase] = {}
        for it in self.all_interfaces():
            hosts = accept_by_port.setdefault(it.port, set())
            if it.host in ("0.0.0.0", "::"):
                hosts |= local_hosts
            else:
                hosts.add(it.host)
                if it.host in loopback_hosts:
                    hosts |= loopback_hosts
            iface_by_port[it.port] = it
        # Admin / dashboard ports count as "own" listeners but never
        # accept model traffic — anyone pointing an upstream at them
        # is an outright misconfiguration.
        misroute_ports: set[int] = set()
        for host, port in own:
            if port in iface_by_port:
                continue
            misroute_ports.add(port)

        def _self_iface_for(base_url: str) -> Optional[_InterfaceBase]:
            parsed = urlparse(base_url or "")
            host = (parsed.hostname or "").lower()
            port = parsed.port
            if not host or port is None:
                return None
            if port in misroute_ports:
                # Distinguish from "not us" so we can raise below.
                return _MISROUTED_LISTENER  # type: ignore[return-value]
            allowed = accept_by_port.get(port)
            if allowed is None:
                return None
            if host in allowed:
                return iface_by_port.get(port)
            return None

        # Build a graph of (iface_name, normalized_wire_id) nodes.
        #
        # Node identity is "the model id this iface receives that
        # selects this exposure". We use ``alias or model`` (folded
        # against ``:latest``) because:
        #
        # * Inbound clients reach an exposure by sending its alias if
        #   set, or its bare ``model`` (tagless fallback) otherwise.
        # * When an upstream loops back to one of our interfaces it
        #   sends the *wire_id* of its source's matching entry — which
        #   for case 1 (alias chain) lines up with the next exposure's
        #   alias, and for case 2 (same name) lines up with the next
        #   exposure's ``model``.
        #
        # Two exposures on the same iface can share an incoming key,
        # so we use an adjacency *list* — a cycle exists if **any**
        # path through that node loops.
        def _entry_key(exp: "ExposureEntry") -> str:
            return _normalize_public_id(exp.alias or exp.model)

        edges: Dict[Tuple[str, str], List[Optional[Tuple[str, str]]]] = {}
        same_proc_edges: List[Tuple[Tuple[str, str], Tuple[str, str], str]] = []
        for it in self.all_interfaces():
            for exposure in it.exposed_models:
                node = (it.name, _entry_key(exposure))
                edges.setdefault(node, [])
                src = self.source_by_name(exposure.target)
                if src is None:
                    edges[node].append(None)
                    continue
                base_url = getattr(src, "base_url", "") or ""
                inner = _self_iface_for(base_url)
                if inner is _MISROUTED_LISTENER:
                    raise ValueError(
                        f"cycle detected: source {src.name!r} base_url={base_url!r} "
                        f"points at this fake_ollama process's admin/dashboard "
                        f"listener, which does not serve model traffic. Remove the "
                        f"upstream or fix the URL."
                    )
                if inner is None:
                    edges[node].append(None)
                    continue
                model_entry = src.entry_for(exposure.model)
                next_pid = (
                    model_entry.wire_id if model_entry is not None else exposure.model
                )
                next_node = (inner.name, _normalize_public_id(next_pid))
                edges[node].append(next_node)
                same_proc_edges.append((node, next_node, src.name))

        if not same_proc_edges:
            # Fast path: nothing self-references, no graph walk needed.
            return

        # DFS with coloring to find cycles. Adjacency entries that are
        # ``None`` are dead-ends (external upstream); missing
        # destination nodes are also dead-ends (the next hop has no
        # matching exposure -> would 404 at runtime).
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[Tuple[str, str], int] = {n: WHITE for n in edges}
        path: List[Tuple[str, str]] = []

        def _visit(n: Tuple[str, str]) -> None:
            c = color.get(n, WHITE)
            if c == GRAY:
                idx = path.index(n)
                loop = path[idx:] + [n]
                pretty = " -> ".join(f"{ifn}::{pid}" for ifn, pid in loop)
                raise ValueError(
                    f"cycle detected in model-forwarding graph: {pretty}. "
                    f"At least one hop reuses the same public model id on the "
                    f"same interface; give one exposure a distinct alias to "
                    f"break the loop."
                )
            if c == BLACK:
                return
            if n not in edges:
                color[n] = BLACK
                return
            color[n] = GRAY
            path.append(n)
            for nxt in edges[n]:
                if nxt is not None:
                    _visit(nxt)
            path.pop()
            color[n] = BLACK

        for node in list(edges):
            _visit(node)

        # No cycle, but there is at least one same-process back-edge.
        # That is the "linear chain via aliases" case 1 in the README.
        # Warn so operators notice they have a self-referential config.
        for src_node, dst_node, src_name in same_proc_edges:
            _LOG.warning(
                "self-referential upstream is linear (no cycle): interface "
                "%r exposure %r -> source %r -> interface %r exposure %r. "
                "Keep aliases distinct on every hop to keep the chain acyclic.",
                src_node[0],
                src_node[1],
                src_name,
                dst_node[0],
                dst_node[1],
            )

    # -- Routing ---------------------------------------------------------

    def source_by_name(self, name: str) -> Optional[Any]:
        for src in self.all_sources():
            if src.name == name:
                return src
        return None

    def backend_by_name(self, name: str) -> Optional["Backend"]:
        src = self.source_by_name(name)
        if src is None:
            return None
        return Backend.from_source(src, self)

    def interface_by_name(self, name: str) -> Optional[_InterfaceBase]:
        for it in self.ollama_interfaces:
            if it.name == name:
                return it
        for it in self.api_interfaces:
            if it.name == name:
                return it
        return None

    def all_interfaces(self) -> List[_InterfaceBase]:
        return [*self.ollama_interfaces, *self.api_interfaces]

    def resolve_request(
        self, requested: str, *, interface_name: str
    ) -> Tuple["Backend", str]:
        """Resolve a client-supplied model id on a specific interface.

        Returns ``(backend, display_model)`` where ``display_model`` is the
        source-level display name (alias_or_name). Callers feed it to
        ``backend.source.resolve_model`` to get the wire-side id.
        """
        iface = self.interface_by_name(interface_name)
        if iface is None:
            raise ValueError(f"unknown interface {interface_name!r}")
        entry = iface.exposure_for_public_id(requested)
        if entry is None:
            ids = iface.public_ids()
            raise ValueError(
                f"model {requested!r} is not exposed on interface "
                f"{interface_name!r}; available: {ids}"
            )
        backend = self.backend_by_name(entry.target)
        if backend is None:
            raise ValueError(
                f"interface {interface_name!r} exposes model {requested!r} via "
                f"target {entry.target!r} which is not configured (config drift)"
            )
        if not backend.serves(entry.model):
            raise ValueError(
                f"interface {interface_name!r} exposes model {requested!r} but "
                f"target {entry.target!r} does not serve display name {entry.model!r}"
            )
        return backend, entry.model

    # -- Profile lookup --------------------------------------------------

    def profile_for(self, key: str) -> ModelProfile:
        """Look up a profile by ``model@target`` then fall back to bare name."""
        candidates: List[str] = [key]
        if "@" in key:
            display, _, _ = key.rpartition("@")
            if display:
                candidates.append(display)
                if ":" in display:
                    candidates.append(_model_base(display))
        elif ":" in key:
            candidates.append(_model_base(key))
        for k in candidates:
            if k in self.model_profiles:
                return ModelProfile.from_dict(self.model_profiles[k])
            match = _find_configured_model(k, self.model_profiles.keys())
            if match is not None:
                return ModelProfile.from_dict(self.model_profiles[match])
        return ModelProfile(
            capabilities=list(DEFAULT_CAPABILITIES),
            context_length=DEFAULT_CONTEXT_LENGTH,
            max_output_tokens=None,
        )

    # -- llama.cpp defaults convenience ---------------------------------

    def effective_llama_cpp_target(self, target: LlamaCppTarget) -> LlamaCppTarget:
        return target.with_defaults(self.llama_cpp_defaults)

    # -- Listener helpers -----------------------------------------------

    @property
    def admin_listener_enabled(self) -> bool:
        return self.admin_enabled and self.admin_port is not None

    @property
    def dashboard_listener_enabled(self) -> bool:
        return self.dashboard_enabled and self.dashboard_port is not None


# ---------------------------------------------------------------------------
# Unified backend view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """A protocol-tagged thin façade over one configured source."""

    name: str
    protocol: str  # "anthropic" | "openai" | "ollama" | "comfyui"
    kind: str  # "remote" | "local"
    base_url: str
    auth_token: str
    source: Any

    @classmethod
    def from_source(cls, src: Any, settings: Settings) -> "Backend":
        if isinstance(src, AnthropicUpstream):
            return cls(
                name=src.name,
                protocol="anthropic",
                kind="remote",
                base_url=src.base_url,
                auth_token=src.auth_token,
                source=src,
            )
        if isinstance(src, OpenAIUpstream):
            return cls(
                name=src.name,
                protocol="openai",
                kind="remote",
                base_url=src.base_url,
                auth_token=src.auth_token,
                source=src,
            )
        if isinstance(src, OllamaTarget):
            kind = "local" if (src.auto_start or src.stop_command) else "remote"
            return cls(
                name=src.name,
                protocol="ollama",
                kind=kind,
                base_url=src.base_url,
                auth_token=src.auth_token,
                source=src,
            )
        if isinstance(src, LlamaCppTarget):
            effective = settings.effective_llama_cpp_target(src)
            return cls(
                name=effective.name,
                protocol="openai",
                kind="local",
                base_url=effective.base_url,
                auth_token=effective.auth_token,
                source=effective,
            )
        if isinstance(src, ComfyUITarget):
            kind = "local" if (src.auto_start or src.stop_command) else "remote"
            return cls(
                name=src.name,
                protocol="comfyui",
                kind=kind,
                base_url=src.base_url,
                auth_token=src.auth_token,
                source=src,
            )
        raise TypeError(f"cannot build Backend from {type(src).__name__}")

    @property
    def models(self) -> List[str]:
        return list(self.source.display_models)

    def serves(self, display_name: str) -> bool:
        return self.source.serves(display_name)

    def resolve_model(self, display_name: str) -> str:
        return self.source.resolve_model(display_name)

    @property
    def supports_lifecycle(self) -> bool:
        return self.kind == "local"

    @property
    def auto_start(self) -> bool:
        if not self.supports_lifecycle:
            return False
        return bool(getattr(self.source, "auto_start", False))


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


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
    data = _read_json(resolved)
    settings = Settings(**data)
    if resolved is not None:
        settings = settings.model_copy(update={"config_path": str(resolved)})
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def _local_host_aliases() -> set[str]:
    """Best-effort list of host strings that name this machine.

    Used by ``detect_upstream_cycles`` to decide whether an upstream URL
    points at a 0.0.0.0-bound listener owned by this process. We avoid
    DNS resolution; this only inspects the local host's own names.
    """
    out: set[str] = set()
    try:
        out.add(socket.gethostname().lower())
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            try:
                ipaddress.ip_address(ip)
            except ValueError:
                continue
            out.add(ip.lower())
    except (OSError, socket.gaierror):
        pass
    return out


# ---------------------------------------------------------------------------
# Token estimation (rough – used only for the cost guardrail)
# ---------------------------------------------------------------------------


def estimate_tokens_from_anthropic_payload(body: Dict[str, Any]) -> int:
    chars = 0
    images = 0

    def add_text(value: Any) -> None:
        nonlocal chars
        if value is not None:
            chars += len(str(value))

    def add_json(value: Any) -> None:
        nonlocal chars
        if value is not None:
            chars += len(json.dumps(value, ensure_ascii=False, sort_keys=True))

    def add_content(content: Any) -> None:
        nonlocal images
        if isinstance(content, str):
            add_text(content)
            return
        if not isinstance(content, Iterable):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                add_text(block.get("text", ""))
            elif btype == "thinking":
                add_text(block.get("thinking", ""))
            elif btype == "image":
                images += 1
            elif btype == "document":
                source = block.get("source") or {}
                if isinstance(source, dict):
                    if source.get("type") == "text":
                        add_text(source.get("data", ""))
                    else:
                        images += 1
            elif btype == "tool_result":
                add_content(block.get("content", ""))
            elif btype == "tool_use":
                add_text(block.get("name", ""))
                add_json(block.get("input") or {})

    sys = body.get("system")
    if isinstance(sys, str):
        add_text(sys)
    elif isinstance(sys, list):
        for block in sys:
            if isinstance(block, dict) and block.get("type") == "text":
                add_text(block.get("text", ""))
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            add_content(msg.get("content"))
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        add_text(tool.get("name", ""))
        add_text(tool.get("description", ""))
        add_json(tool.get("input_schema") or {})
    add_json(body.get("tool_choice"))
    add_json(body.get("thinking"))
    overhead = 4 * len(body.get("messages") or [])
    return math.ceil(chars / 3) + overhead + images * 1500
