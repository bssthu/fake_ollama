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
import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

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


def _model_base(name: str) -> str:
    """Return the Ollama-style tagless model name."""
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


def _find_configured_model(display_name: str, models: Iterable[str]) -> Optional[str]:
    """Find the configured display name matching a client-supplied model.

    Ollama model names are ``model:tag`` where an omitted tag defaults to
    ``latest``. Therefore ``foo`` and ``foo:latest`` are equivalent, but
    ``foo`` must not match a different explicit tag such as ``foo:q4_K_M``.
    """
    model_list = list(models)
    if display_name in model_list:
        return display_name

    base = _model_base(display_name)
    if ":" in display_name:
        tag = display_name.split(":", 1)[1]
        if tag == "latest" and base in model_list:
            return base
        return None

    latest = f"{base}:latest"
    if latest in model_list:
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
        budget = data.get("thinking_budget_tokens") or data.get("thinking_budget") or DEFAULT_THINKING_BUDGET_TOKENS
        show = data.get("show_thinking")
        raw_vram = data.get("estimated_vram_gb")
        estimated_vram_gb = None
        if raw_vram not in (None, ""):
            try:
                parsed_vram = float(raw_vram)
                if parsed_vram > 0:
                    estimated_vram_gb = parsed_vram
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

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def matching_model(self, display_name: str) -> Optional[str]:
        return _find_configured_model(display_name, self.models)

    def resolve_model(self, display_name: str) -> str:
        configured = self.matching_model(display_name)
        for key in (display_name, configured, _model_base(display_name)):
            if key and key in self.model_map:
                return self.model_map[key]
        if configured:
            base = _model_base(configured)
            if base in self.model_map:
                return self.model_map[base]
            return configured
        if ":" in display_name:
            base = _model_base(display_name)
            if base in self.model_map:
                return self.model_map[base]
            return base
        return display_name

    def serves(self, display_name: str) -> bool:
        return self.matching_model(display_name) is not None


class OpenAIUpstream(Upstream):
    """An OpenAI-compatible remote upstream (OpenAI, DeepSeek, Together, …).

    The data shape and routing semantics are identical to
    :class:`Upstream`; only the wire format spoken to the upstream
    differs (POST ``/v1/chat/completions`` instead of ``/v1/messages``).
    Keeping it as a distinct Pydantic class lets the admin UI / config
    file declare which protocol an entry uses without resorting to a
    free-form ``protocol`` field on the unified backend list.
    """


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
    # Optional daemon lifecycle management. Leave ``auto_start`` false when
    # Ollama is installed as a separately managed service / desktop app.
    auto_start: bool = False
    start_command: Optional[str] = None
    stop_command: Optional[str] = None
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: float = 60.0
    health_path: str = "/api/version"
    cwd: Optional[str] = None

    model_config = {"extra": "ignore"}  # tolerate legacy ``api_token`` field

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: str) -> str:
        if not v:
            return "/api/version"
        return v if v.startswith("/") else "/" + v

    def matching_model(self, display_name: str) -> Optional[str]:
        return _find_configured_model(display_name, self.models)

    def resolve_model(self, display_name: str) -> str:
        configured = self.matching_model(display_name)
        for key in (display_name, configured, _model_base(display_name)):
            if key and key in self.model_map:
                return self.model_map[key]
        if configured:
            base = _model_base(configured)
            if base in self.model_map:
                return self.model_map[base]
            return configured
        return display_name

    def serves(self, display_name: str) -> bool:
        return self.matching_model(display_name) is not None


class LlamaCppDefaults(BaseModel):
    """Defaults inherited by each llama.cpp target unless overridden."""

    auto_start: bool = False
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: float = 120.0
    health_path: str = "/health"
    cwd: Optional[str] = None
    # Defaults for synthesised start commands. Targets that leave
    # ``start_command`` empty have one built from these (plus their own
    # model_path / mmproj_path / port etc.).
    binary_path: Optional[str] = None
    runtime_root: Optional[str] = None
    gpu_layers: Optional[int] = None
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None
    # Logical (-b) / physical (-ub) batch sizes. llama.cpp defaults are
    # 2048 / 512; leave None to inherit llama.cpp's default.
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    # Enable FlashAttention. Current llama.cpp expects a value:
    # -fa/--flash-attn [on|off|auto].
    flash_attn: Optional[bool] = None
    # KV cache quantisation (llama.cpp -ctk / -ctv). Default upstream is f16;
    # common values are f16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl. Lowering
    # these can cut KV-cache VRAM substantially at some quality cost.
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    extra_args: Optional[str] = None

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: str) -> str:
        if not v:
            return "/health"
        return v if v.startswith("/") else "/" + v


class LlamaCppTarget(BaseModel):
    """One llama.cpp server process exposed through fake-ollama's reverse proxy.

    llama.cpp server speaks OpenAI-compatible ``/v1/chat/completions`` and
    ``/v1/models``. A llama.cpp server process loads one model, so each target
    represents exactly one display model with its own port and lifecycle
    commands. Configure additional models as additional ``llama_cpp_targets``.
    """

    name: str = ""
    base_url: str = "http://127.0.0.1:8080"
    auth_token: str = ""
    model: str = ""
    model_alias: Optional[str] = None

    # Optional lifecycle management. If ``auto_start`` is true and the health
    # check fails, fake-ollama runs ``start_command`` before forwarding the
    # request. ``idle_timeout_seconds`` stops only processes started by this
    # fake-ollama instance unless ``stop_command`` is configured.
    auto_start: Optional[bool] = None
    start_command: Optional[str] = None
    stop_command: Optional[str] = None
    idle_timeout_seconds: Optional[float] = None
    startup_timeout_seconds: Optional[float] = None
    health_path: Optional[str] = None
    cwd: Optional[str] = None

    # Synthesised start_command parameters. Used only when ``start_command``
    # is not set: fake-ollama assembles a llama-server invocation from these
    # fields plus ``base_url`` (for host/port).
    binary_path: Optional[str] = None
    runtime_root: Optional[str] = None
    model_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    gpu_layers: Optional[int] = None
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None
    # See LlamaCppDefaults; unset on target = inherit defaults; defaults
    # unset = llama.cpp defaults (2048 / 512 / off / f16).
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    flash_attn: Optional[bool] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    extra_args: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_single_model_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        legacy_models = out.pop("models", None)
        if not out.get("model") and legacy_models is not None:
            if isinstance(legacy_models, list) and len(legacy_models) == 1:
                out["model"] = legacy_models[0]
            else:
                raise ValueError(
                    "Each llama_cpp_target must declare exactly one model. Replace "
                    "legacy `models` lists with a single `model` string and "
                    "configure one target per llama.cpp process."
                )

        legacy_map = out.pop("model_map", None)
        if not out.get("model_alias") and isinstance(legacy_map, dict):
            model = str(out.get("model") or "")
            base = _model_base(model) if model else ""
            for key in (model, base):
                if key and key in legacy_map:
                    out["model_alias"] = legacy_map[key]
                    break
            else:
                values = [v for v in legacy_map.values() if v]
                if len(values) == 1:
                    out["model_alias"] = values[0]
        return out

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("health_path")
    @classmethod
    def _normalise_health_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not v:
            return "/health"
        return v if v.startswith("/") else "/" + v

    @model_validator(mode="after")
    def _validate_single_model_process(self) -> "LlamaCppTarget":
        if not self.model:
            raise ValueError(
                "Each llama_cpp_target must declare a non-empty `model` because "
                "each llama.cpp server process loads one model. Configure a "
                "separate llama_cpp_targets entry with its own base_url, "
                "start_command, stop_command, and port for each model."
            )
        if not self.name:
            object.__setattr__(self, "name", self.model)
        return self

    @property
    def models(self) -> List[str]:
        return [self.model] if self.model else []

    def with_defaults(self, defaults: LlamaCppDefaults) -> "LlamaCppTarget":
        return self.model_copy(
            update={
                "auto_start": (
                    self.auto_start
                    if self.auto_start is not None
                    else defaults.auto_start
                ),
                "idle_timeout_seconds": (
                    self.idle_timeout_seconds
                    if self.idle_timeout_seconds is not None
                    else defaults.idle_timeout_seconds
                ),
                "startup_timeout_seconds": (
                    self.startup_timeout_seconds
                    if self.startup_timeout_seconds is not None
                    else defaults.startup_timeout_seconds
                ),
                "health_path": self.health_path or defaults.health_path,
                "cwd": self.cwd if self.cwd is not None else defaults.cwd,
                "binary_path": (
                    self.binary_path
                    if self.binary_path is not None
                    else defaults.binary_path
                ),
                "runtime_root": (
                    self.runtime_root
                    if self.runtime_root is not None
                    else defaults.runtime_root
                ),
                "gpu_layers": (
                    self.gpu_layers
                    if self.gpu_layers is not None
                    else defaults.gpu_layers
                ),
                "ctx_size": (
                    self.ctx_size
                    if self.ctx_size is not None
                    else defaults.ctx_size
                ),
                "parallel": (
                    self.parallel
                    if self.parallel is not None
                    else defaults.parallel
                ),
                "batch_size": (
                    self.batch_size
                    if self.batch_size is not None
                    else defaults.batch_size
                ),
                "ubatch_size": (
                    self.ubatch_size
                    if self.ubatch_size is not None
                    else defaults.ubatch_size
                ),
                "flash_attn": (
                    self.flash_attn
                    if self.flash_attn is not None
                    else defaults.flash_attn
                ),
                "cache_type_k": (
                    self.cache_type_k
                    if self.cache_type_k is not None
                    else defaults.cache_type_k
                ),
                "cache_type_v": (
                    self.cache_type_v
                    if self.cache_type_v is not None
                    else defaults.cache_type_v
                ),
                "extra_args": (
                    self.extra_args
                    if self.extra_args is not None
                    else defaults.extra_args
                ),
            }
        )

    def synthesize_start_command(self) -> Optional[str]:
        """Return ``start_command`` if set, otherwise build one from fields.

        Returns ``None`` only when no ``start_command`` is configured *and*
        ``model_path`` is empty (in which case fake-ollama cannot launch the
        server itself and falls back to the legacy "bring your own process"
        behaviour).
        """
        if self.start_command:
            return self.start_command
        argv = self.synthesize_start_argv()
        if argv is None:
            return None
        # Use platform-appropriate quoting: shlex.quote (POSIX single quotes)
        # would break on Windows because cmd.exe does not strip single
        # quotes and would refuse to launch the executable.
        return " ".join(_shell_quote(a) for a in argv)

    def synthesize_start_argv(self) -> Optional[List[str]]:
        """Return argv for launching ``llama-server`` directly (exec, no shell).

        Returns ``None`` if a user-provided ``start_command`` is set (we
        cannot safely split an arbitrary shell string into argv) or if no
        ``model_path`` is configured.

        Preferring exec over shell is important on Windows: ``cmd.exe /c``
        wraps the actual server in a launcher process whose lifecycle is
        decoupled from the child. When fake-ollama later wants to free
        VRAM by killing the server, the wrapper may already have exited
        (returncode set), leaving the real ``llama-server.exe`` orphaned
        and unkillable via the captured PID.
        """
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
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            self.model_path,
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
        if self.model_alias:
            argv += ["--alias", self.model_alias]
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
        """Return env dict for launching this target, or ``None`` if no
        adjustments are needed (caller will inherit os.environ).

        The CUDA-build llama.cpp release ships with a separate
        ``cudart-llama-bin-win-cuda-XX.X-x64`` folder containing
        ``cudart64_*.dll`` etc. Without that directory on PATH, llama-server
        fails to load CUDA and silently falls back to CPU (no VRAM growth).
        Mirrors the original PowerShell launcher's
        ``$env:PATH = \"$binRoot;$runtimeRoot;\" + $env:PATH``.
        """
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
        """Resolve ``binary_path`` to an actual executable.

        Users sometimes paste the extracted llama.cpp release folder (e.g.
        ``C:\\...\\llama-b8994-bin-win-cuda-13.1-x64``) instead of the
        ``llama-server[.exe]`` inside it. A bare directory passed as a shell
        command silently fails on Windows (``cmd.exe`` reports it as
        "is not recognized") and the caller would then wait the full
        ``startup_timeout_seconds`` with no useful diagnostic.
        """
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

    def matching_model(self, display_name: str) -> Optional[str]:
        return _find_configured_model(display_name, [self.model])

    def resolve_model(self, display_name: str) -> str:
        configured = self.matching_model(display_name)
        if configured and self.model_alias:
            return self.model_alias
        if configured:
            return configured
        return display_name

    def serves(self, display_name: str) -> bool:
        return self.matching_model(display_name) is not None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    # ---- Internal listener (Ollama-compatible /api/*) -------------------
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

    # ---- Admin listener (/admin/* only) ---------------------------------
    # The admin UI has no authentication, so it lives on its own localhost
    # listener by default instead of sharing the Ollama-compatible port.
    # Set admin_port=null to intentionally mount /admin on the internal
    # listener (legacy/single-port mode).
    admin_host: str = "127.0.0.1"
    admin_port: Optional[int] = 21433

    # ---- Dashboard listener (/dashboard/* only) ------------------------
    # Runtime metrics are exposed on their own localhost listener so the
    # unauthenticated dashboard does not share the admin or API ports.
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: Optional[int] = 21432
    dashboard_sample_interval_seconds: float = 10.0
    dashboard_retention_seconds: float = 7 * 24 * 60 * 60
    dashboard_data_path: Optional[str] = "logs/dashboard_history.json"
    dashboard_model_reclaim_enabled: bool = False

    # Low-VRAM safety monitor. Every dashboard_sample_interval_seconds it
    # checks nvidia-smi; if free VRAM is below the threshold, it asks the
    # coordinator to release eligible idle local models.
    vram_low_free_reclaim_enabled: bool = True
    vram_low_free_threshold_mib: float = 200.0

    advertised_version: str = "0.6.4"
    default_max_tokens: int = 4096
    timeout_seconds: float = 300.0
    use_system_proxy: bool = False
    enforce_context_limit: bool = True
    upstreams: List[Upstream] = Field(default_factory=list)
    openai_upstreams: List[OpenAIUpstream] = Field(default_factory=list)
    ollama_targets: List[OllamaTarget] = Field(default_factory=list)
    llama_cpp_defaults: LlamaCppDefaults = Field(default_factory=LlamaCppDefaults)
    llama_cpp_targets: List[LlamaCppTarget] = Field(default_factory=list)
    # Per-interface exposed-model whitelists. Each entry is a composite
    # ``model@target`` identifier; ``model`` here is the **display name**
    # the target/upstream advertises (i.e. an entry from its ``models``
    # list, not the wire-side id from ``model_map``). Default ``[]``
    # means nothing is exposed on that interface; a fresh config must
    # opt models in explicitly. ``None`` is reserved for "expose all"
    # if you ever want it back; the admin UI normalises empty to ``[]``.
    internal_exposed_models: List[str] = Field(default_factory=list)
    external_exposed_models: List[str] = Field(default_factory=list)
    model_profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Web admin UI (mounted at /admin). Set to false to disable entirely.
    admin_enabled: bool = True

    # Where the JSON config came from (empty string if no file was used).
    config_path: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        # All backend source names must be globally unique because they
        # form the right-hand side of the composite ``model@target``
        # identifier used by both /api/tags and routing. Allowing two
        # sources to share a name would make composite identifiers
        # ambiguous.
        all_source_names: List[str] = []
        for up in self.upstreams:
            all_source_names.append(up.name)
        for up in self.openai_upstreams:
            all_source_names.append(up.name)
        for t in self.ollama_targets:
            all_source_names.append(t.name)
        for t in self.llama_cpp_targets:
            all_source_names.append(t.name or t.model)
        dupes = sorted({n for n in all_source_names if all_source_names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"Duplicate source names across upstreams/openai_upstreams/"
                f"ollama_targets/llama_cpp_targets: {dupes}. Source names "
                f"are used as the target component of composite model IDs "
                f"(model@target) and must be globally unique."
            )
        # ``@`` is reserved as the composite-identifier separator.
        for name in all_source_names:
            if "@" in name:
                raise ValueError(
                    f"Source name {name!r} contains '@'; '@' is reserved "
                    f"as the model/target separator in composite IDs."
                )
        enabled_ports = {"internal": self.port}
        if self.external_port is not None:
            enabled_ports["external"] = self.external_port
        if self.admin_enabled and self.admin_port is not None:
            enabled_ports["admin"] = self.admin_port
        if self.dashboard_enabled and self.dashboard_port is not None:
            enabled_ports["dashboard"] = self.dashboard_port
        seen_ports: Dict[int, str] = {}
        for label, port in enabled_ports.items():
            if port in seen_ports:
                raise ValueError(
                    f"{label}_port={port} conflicts with {seen_ports[port]}_port; "
                    "internal, external, admin, and dashboard listeners must use distinct ports"
                )
            seen_ports[port] = label
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
    def admin_listener_enabled(self) -> bool:
        return self.admin_enabled and self.admin_port is not None

    @property
    def dashboard_listener_enabled(self) -> bool:
        return self.dashboard_enabled and self.dashboard_port is not None

    # -- Composite model identity ---------------------------------------
    #
    # Every model the user can talk to is identified globally by a
    # composite ``model@target`` string, where ``model`` is the display
    # name the source advertises and ``target`` is the source's ``name``.
    # The composite form is the *only* form clients may send (bare model
    # names are 400'd) and is what /api/tags / /v1/models expose. Inside
    # routing we split composite -> (target_name, model) and look the
    # backend up by target name directly.

    COMPOSITE_SEP: str = "@"

    @staticmethod
    def compose_model_id(model: str, target: str) -> str:
        return f"{model}@{target}"

    @staticmethod
    def split_composite(name: str) -> Optional[tuple[str, str]]:
        """Split ``model@target`` into ``(model, target)``; ``None`` if not composite."""
        if "@" not in name:
            return None
        model, _, target = name.rpartition("@")
        if not model or not target:
            return None
        return model, target

    def all_composite_ids(self) -> List[str]:
        """Every ``model@target`` advertised by any configured source.

        Order: ollama_targets → llama_cpp_targets → openai_upstreams →
        anthropic upstreams; within each source the source's own model
        order is preserved. Duplicates within one source are dropped.
        """
        out: List[str] = []
        seen: Dict[str, None] = {}
        for backend in self.backends:
            for model in backend.models:
                cid = self.compose_model_id(model, backend.name)
                if cid not in seen:
                    seen[cid] = None
                    out.append(cid)
        return out

    def exposed_composite_ids(self, surface: str) -> List[str]:
        """Composite ids the given surface (``internal`` / ``external``) exposes.

        Order matches ``all_composite_ids``; entries listed in the
        per-surface whitelist but not actually backed by any source are
        silently skipped.
        """
        if surface == "internal":
            allowed = set(self.internal_exposed_models)
        elif surface == "external":
            allowed = set(self.external_exposed_models)
        else:
            raise ValueError(f"unknown surface: {surface!r}")
        return [cid for cid in self.all_composite_ids() if cid in allowed]

    def is_exposed(self, surface: str, composite_id: str) -> bool:
        if surface == "internal":
            return composite_id in self.internal_exposed_models
        if surface == "external":
            return composite_id in self.external_exposed_models
        raise ValueError(f"unknown surface: {surface!r}")

    def is_valid_external_token(self, token: str) -> bool:
        """True iff ``token`` is in ``external_access_tokens``."""
        if not token:
            return False
        return token in self.external_access_tokens

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
        """All advertised composite model ids across every source."""
        return self.all_composite_ids()

    # -- Routing helpers -------------------------------------------------

    def effective_llama_cpp_target(self, target: LlamaCppTarget) -> LlamaCppTarget:
        return target.with_defaults(self.llama_cpp_defaults)

    def profile_for(self, composite_or_bare: str) -> ModelProfile:
        """Look up a profile by composite ``model@target`` then fall back.

        Lookup order:
        1. Exact composite id (most specific).
        2. Bare model name (generic default shared across all targets
           that serve this model).
        3. Tagless base of the bare model (Ollama-style; ``foo:latest``
           falls back to ``foo``).

        Returns the built-in defaults when nothing matches.
        """
        keys: List[str] = []
        split = self.split_composite(composite_or_bare)
        if split is not None:
            model, _target = split
            keys.append(composite_or_bare)
            keys.append(model)
            if ":" in model:
                keys.append(_model_base(model))
        else:
            keys.append(composite_or_bare)
            if ":" in composite_or_bare:
                keys.append(_model_base(composite_or_bare))
        for k in keys:
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

    # -- Unified backends view -------------------------------------------

    @property
    def backends(self) -> List["Backend"]:
        """Backends in routing priority order.

        Local backends (Ollama, then llama.cpp) come before remote
        Anthropic upstreams so the same display name on both a local
        target and a remote upstream resolves to the local one. This
        mirrors the precedence the legacy three-handler routing
        already implemented via ``ollama_target_for`` /
        ``llama_cpp_target_for`` being consulted before falling back to
        ``upstreams``.
        """
        out: List[Backend] = []
        for tgt in self.ollama_targets:
            out.append(Backend.from_ollama_target(tgt))
        for raw_t in self.llama_cpp_targets:
            out.append(
                Backend.from_llama_cpp_target(self.effective_llama_cpp_target(raw_t))
            )
        for up in self.openai_upstreams:
            out.append(Backend.from_openai_upstream(up))
        for up in self.upstreams:
            out.append(Backend.from_anthropic_upstream(up))
        return out

    def backend_by_name(self, target_name: str) -> Optional["Backend"]:
        """Look up a backend by its source name (the right side of ``model@target``)."""
        for backend in self.backends:
            if backend.name == target_name:
                return backend
        return None

    def resolve_request(
        self, requested: str, *, surface: str
    ) -> tuple["Backend", str]:
        """Resolve a client-supplied model id on ``surface``.

        ``requested`` must be a composite ``model@target`` string; bare
        model names are rejected with ``ValueError`` carrying a
        client-friendly message listing valid composite ids exposed on
        the surface. The returned tuple is ``(backend, display_model)``
        where ``display_model`` is the bare model name; callers feed it
        through ``backend.resolve_model`` to get the wire-side id.
        """
        split = self.split_composite(requested)
        if split is None:
            exposed = self.exposed_composite_ids(surface)
            raise ValueError(
                f"model {requested!r} must be specified as 'model@target'; "
                f"available on {surface}: {exposed}"
            )
        model, target_name = split
        backend = self.backend_by_name(target_name)
        if backend is None:
            raise ValueError(
                f"unknown target {target_name!r} in model id {requested!r}"
            )
        if not backend.serves(model):
            raise ValueError(
                f"target {target_name!r} does not serve model {model!r}"
            )
        composite_id = self.compose_model_id(model, target_name)
        if not self.is_exposed(surface, composite_id):
            raise ValueError(
                f"model {composite_id!r} is not exposed on {surface}"
            )
        return backend, model

    def backend_for(
        self, composite_id: str, *, surface: Optional[str] = None
    ) -> Optional["Backend"]:
        """Return the backend that should handle ``composite_id`` on ``surface``.

        ``composite_id`` is a ``model@target`` string. Returns ``None``
        when the target is unknown, does not serve the named model, or
        (when ``surface`` is given) is not exposed on that surface.
        """
        split = self.split_composite(composite_id)
        if split is None:
            return None
        model, target_name = split
        backend = self.backend_by_name(target_name)
        if backend is None or not backend.serves(model):
            return None
        if surface is not None and not self.is_exposed(surface, composite_id):
            return None
        return backend


# ---------------------------------------------------------------------------
# Unified backend view
# ---------------------------------------------------------------------------


_BackendSource = Any  # Upstream | OllamaTarget | LlamaCppTarget


@dataclass(frozen=True)
class Backend:
    """A protocol-tagged view over one backend declaration.

    ``Backend`` is intentionally a thin façade: routing code stays
    independent of which legacy field (``upstreams`` / ``ollama_targets``
    / ``llama_cpp_targets``) declared the backend, while still letting
    callers reach into ``source`` for protocol-specific fields (lifecycle
    commands, llama.cpp launch args, etc.).

    Tags:

    * ``protocol``: wire format spoken to this backend. ``anthropic`` for
      Anthropic Messages API upstreams, ``ollama`` for the Ollama
      ``/api/chat`` JSON format, ``openai`` for the OpenAI Chat
      Completions format that llama.cpp server and (future) remote
      OpenAI-compatible upstreams speak.
    * ``kind``: ``remote`` for HTTP-only backends fake-ollama only proxies
      to; ``local`` for backends fake-ollama may also own the process
      lifecycle of (health check, auto-start, idle stop). ``auto_start``
      on a ``remote`` backend is meaningless and ignored.
    """

    name: str
    protocol: str  # "anthropic" | "ollama" | "openai"
    kind: str  # "remote" | "local"
    base_url: str
    auth_token: str
    models: List[str]
    source: _BackendSource

    # -- Constructors ----------------------------------------------------

    @classmethod
    def from_anthropic_upstream(cls, up: "Upstream") -> "Backend":
        return cls(
            name=up.name,
            protocol="anthropic",
            kind="remote",
            base_url=up.base_url,
            auth_token=up.auth_token,
            models=list(up.models),
            source=up,
        )

    @classmethod
    def from_openai_upstream(cls, up: "OpenAIUpstream") -> "Backend":
        return cls(
            name=up.name,
            protocol="openai",
            kind="remote",
            base_url=up.base_url,
            auth_token=up.auth_token,
            models=list(up.models),
            source=up,
        )

    @classmethod
    def from_ollama_target(cls, tgt: "OllamaTarget") -> "Backend":
        # Ollama targets are usually local (``http://127.0.0.1:11434``)
        # but may point at a remote daemon; the practical differentiator
        # is whether fake-ollama owns the lifecycle (``auto_start``).
        kind = "local" if tgt.auto_start or tgt.stop_command else "remote"
        return cls(
            name=tgt.name,
            protocol="ollama",
            kind=kind,
            base_url=tgt.base_url,
            auth_token="",
            models=list(tgt.models),
            source=tgt,
        )

    @classmethod
    def from_llama_cpp_target(cls, tgt: "LlamaCppTarget") -> "Backend":
        return cls(
            name=tgt.name or tgt.model,
            protocol="openai",
            kind="local",
            base_url=tgt.base_url,
            auth_token=tgt.auth_token,
            models=list(tgt.models),
            source=tgt,
        )

    # -- Routing helpers -------------------------------------------------

    def serves(self, display_name: str) -> bool:
        return self.source.serves(display_name)

    def resolve_model(self, display_name: str) -> str:
        return self.source.resolve_model(display_name)

    # -- Lifecycle helpers ----------------------------------------------

    @property
    def supports_lifecycle(self) -> bool:
        """Whether ``auto_start`` / ``idle_timeout`` are meaningful here."""
        return self.kind == "local"

    @property
    def auto_start(self) -> bool:
        if not self.supports_lifecycle:
            return False
        return bool(getattr(self.source, "auto_start", False))


# ---------------------------------------------------------------------------
# Loader
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
    """Build a Settings object from JSON config.

    All runtime knobs live in the config file (or are set via the admin
    UI); the only environment variable consulted at load time is
    ``FAKE_OLLAMA_CONFIG`` which selects the config-file path. Use
    ``tests/conftest.py`` to inject test-only env if you need it.
    """
    resolved = _resolve_config_path(config_path)
    data = _read_json(resolved)
    settings = Settings(**data)
    if resolved is not None:
        settings = settings.model_copy(update={"config_path": str(resolved)})
    return settings


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
