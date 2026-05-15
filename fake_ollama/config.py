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

    def exposes(self, display_name: str) -> bool:
        configured = self.matching_model(display_name)
        if not configured:
            return False
        if self.expose_external is None:
            return True
        return _find_configured_model(configured, self.expose_external) is not None


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
    # Subset of ``models`` visible on the EXTERNAL reverse-proxy surface
    # (/v1/models, /v1/messages). Same semantics as ``Upstream.expose_external``:
    #   None  -> expose all (default; back-compat).
    #   []    -> expose nothing externally (target becomes internal-only).
    #   [...] -> expose only the listed display names.
    expose_external: Optional[List[str]] = None
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

    def exposes(self, display_name: str) -> bool:
        configured = self.matching_model(display_name)
        if not configured:
            return False
        if self.expose_external is None:
            return True
        return _find_configured_model(configured, self.expose_external) is not None


class LlamaCppDefaults(BaseModel):
    """Defaults inherited by each llama.cpp target unless overridden."""

    expose_external: Optional[bool] = None
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
    expose_external: Optional[bool] = None

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

        legacy_expose = out.get("expose_external")
        if isinstance(legacy_expose, list):
            model = str(out.get("model") or "")
            out["expose_external"] = bool(
                model and _find_configured_model(model, legacy_expose) is not None
            )
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
                "expose_external": (
                    self.expose_external
                    if self.expose_external is not None
                    else defaults.expose_external
                ),
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

    def exposes(self, display_name: str) -> bool:
        configured = self.matching_model(display_name)
        if not configured:
            return False
        if self.expose_external is None:
            return True
        return bool(self.expose_external)


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
    ollama_targets: List[OllamaTarget] = Field(default_factory=list)
    llama_cpp_defaults: LlamaCppDefaults = Field(default_factory=LlamaCppDefaults)
    llama_cpp_targets: List[LlamaCppTarget] = Field(default_factory=list)
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
        llama_target_names = [t.name for t in self.llama_cpp_targets]
        if len(set(llama_target_names)) != len(llama_target_names):
            raise ValueError(f"Duplicate llama_cpp_target names: {llama_target_names}")
        llama_models = [t.model for t in self.llama_cpp_targets]
        if len(set(llama_models)) != len(llama_models):
            raise ValueError(f"Duplicate llama_cpp_target models: {llama_models}")
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

    @property
    def reverse_proxy_models(self) -> List[str]:
        """Local target model names visible on /v1/* (subject to expose_external)."""
        seen: Dict[str, None] = {}
        for t in self.ollama_targets:
            allowed = t.models if t.expose_external is None else [
                m for m in t.models if m in set(t.expose_external)
            ]
            for m in allowed:
                if m not in seen:
                    seen[m] = None
        for raw_t in self.llama_cpp_targets:
            t = self.effective_llama_cpp_target(raw_t)
            if t.expose_external is False:
                continue
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

        Both ollama_targets and upstreams honour their own
        ``expose_external`` whitelist (None = expose all, [] = none,
        [...] = subset).
        """
        for t in self.ollama_targets:
            if t.exposes(display_name):
                return True
        for raw_t in self.llama_cpp_targets:
            t = self.effective_llama_cpp_target(raw_t)
            if t.exposes(display_name):
                return True
        for up in self.upstreams:
            if up.exposes(display_name):
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

    def llama_cpp_target_for(self, display_name: str):
        """Return the LlamaCppTarget that should serve the given display name,
        or ``None`` if no target serves it."""
        for t in self.llama_cpp_targets:
            if t.serves(display_name):
                return t
        return None

    def effective_llama_cpp_target(self, target: LlamaCppTarget) -> LlamaCppTarget:
        return target.with_defaults(self.llama_cpp_defaults)

    def profile_for(self, display_name: str) -> ModelProfile:
        raw = self.model_profiles.get(display_name)
        if raw is None:
            key = _find_configured_model(display_name, self.model_profiles.keys())
            if key is not None:
                raw = self.model_profiles.get(key)
        if raw is None and ":" in display_name:
            raw = self.model_profiles.get(_model_base(display_name))
        if raw is None:
            for up in self.upstreams:
                configured = up.matching_model(display_name)
                if configured and configured in self.model_profiles:
                    raw = self.model_profiles[configured]
                    break
        if raw is None:
            for target in self.ollama_targets:
                configured = target.matching_model(display_name)
                if configured and configured in self.model_profiles:
                    raw = self.model_profiles[configured]
                    break
        if raw is None:
            for target in self.llama_cpp_targets:
                configured = target.matching_model(display_name)
                if configured and configured in self.model_profiles:
                    raw = self.model_profiles[configured]
                    break
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
    "FAKE_OLLAMA_ADMIN_HOST": ("admin_host", str),
    "FAKE_OLLAMA_ADMIN_PORT": ("admin_port", int),
    "FAKE_OLLAMA_DASHBOARD_ENABLED": ("dashboard_enabled", _parse_bool),
    "FAKE_OLLAMA_DASHBOARD_HOST": ("dashboard_host", str),
    "FAKE_OLLAMA_DASHBOARD_PORT": ("dashboard_port", int),
    "FAKE_OLLAMA_DASHBOARD_SAMPLE_INTERVAL_SECONDS": (
        "dashboard_sample_interval_seconds",
        float,
    ),
    "FAKE_OLLAMA_DASHBOARD_RETENTION_SECONDS": ("dashboard_retention_seconds", float),
    "FAKE_OLLAMA_DASHBOARD_DATA_PATH": ("dashboard_data_path", str),
    "FAKE_OLLAMA_VRAM_LOW_FREE_RECLAIM_ENABLED": (
        "vram_low_free_reclaim_enabled",
        _parse_bool,
    ),
    "FAKE_OLLAMA_VRAM_LOW_FREE_THRESHOLD_MIB": (
        "vram_low_free_threshold_mib",
        float,
    ),
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
