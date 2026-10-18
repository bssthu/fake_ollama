"""Optional StartLux launch planning; importing this module needs no ML packages.

This module does not install packages, download models, start processes or infer
model capabilities from model names. The companion delegates inference to the
official package in the operator-selected Python environment.
"""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from urllib.parse import urlsplit

STARTLUX_REFERENCE_VERSION = "1.1.0"
STARTLUX_REPOSITORY = "https://github.com/StartLuxLabs/StartLux-Decision"
STARTLUX_DEPLOYMENT = STARTLUX_REPOSITORY + "/blob/main/docs/deployment.md"
LLAMA_SYSTEMONE_RELEASE = "https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0"
VLLM_SYSTEMONE_PR = "https://github.com/vllm-project/vllm/pull/59299"
VLLM_031_RELEASE = "https://github.com/vllm-project/vllm/releases/tag/v0.31.0"


class StartLuxSetupError(ValueError):
    """Actionable local configuration or optional dependency problem."""


def validate_local_model_directory(model_path: str | None, *, cwd: str | None = None) -> Path:
    """Resolve local metadata only; the official package validates its semantics."""
    if not isinstance(model_path, str) or not model_path.strip() or "\x00" in model_path:
        raise StartLuxSetupError("StartLux requires model_path to be your downloaded local model directory.")
    if "://" in model_path:
        raise StartLuxSetupError("StartLux model_path must be a local directory, not a URL or model repository ID.")
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path(cwd or Path.cwd()).expanduser() / path
    path = path.resolve()
    if not path.is_dir():
        raise StartLuxSetupError(
            f"StartLux model directory does not exist: {path}. Supply your local download; no model is downloaded automatically."
        )
    for filename in ("config.json", "decision_config.json"):
        try:
            with (path / filename).open(encoding="utf-8") as source:
                metadata = json.load(source)
        except (OSError, ValueError) as exc:
            raise StartLuxSetupError(
                f"StartLux needs readable {filename} in {path}; restore the model's original local metadata."
            ) from exc
        if not isinstance(metadata, dict):
            raise StartLuxSetupError(f"StartLux {filename} must contain a JSON object.")
    return path


def _managed_address(base_url: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(base_url)
        host, port = parsed.hostname, parsed.port
        loopback = host == "localhost" or bool(host and ipaddress.ip_address(host).is_loopback)
    except (ValueError, TypeError):
        loopback = False
    if not loopback or parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise StartLuxSetupError("Managed StartLux base_url must be a loopback HTTP URL, e.g. http://127.0.0.1:8090.")
    if parsed.path not in ("", "/"):
        raise StartLuxSetupError("Managed StartLux base_url must not contain a path; use endpoint_path for the API route.")
    if port is None or not 1 <= port <= 65535:
        raise StartLuxSetupError("Managed StartLux base_url needs an explicit port from 1 to 65535.")
    return host, port


def _python_executable(target: Any) -> str:
    value = getattr(target, "python_path", None) or sys.executable
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StartLuxSetupError("python_path must name the Python executable in the optional StartLux environment.")
    path = Path(value).expanduser()
    if not path.is_absolute() and ("/" in value or "\\" in value):
        path = Path(getattr(target, "cwd", None) or Path.cwd()).expanduser() / path
    if path.is_file():
        # Do not resolve symlinks: a venv Python symlink must retain its venv identity.
        return str(path.absolute())
    found = shutil.which(value)
    if found:
        return found
    raise StartLuxSetupError(f"StartLux Python executable was not found: {value}. Set python_path to its virtual environment Python.")


def _start_args(target: Any) -> list[str]:
    values = getattr(target, "start_args", None) or []
    if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) and v and "\x00" not in v for v in values):
        raise StartLuxSetupError("StartLux start_args must be a list of individual command-line strings, not a shell command.")
    owned = {"--model", "--backend", "--host", "--port", "--vllm-python", "--ssl-certfile", "--ssl-keyfile", "--check", "--help", "-h", "--"}
    for arg in values:
        if arg.split("=", 1)[0] in owned:
            raise StartLuxSetupError(f"StartLux start_args cannot override {arg.split('=', 1)[0]}; use the managed target fields.")
    return list(values)


def validate_startlux_setup(target: Any) -> None:
    """Cheap checks at managed start time; no optional imports or subprocesses."""
    validate_local_model_directory(getattr(target, "model_path", None), cwd=getattr(target, "cwd", None))
    _managed_address(getattr(target, "base_url", ""))
    _python_executable(target)
    backend = getattr(target, "runtime_backend", None) or "vllm"
    if backend not in ("vllm", "transformers"):
        raise StartLuxSetupError("Managed StartLux runtime_backend must be vllm or transformers; see docs for external llama.cpp/MLX setups.")
    if getattr(target, "dialect", "systemone") != "systemone":
        raise StartLuxSetupError("Official StartLux uses the systemone dialect for both /v1/systemone and its /v1/decisions alias.")
    args = _start_args(target)
    if backend != "vllm" and any(v.split("=", 1)[0] in ("--launch-vllm", "--vllm-url") for v in args):
        raise StartLuxSetupError("vLLM launch/URL options require runtime_backend='vllm'.")
    if any(v.split("=", 1)[0] == "--vllm-url" for v in args) and "--launch-vllm" in args:
        raise StartLuxSetupError("Choose an existing --vllm-url or --launch-vllm, not both.")


def build_startlux_argv(target: Any) -> list[str]:
    """Build an exec-style argv, never a shell string; safe for paths with spaces."""
    validate_startlux_setup(target)
    host, port = _managed_address(target.base_url)
    model = validate_local_model_directory(target.model_path, cwd=getattr(target, "cwd", None))
    backend = getattr(target, "runtime_backend", None) or "vllm"
    args = _start_args(target)
    if backend == "vllm" and not any(v.split("=", 1)[0] in ("--vllm-url", "--launch-vllm") for v in args):
        args.insert(0, "--launch-vllm")
    return [
        _python_executable(target), str(Path(__file__).with_name("startlux_companion.py").absolute()),
        "serve", "--model", str(model), "--backend", backend, "--host", host, "--port", str(port), *args,
    ]


def verified_runtime_support(runtime: str, version: str | None, dialect: str, *, endpoint_verified: bool = False) -> dict:
    """Dated release evidence, not a replacement for live capability checks.

    `endpoint_verified` must mean a successful decision-protocol capability
    probe, not merely /health or an HTTP 200. Custom/backported builds may pass
    that probe even when their upstream release did not ship the endpoint.
    `startlux_vllm` means vLLM behind the official wrapper, not native vLLM.
    """
    out = {"runtime": runtime, "version": version, "dialect": dialect, "status": "unverified",
           "errors": [], "warnings": [], "evidence": []}
    normalized = runtime.lower().replace(".", "_").replace("-", "_")
    if normalized in ("startlux", "startlux_vllm"):
        out["evidence"] = [STARTLUX_DEPLOYMENT]
        if dialect != "systemone":
            out["status"] = "unsupported"
            out["errors"].append("Official StartLux serves the SystemOne schema at both decision paths; configure dialect='systemone'.")
            return out
        known = ("1.1.0",) if normalized == "startlux" else ("0.19.1", "0.31.0")
        if version in known:
            out["status"] = "supported"
        else:
            out["warnings"].append("This version is outside the dated upstream validation set; check the installed package's APIs and engine capabilities.")
        if normalized == "startlux_vllm":
            out["warnings"].append("Instant requires the official bidirectional plugin and whole-prompt launch settings; version alone cannot verify attention correctness.")
    elif normalized in ("llama_cpp", "llamacpp"):
        out["evidence"] = [LLAMA_SYSTEMONE_RELEASE]
        release = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version or "")
        if dialect != "systemone":
            out["warnings"].append("llama.cpp v0.6.0 documents the SystemOne schema; verify the configured dialect against this server.")
        elif release:
            if tuple(map(int, release.groups())) >= (0, 6, 0):
                out["status"] = "supported"
            else:
                out["status"] = "unsupported"
                out["errors"].append("Upstream llama.cpp first shipped /v1/systemone in v0.6.0 (2026-10-05); use that release or a capability-verified backport.")
        else:
            out["warnings"].append("A llama.cpp build tag or custom version cannot establish endpoint support; probe the decision protocol.")
    elif normalized == "vllm":
        out["evidence"] = [VLLM_SYSTEMONE_PR, VLLM_031_RELEASE]
        release = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version or "")
        if release and tuple(map(int, release.groups())) <= (0, 31, 0):
            out["status"] = "unsupported"
            out["errors"].append("Native vLLM /v1/systemone merged on 2026-10-08 after v0.31.0 (2026-10-05). Use a capability-verified newer/main build or the official StartLux wrapper.")
        else:
            out["warnings"].append("Native vLLM decision support must be probed on this build; a nightly/version label alone does not establish its schema or supported question types.")
    else:
        out["warnings"].append("No release assertion is available for this runtime; use its declared capabilities and decision-protocol probe.")
    if endpoint_verified:
        out["warnings"].extend(out["errors"])
        out["errors"] = []
        out["status"] = "supported"
    out["warnings"].append("Endpoint/version support does not verify model quality, calibration, image support or deployment-specific attention settings.")
    return out


def derive_startlux_capabilities(health: Any) -> dict[str, list[str]]:
    """Return only capability restrictions proven by official /health fields.

    This is not a complete capability declaration. Missing fields (including
    auth-redacted health responses) must not disable or enable extra features.
    AutoThink/v1 names do not authorize thinking or vision beyond configuration.
    Reference: startlux_decide.server.health in the official 1.1.0 package.
    """
    if not isinstance(health, dict) or health.get("status") != "ok":
        return {}
    restrictions: dict[str, list[str]] = {}
    if health.get("kind") == "instant" or health.get("prompt_attention") == "bidirectional":
        restrictions["think_modes"] = ["never"]
    if health.get("images") is False:
        restrictions["modalities"] = ["text"]
    return restrictions
