"""Isolated entry point for the optional official StartLux package.

Run this FILE with the optional environment's Python. It intentionally has no
FastAPI/Pydantic or torch import and works from a normal fake-ollama wheel.
The official package alone owns prompts, calibration, AutoThink and attention.
"""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
import sys
from typing import Any

if __package__:
    from .decision_launch import STARTLUX_REFERENCE_VERSION, StartLuxSetupError, validate_local_model_directory, verified_runtime_support
else:  # Executed as a file in an independent virtual environment.
    from decision_launch import STARTLUX_REFERENCE_VERSION, StartLuxSetupError, validate_local_model_directory, verified_runtime_support


def offline_model_environment() -> None:
    """Model artifacts are supplied by the user; never fetch missing files."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")


def _disable_abbreviations(parser: argparse.ArgumentParser) -> None:
    parser.allow_abbrev = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _disable_abbreviations(child)


def inspect_official_package(argv: list[str]) -> tuple[Any, dict]:
    """Check installed metadata/API and local model metadata without inference."""
    offline_model_environment()
    try:
        installed_version = metadata.version("startlux-decide")
    except metadata.PackageNotFoundError as exc:
        raise StartLuxSetupError(
            "The selected Python has no startlux-decide package. Install the official StartLux-Decision package in a separate environment and set python_path to that environment's Python. See docs/decision-startlux.md."
        ) from exc
    if not isinstance(installed_version, str) or not installed_version.strip():
        raise StartLuxSetupError("Installed startlux-decide has no version metadata; reinstall the official package in the selected environment.")
    try:
        cli = importlib.import_module("startlux_decide.cli")
        modelcfg = importlib.import_module("startlux_decide.modelcfg")
    except ImportError as exc:
        raise StartLuxSetupError(f"startlux-decide {installed_version} cannot import its CLI/dependencies ({exc.name}); install its selected backend extra in this Python environment.") from exc
    if not all(callable(getattr(owner, name, None)) for owner, name in ((cli, "main"), (cli, "build_parser"), (modelcfg, "load_model_folder"))):
        raise StartLuxSetupError(f"startlux-decide {installed_version} does not expose the required serve/model APIs. Reference package: {STARTLUX_REFERENCE_VERSION}.")
    parser = cli.build_parser()
    _disable_abbreviations(parser)
    args = parser.parse_args(argv)
    if getattr(args, "command", None) != "serve":
        raise StartLuxSetupError("The companion accepts only the official serve command; it does not download or quantize models.")
    folder_path = validate_local_model_directory(getattr(args, "model", None))
    args.model = str(folder_path)
    try:
        folder = modelcfg.load_model_folder(str(folder_path))
    except (OSError, ValueError, TypeError) as exc:
        raise StartLuxSetupError(f"The official package rejected local model metadata: {exc}") from exc
    report = {
        "package": "startlux-decide", "version": installed_version,
        "backend": args.backend, "kind": folder.kind, "prompt_attention": folder.prompt_attention,
        "model_path": str(folder_path), "offline_model_files": True,
        "status": "preflight_only", "warnings": [],
    }
    if installed_version != STARTLUX_REFERENCE_VERSION:
        report["warnings"].append(f"Reference package is {STARTLUX_REFERENCE_VERSION}; installed {installed_version} passed the required CLI/API checks.")
    if args.backend == "vllm" and getattr(args, "launch_vllm", False):
        try:
            vllm_version = metadata.version("vllm")
        except metadata.PackageNotFoundError as exc:
            raise StartLuxSetupError("Local --launch-vllm needs vLLM in the selected StartLux environment; install the official [vllm] extra or explicitly supply --vllm-url.") from exc
        report["engine_version"] = vllm_version
        report["warnings"].extend(verified_runtime_support("startlux_vllm", vllm_version, "systemone")["warnings"])
        # The official launcher checks plugin activation and correctness flags.
        # Do not duplicate or silently substitute its attention implementation.
    elif args.backend == "vllm":
        report["warnings"].append("An external vLLM engine's version and attention settings must be checked on that engine; preflight does not verify them.")
    if args.backend == "llamacpp":
        report["warnings"].append("Configure llama-server explicitly using the upstream Instant/non-causal, whole-prompt batching and mmproj guidance; preflight cannot verify a remote engine's flags.")
    return cli, report


def main(argv: list[str] | None = None) -> int:
    wrapper = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    wrapper.add_argument("--check", action="store_true")
    options, official_args = wrapper.parse_known_args(sys.argv[1:] if argv is None else argv)
    try:
        cli, report = inspect_official_package(official_args)
        if not options.check and report["backend"] == "transformers" and report["kind"] == "instant" and sys.platform == "win32":
            if __package__:
                from .startlux_windows import configure_windows_attention
            else:
                from startlux_windows import configure_windows_attention
            try:
                configured = configure_windows_attention()
            except ImportError as exc:
                raise StartLuxSetupError(
                    f"Windows Instant requires the optional Transformers/CUDA dependencies ({exc.name}); "
                    "see services/startlux/requirements-windows-cu128.txt."
                ) from exc
            if configured:
                report["runtime_compatibility"] = "Windows fused attention: equivalent repeated K/V heads for bidirectional GQA; official self-checks enabled"
        print(json.dumps(report, ensure_ascii=False), file=sys.stdout if options.check else sys.stderr, flush=True)
        if options.check:
            return 0
        return cli.main(official_args) or 0
    except StartLuxSetupError as exc:
        print(f"StartLux setup error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
