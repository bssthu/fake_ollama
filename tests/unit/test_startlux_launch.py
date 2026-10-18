"""Offline optional-integration tests: no installed StartLux, GPU, downloads or providers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fake_ollama import decision_launch, startlux_companion
from fake_ollama.decision_launch import StartLuxSetupError, build_startlux_argv, verified_runtime_support


@pytest.fixture
def local_model(tmp_path):
    folder = tmp_path / "my model; still a path"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "decision_config.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture
def target(local_model):
    return SimpleNamespace(model_path=str(local_model), python_path=None, runtime_backend="vllm",
                           base_url="http://127.0.0.1:8090", start_args=[], cwd=None, dialect="systemone")


def test_argv_keeps_paths_separate_and_uses_packaged_script(target):
    command = build_startlux_argv(target)
    assert command[1] == str(Path(decision_launch.__file__).with_name("startlux_companion.py"))
    assert command[command.index("--model") + 1] == target.model_path
    assert "--launch-vllm" in command
    assert all(isinstance(item, str) for item in command)


def test_transformers_launch_is_optional_and_not_vllm(target):
    target.runtime_backend = "transformers"
    target.start_args = ["--device", "cpu", "--dtype", "float32"]
    assert "--launch-vllm" not in build_startlux_argv(target)


def test_existing_engine_url_does_not_launch_or_require_local_engine(target):
    target.start_args = ["--vllm-url=http://inference.example.test:8000"]
    assert "--launch-vllm" not in build_startlux_argv(target)


@pytest.mark.parametrize("value", [None, "", "startlux-models/not-downloaded", "https://huggingface.co/some/model"])
def test_no_remote_or_missing_model_is_accepted(target, value):
    target.model_path = value
    with pytest.raises(StartLuxSetupError, match="local|directory"):
        build_startlux_argv(target)


def test_missing_and_malformed_original_metadata_rejected(target, local_model):
    (local_model / "decision_config.json").unlink()
    with pytest.raises(StartLuxSetupError, match="decision_config.json"):
        build_startlux_argv(target)
    (local_model / "decision_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(StartLuxSetupError, match="JSON object"):
        build_startlux_argv(target)


@pytest.mark.parametrize("url", ["http://0.0.0.0:8090", "http://example.test:8090", "https://127.0.0.1:8090",
                                "http://127.0.0.1:8090/v1", "http://user:pass@127.0.0.1:8090", "http://localhost",
                                "http://[::1]:0", "http://127.0.0.1:8090?x=1"])
def test_managed_address_is_explicit_loopback_origin(target, url):
    target.base_url = url
    with pytest.raises(StartLuxSetupError):
        build_startlux_argv(target)


def test_ipv6_loopback_argv_is_unbracketed(target):
    target.base_url = "http://[::1]:8090"
    command = build_startlux_argv(target)
    assert command[command.index("--host") + 1] == "::1"


@pytest.mark.parametrize("args", [["--model", "remote/model"], ["--host=0.0.0.0"], ["--port", "99"],
                                 ["--backend=mlx"], ["--vllm-python", "other-python"], ["--help"], ["--"],
                                 "--device cpu", ["--device", "cpu\x00bad"]])
def test_extra_arguments_cannot_override_managed_identity(target, args):
    target.start_args = args
    with pytest.raises(StartLuxSetupError):
        build_startlux_argv(target)


def test_incompatible_engine_and_dialect_fail_locally(target):
    target.dialect = "openai_decisions"
    with pytest.raises(StartLuxSetupError, match="systemone"):
        build_startlux_argv(target)
    target.dialect = "systemone"
    target.runtime_backend = "transformers"
    target.start_args = ["--launch-vllm"]
    with pytest.raises(StartLuxSetupError, match="runtime_backend"):
        build_startlux_argv(target)


def test_relative_model_and_interpreter_use_configured_cwd(target, local_model, tmp_path):
    executable = tmp_path / "python-in-env"
    executable.touch()
    target.cwd = str(tmp_path)
    target.model_path = local_model.name
    target.python_path = "./python-in-env"
    command = build_startlux_argv(target)
    assert command[0] == str(executable)
    assert command[command.index("--model") + 1] == str(local_model)


def test_interpreter_symlink_preserves_virtual_environment_identity(target, tmp_path):
    executable = tmp_path / "real-python"
    executable.touch()
    venv_python = tmp_path / "venv-python"
    try:
        venv_python.symlink_to(executable)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks permission to create symbolic links")
        raise
    target.python_path = str(venv_python)
    assert build_startlux_argv(target)[0] == str(venv_python)


def test_version_checks_distinguish_native_vllm_from_official_wrapper():
    native = verified_runtime_support("vllm", "0.31.0", "systemone")
    wrapper = verified_runtime_support("startlux_vllm", "0.31.0", "systemone")
    assert native["status"] == "unsupported" and "2026-10-08" in native["errors"][0]
    assert wrapper["status"] == "supported" and not wrapper["errors"]
    assert verified_runtime_support("vllm", "0.32.0.dev17", "systemone")["status"] == "unverified"
    assert verified_runtime_support("vllm", "0.31.0", "systemone", endpoint_verified=True)["status"] == "supported"
    assert verified_runtime_support("llama.cpp", "v0.6.0", "systemone")["status"] == "supported"
    assert verified_runtime_support("llama_cpp", "0.5.0", "systemone")["status"] == "unsupported"
    assert verified_runtime_support("llama_cpp", "b11429", "systemone")["status"] == "unverified"
    assert verified_runtime_support("startlux", "1.1.0", "openai_decisions", endpoint_verified=True)["status"] == "unsupported"


@pytest.fixture
def official_mock(monkeypatch):
    def parser():
        root = argparse.ArgumentParser()
        subs = root.add_subparsers(dest="command", required=True)
        serve = subs.add_parser("serve")
        serve.add_argument("--model", required=True)
        serve.add_argument("--backend", default="vllm")
        serve.add_argument("--launch-vllm", action="store_true")
        serve.add_argument("--vllm-url")
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8090)
        return root
    cli = SimpleNamespace(build_parser=parser, main=Mock(return_value=0))
    modelcfg = SimpleNamespace(load_model_folder=Mock(return_value=SimpleNamespace(kind="instant", prompt_attention="bidirectional")))
    def import_module(name):
        # Offline flags must precede all optional imports.
        assert startlux_companion.os.environ["HF_HUB_OFFLINE"] == "1"
        assert startlux_companion.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return {"startlux_decide.cli": cli, "startlux_decide.modelcfg": modelcfg}[name]
    monkeypatch.setattr(startlux_companion.importlib, "import_module", import_module)
    monkeypatch.setattr(startlux_companion.metadata, "version", lambda name: "1.1.0" if name == "startlux-decide" else "0.31.0")
    return cli, modelcfg


def test_check_uses_official_metadata_without_inference(local_model, official_mock, capsys, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    result = startlux_companion.main(["--check", "serve", "--model", str(local_model), "--launch-vllm"])
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == "1.1.0"
    assert report["engine_version"] == "0.31.0"
    assert report["status"] == "preflight_only"
    assert report["offline_model_files"] is True
    official_mock[0].main.assert_not_called()
    official_mock[1].load_model_folder.assert_called_once_with(str(local_model))


def test_actual_serving_delegates_exactly_to_official_cli(local_model, official_mock, monkeypatch):
    from fake_ollama import startlux_windows
    monkeypatch.setattr(startlux_windows, "configure_windows_attention", Mock(return_value=True))
    argv = ["serve", "--model", str(local_model), "--backend", "transformers"]
    assert startlux_companion.main(argv) == 0
    official_mock[0].main.assert_called_once_with(argv)


@pytest.mark.parametrize("check,kind,platform,expected", [
    (True, "instant", "win32", False),
    (False, "auto_think", "win32", False),
    (False, "instant", "linux", False),
    (False, "instant", "win32", True),
])
def test_windows_adapter_is_only_loaded_for_actual_instant_serving(
    local_model, official_mock, monkeypatch, check, kind, platform, expected,
):
    from fake_ollama import startlux_windows
    configure = Mock(return_value=True)
    monkeypatch.setattr(startlux_windows, "configure_windows_attention", configure)
    monkeypatch.setattr(startlux_companion.sys, "platform", platform)
    official_mock[1].load_model_folder.return_value.kind = kind
    argv = ["serve", "--model", str(local_model), "--backend", "transformers"]
    assert startlux_companion.main((["--check"] if check else []) + argv) == 0
    assert configure.call_count == int(expected)


def test_missing_optional_package_is_actionable_without_import(monkeypatch, capsys):
    def absent(name):
        raise startlux_companion.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(startlux_companion.metadata, "version", absent)
    imported = Mock(side_effect=AssertionError("Optional import must not occur"))
    monkeypatch.setattr(startlux_companion.importlib, "import_module", imported)
    assert startlux_companion.main(["--check", "serve"]) == 2
    assert "python_path" in capsys.readouterr().err
    imported.assert_not_called()


def test_future_package_version_allowed_if_required_api_passes(local_model, official_mock, monkeypatch):
    monkeypatch.setattr(startlux_companion.metadata, "version", lambda name: "1.2.0")
    _, report = startlux_companion.inspect_official_package(["serve", "--model", str(local_model), "--backend", "transformers"])
    assert report["version"] == "1.2.0"
    assert "passed" in report["warnings"][0]


def test_model_validation_remains_owned_by_official_package(local_model, official_mock, capsys):
    official_mock[1].load_model_folder.side_effect = ValueError("synthetic model metadata mismatch")
    assert startlux_companion.main(["--check", "serve", "--model", str(local_model)]) == 2
    assert "official package rejected" in capsys.readouterr().err
    official_mock[0].main.assert_not_called()


@pytest.mark.parametrize("option", ["--mod", "--ho", "--po"])
def test_abbreviated_overrides_rejected_before_actual_cli(local_model, official_mock, option):
    argv = ["serve", "--model", str(local_model), option, "other"]
    with pytest.raises(SystemExit):
        startlux_companion.main(argv)
    official_mock[0].main.assert_not_called()


def test_local_vllm_missing_is_actionable_but_external_engine_not_required(local_model, official_mock, monkeypatch, capsys):
    def version(name):
        if name == "vllm":
            raise startlux_companion.metadata.PackageNotFoundError(name)
        return "1.1.0"
    monkeypatch.setattr(startlux_companion.metadata, "version", version)
    base = ["--check", "serve", "--model", str(local_model)]
    assert startlux_companion.main(base + ["--launch-vllm"]) == 2
    assert "--vllm-url" in capsys.readouterr().err
    assert startlux_companion.main(base + ["--vllm-url", "http://inference.example.test:8000"]) == 0


def test_packaged_script_imports_without_optional_packages(monkeypatch):
    import builtins
    import runpy
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "startlux_decide", "vllm"}:
            raise AssertionError(f"Unexpected optional import: {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.syspath_prepend(str(Path(decision_launch.__file__).parent))
    loaded = runpy.run_path(str(Path(decision_launch.__file__).with_name("startlux_companion.py")))
    assert callable(loaded["main"])


def test_documented_configuration_is_valid_and_official_source_is_pinned():
    import re
    from fake_ollama.config import DecisionTarget
    root = Path(__file__).resolve().parents[2]
    document = (root / "docs/decision-startlux.md").read_text(encoding="utf-8")
    config = json.loads(re.search(r"```json\n(.*?)\n```", document, re.S).group(1))
    target = DecisionTarget.model_validate(config["decision_targets"][0])
    assert target.runtime == "startlux"
    assert target.dialect == "systemone"
    assert target.effective_endpoint_path == "/v1/decisions"
    requirements = (root / "services/startlux/requirements-vllm.txt").read_text(encoding="utf-8")
    assert "startlux-decide[vllm] @ git+https://github.com/StartLuxLabs/StartLux-Decision.git@83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01" in requirements


@pytest.mark.parametrize("health,expected", [
    ({"status": "ok", "kind": "instant"}, {"think_modes": ["never"]}),
    ({"status": "ok", "prompt_attention": "bidirectional", "images": False}, {"think_modes": ["never"], "modalities": ["text"]}),
    ({"status": "ok", "kind": "auto_think", "images": True}, {}),
    ({"status": "ok", "kind": "v1"}, {}),
    ({"status": "ok", "kind": "auto_think", "images": False}, {"modalities": ["text"]}),
    ({"status": "ok", "model": "My-Instant-Model"}, {}),
    ({"status": "loading", "kind": "instant"}, {}),
    (None, {}),
])
def test_official_health_only_restricts_confirmed_capabilities(health, expected):
    assert decision_launch.derive_startlux_capabilities(health) == expected
