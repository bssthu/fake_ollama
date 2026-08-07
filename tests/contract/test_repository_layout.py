"""Repository layout and packaging contracts."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_canonical_config_example_has_no_backup_peer() -> None:
    assert (ROOT / "config.json.example").is_file()
    assert not (ROOT / "config.json.bak").exists()
    assert not list(ROOT.glob("config.json.bak.*"))


def test_mage_vl_is_a_self_contained_service_package() -> None:
    service = ROOT / "services" / "mage_vl"
    metadata = tomllib.loads((service / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"]["mage-vl-adapter"] == (
        "mage_vl_adapter.server:main"
    )
    assert (service / "src" / "mage_vl_adapter" / "__main__.py").is_file()
    assert (service / "tests" / "test_adapter.py").is_file()
    assert not (ROOT / "mage_vl_adapter").exists()
    assert not (ROOT / "requirements.txt").exists()


def test_web_assets_are_packaged_as_static_resources() -> None:
    static = resources.files("fake_ollama").joinpath("static")
    assert "<!doctype html>" in static.joinpath("admin.html").read_text(
        encoding="utf-8"
    ).lower()
    assert "<!doctype html>" in static.joinpath("dashboard.html").read_text(
        encoding="utf-8"
    ).lower()


def test_example_and_validation_scripts_use_layered_directories() -> None:
    examples = ROOT / "scripts" / "examples"
    validation = ROOT / "scripts" / "validation"
    assert {path.name for path in examples.glob("*.py")} == {
        "call_joyai_vl_recognition.py",
        "call_mage_vl_video.py",
        "call_video_generation.py",
    }
    assert (validation / "validate_playground_camera.mjs").is_file()
    assert not list((ROOT / "scripts").glob("call_*.py"))
