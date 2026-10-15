import json
from pathlib import Path

import pytest

from scripts import install_h3_guides as installer


@pytest.fixture
def installation(tmp_path, monkeypatch):
    root = tmp_path / "comfy"
    root.mkdir()
    files = []
    for name in ("model.py", "base.py", "nodes.py"):
        (root / name).write_bytes(b"VALUE = 1\r\n")
        files.append(dict(path=name, source_sha256=installer.digest("VALUE = 1\n"),
                          result_sha256=installer.digest("VALUE = 2\n"),
                          edits=[dict(start=0, end=1, lines=["VALUE = 2\n"])]))
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"files": files}), encoding="utf-8")
    monkeypatch.setattr(installer, "PATCH", patch)
    return root


def test_dry_run_backup_and_idempotent_install(installation):
    result = installer.install(installation)
    assert len(result["pending"]) == 3
    assert not result["applied"]
    result = installer.install(installation, apply=True)
    assert result["applied"]
    for name in ("model.py", "base.py", "nodes.py"):
        assert (Path(result["backup"]) / name).read_bytes() == b"VALUE = 1\r\n"
        assert (installation / name).read_text() == "VALUE = 2\n"
    assert installer.install(installation, apply=True)["pending"] == []


def test_unknown_version_rejected_before_any_write(installation):
    (installation / "nodes.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="Unsupported or modified"):
        installer.install(installation, apply=True)
    assert (installation / "model.py").read_bytes() == b"VALUE = 1\r\n"
    assert not list(installation.glob("h3-guide-backup-*"))


def test_partial_write_failure_restores_original_bytes(installation, monkeypatch):
    original_write = Path.write_text

    def fail_second(path, *args, **kwargs):
        if path == installation / "base.py":
            raise OSError("simulated write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_second)
    with pytest.raises(OSError, match="simulated"):
        installer.install(installation, apply=True)
    for name in ("model.py", "base.py", "nodes.py"):
        assert (installation / name).read_bytes() == b"VALUE = 1\r\n"
