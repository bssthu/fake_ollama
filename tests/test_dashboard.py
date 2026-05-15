"""Tests for dashboard telemetry persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fake_ollama.config import Settings
from fake_ollama.dashboard import DashboardState, _DASHBOARD_HTML


def _settings(path: Path, *, retention_seconds: float = 3600.0) -> Settings:
    return Settings(
        upstreams=[
            {
                "name": "u",
                "base_url": "http://upstream.test",
                "auth_token": "tok",
                "models": ["m"],
            }
        ],
        dashboard_data_path=str(path),
        dashboard_retention_seconds=retention_seconds,
        dashboard_sample_interval_seconds=10.0,
    )


def _app(settings: Settings) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(settings=settings))


def test_dashboard_chart_breaks_lines_across_sample_gaps() -> None:
    assert "function maxSampleGapSeconds()" in _DASHBOARD_HTML
    assert "p.ts - last.ts > maxGap" in _DASHBOARD_HTML


def test_dashboard_state_persists_and_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dashboard-history.json"
    app = _app(_settings(path))

    monkeypatch.setattr("fake_ollama.dashboard.time.time", lambda: 1000.0)
    state = DashboardState()
    sample = asyncio.run(state.sample(app))

    assert sample["ts"] == 1000.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["samples"][0]["ts"] == 1000.0

    monkeypatch.setattr("fake_ollama.dashboard.time.time", lambda: 1005.0)
    reloaded = DashboardState()
    data = asyncio.run(reloaded.data(app, range_seconds=3600.0))

    assert [s["ts"] for s in data["samples"]] == [1000.0]


def test_dashboard_relative_history_path_uses_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "config.json"
    settings = _settings(Path("history.json"))
    settings = settings.model_copy(
        update={
            "config_path": str(cfg_path),
            "dashboard_data_path": "history.json",
        }
    )
    app = _app(settings)

    monkeypatch.setattr("fake_ollama.dashboard.time.time", lambda: 1500.0)
    asyncio.run(DashboardState().sample(app))

    assert (tmp_path / "history.json").exists()


def test_dashboard_state_ignores_corrupt_history_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dashboard-history.json"
    path.write_text("{not-json", encoding="utf-8")
    app = _app(_settings(path))

    monkeypatch.setattr("fake_ollama.dashboard.time.time", lambda: 2000.0)
    data = asyncio.run(DashboardState().data(app, range_seconds=3600.0))

    assert [s["ts"] for s in data["samples"]] == [2000.0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["samples"][0]["ts"] == 2000.0


def test_dashboard_state_drops_expired_and_bad_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dashboard-history.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "samples": [
                    {"ts": 1900, "memory_free_mib": 1, "models": {"old": 1}},
                    {
                        "ts": 1995,
                        "memory_free_mib": "512",
                        "models": {"active": "1024", "bad": "nan"},
                    },
                    {"ts": "bad"},
                    "bad-record",
                ],
            }
        ),
        encoding="utf-8",
    )
    app = _app(_settings(path, retention_seconds=60.0))

    monkeypatch.setattr("fake_ollama.dashboard.time.time", lambda: 2000.0)
    data = asyncio.run(DashboardState().data(app, range_seconds=3600.0))

    assert [s["ts"] for s in data["samples"]] == [1995.0]
    assert data["samples"][0]["memory_free_mib"] == 512.0
    assert data["samples"][0]["models"] == {"active": 1024.0}
