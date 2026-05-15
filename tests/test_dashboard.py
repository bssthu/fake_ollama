"""Tests for dashboard telemetry persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.dashboard import DashboardState, _DASHBOARD_HTML, register_dashboard_routes


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


def _model_snapshot() -> dict[str, object]:
    return {
        "backend": "ollama",
        "target_id": "ollama:t",
        "model": "m",
        "estimated_vram_gb": 1.0,
        "estimated_vram_mib": 1024.0,
        "active_requests": 0,
        "request_refs": 0,
        "idle_seconds": 120.0,
        "reclaimable": True,
    }


class _SnapshotClient:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = snapshot

    def loaded_model_snapshots(self, *, now: float) -> list[dict[str, object]]:
        return [dict(self.snapshot)]


class _RecordingCoordinator:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def reclaim_model(self, *, target_id: str, model: str) -> dict[str, object]:
        self.calls.append((target_id, model))
        return dict(self.result)


def _route_app(
    settings: Settings,
    *,
    snapshot: dict[str, object] | None = None,
    coordinator: _RecordingCoordinator | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.settings = settings
    app.state.dashboard_state = DashboardState()
    app.state.vram_coordinator = coordinator
    app.state.ollama_clients = {"t": _SnapshotClient(snapshot or _model_snapshot())}
    app.state.llama_cpp_clients = {}
    register_dashboard_routes(app)
    return app


def test_dashboard_chart_breaks_lines_across_sample_gaps() -> None:
    assert "function maxSampleGapSeconds()" in _DASHBOARD_HTML
    assert "p.ts - last.ts > maxGap" in _DASHBOARD_HTML
    assert "/dashboard/reclaim-model" in _DASHBOARD_HTML


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
    assert data["permissions"]["dashboard_model_reclaim_enabled"] is False


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


def test_dashboard_reclaim_model_rejects_when_disabled(tmp_path: Path) -> None:
    coord = _RecordingCoordinator(
        {"target_id": "ollama:t", "model": "m", "released": True}
    )
    app = _route_app(_settings(tmp_path / "history.json"), coordinator=coord)

    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/reclaim-model", json={"key": "ollama|ollama:t|m"}
        )

    assert resp.status_code == 403
    assert coord.calls == []


def test_dashboard_reclaim_model_calls_coordinator_when_enabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "history.json").model_copy(
        update={"dashboard_model_reclaim_enabled": True}
    )
    coord = _RecordingCoordinator(
        {"target_id": "ollama:t", "model": "m", "released": True}
    )
    app = _route_app(settings, coordinator=coord)

    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/reclaim-model", json={"key": "ollama|ollama:t|m"}
        )

    assert resp.status_code == 200
    assert resp.json()["released"] is True
    assert coord.calls == [("ollama:t", "m")]
