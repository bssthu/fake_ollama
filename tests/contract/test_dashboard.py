"""Contract tests for dashboard telemetry and persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fake_ollama.config import Settings
from fake_ollama.dashboard import (
    DashboardState,
    RequestMetrics,
    _DASHBOARD_HTML,
    _write_dashboard_samples,
    register_dashboard_routes,
)


def _settings(path: Path, *, retention_seconds: float = 3600.0) -> Settings:
    return Settings(
        anthropic_upstreams=[
            {
                "name": "u",
                "base_url": "http://upstream.test",
                "auth_token": "tok",
                "models": [{"name": "m"}],
            }
        ],
        ollama_interfaces=[],
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

    def loaded_model_snapshots(self, *, now: float, idle_reclaim_seconds: float = 60.0) -> list[dict[str, object]]:
        return [dict(self.snapshot)]


class _RecordingCoordinator:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[str, str, bool]] = []

    async def reclaim_model(
        self, *, target_id: str, model: str, force: bool = False
    ) -> dict[str, object]:
        self.calls.append((target_id, model, force))
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


def test_dashboard_write_retries_transient_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dashboard-history.json"
    original_replace = Path.replace
    failures: list[str] = []

    def flaky_replace(self: Path, target: Path) -> Path:
        if target == path and not failures:
            failures.append(self.name)
            raise OSError(32, "file is in use")
        return original_replace(self, target)

    monkeypatch.setattr("fake_ollama.dashboard.time.sleep", lambda _: None)
    monkeypatch.setattr(Path, "replace", flaky_replace)

    _write_dashboard_samples(path, [{"ts": 1234.0, "models": {}}])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["samples"][0]["ts"] == 1234.0
    assert failures
    assert not list(tmp_path.glob(".dashboard-history.json.*.tmp"))


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
    assert coord.calls == [("ollama:t", "m", False)]


def test_dashboard_reclaim_model_passes_force_when_enabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "history.json").model_copy(
        update={"dashboard_model_reclaim_enabled": True}
    )
    coord = _RecordingCoordinator(
        {"target_id": "ollama:t", "model": "m", "released": True}
    )
    snapshot = _model_snapshot()
    snapshot["reclaimable"] = False
    snapshot["active_requests"] = 1
    app = _route_app(settings, snapshot=snapshot, coordinator=coord)

    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/reclaim-model",
            json={"key": "ollama|ollama:t|m", "force": True},
        )

    assert resp.status_code == 200
    assert resp.json()["released"] is True
    assert coord.calls == [("ollama:t", "m", True)]


def _begin(metrics: RequestMetrics, *, started_monotonic: float, started_wall: float, **kwargs) -> int:
    defaults = {
        "listener": "api",
        "port": 21435,
        "surface": "openai",
        "client": "127.0.0.1",
        "method": "POST",
        "path": "/v1/chat/completions",
    }
    defaults.update(kwargs)
    return metrics.begin(
        started_monotonic=started_monotonic,
        started_wall=started_wall,
        **defaults,
    )


def test_request_metrics_tracks_inflight_then_moves_to_history() -> None:
    metrics = RequestMetrics()
    rid = _begin(metrics, started_monotonic=100.0, started_wall=1000.0)

    inflight = metrics.inflight_snapshot()
    assert len(inflight) == 1
    assert inflight[0]["req_id"] == rid
    assert inflight[0]["surface"] == "openai"
    assert inflight[0]["error_type"] is None
    # elapsed_ms is measured from wall-clock monotonic; just sanity-check it's non-negative.
    assert inflight[0]["elapsed_ms"] >= 0

    metrics.end(rid, status=200, finished_monotonic=103.5)
    assert metrics.inflight_snapshot() == []

    stats = metrics.stats(now_wall=1100.0, windows=(300.0,))
    window = stats["windows"]["300"]
    assert window["total"] == 1
    group = window["groups"][0]
    assert group["surface"] == "openai"
    assert group["total"] == 1
    assert group["by_status_class"] == {"2xx": 1}
    assert group["p50_ms"] == pytest.approx(3500.0)
    assert group["max_ms"] == pytest.approx(3500.0)


def test_request_metrics_set_error_type_persists_after_end() -> None:
    metrics = RequestMetrics()
    rid = _begin(metrics, started_monotonic=10.0, started_wall=500.0)

    metrics.set_error_type(rid, "ReadTimeout")
    inflight = metrics.inflight_snapshot()
    assert inflight[0]["error_type"] == "ReadTimeout"

    # set_error_type after end is a no-op (the record is already in history).
    metrics.end(rid, status=502, finished_monotonic=610.0)
    metrics.set_error_type(rid, "ConnectError")

    stats = metrics.stats(now_wall=600.0, windows=(300.0,))
    group = stats["windows"]["300"]["groups"][0]
    assert group["by_status_class"] == {"5xx": 1}
    assert group["errors"] == {"ReadTimeout": 1}


def test_request_metrics_inflight_sorted_by_elapsed_desc() -> None:
    metrics = RequestMetrics()
    older = _begin(metrics, started_monotonic=100.0, started_wall=1000.0, path="/older")
    _ = _begin(metrics, started_monotonic=200.0, started_wall=1100.0, path="/newer")

    inflight = metrics.inflight_snapshot()
    # The earlier started_monotonic produces the larger elapsed_ms, so it sorts first.
    assert inflight[0]["req_id"] == older
    assert inflight[0]["path"] == "/older"
    assert inflight[0]["elapsed_ms"] > inflight[1]["elapsed_ms"]


def test_request_metrics_stats_window_drops_old_requests() -> None:
    metrics = RequestMetrics()
    # Inside the 5-min window (now=2000 - 100 = 1900 >= 2000-300=1700).
    rid_recent = _begin(metrics, started_monotonic=10.0, started_wall=1900.0)
    metrics.end(rid_recent, status=200, finished_monotonic=12.0)
    # Outside the 5-min window (1500 < 1700).
    rid_old = _begin(metrics, started_monotonic=1.0, started_wall=1500.0)
    metrics.end(rid_old, status=500, finished_monotonic=3.0)

    stats = metrics.stats(now_wall=2000.0, windows=(300.0, 3600.0))
    assert stats["windows"]["300"]["total"] == 1
    assert stats["windows"]["3600"]["total"] == 2


def test_request_metrics_groups_by_listener_port_surface() -> None:
    metrics = RequestMetrics()
    rid_a = _begin(
        metrics, started_monotonic=10.0, started_wall=1000.0,
        listener="api", port=21435, surface="openai",
    )
    rid_b = _begin(
        metrics, started_monotonic=11.0, started_wall=1001.0,
        listener="api", port=21435, surface="openai",
    )
    rid_c = _begin(
        metrics, started_monotonic=12.0, started_wall=1002.0,
        listener="ollama", port=21434, surface="ollama",
    )
    metrics.end(rid_a, status=200, finished_monotonic=11.0)
    metrics.set_error_type(rid_b, "ReadTimeout")
    metrics.end(rid_b, status=502, finished_monotonic=15.0)
    metrics.end(rid_c, status=200, finished_monotonic=14.0)

    groups = metrics.stats(now_wall=1100.0)["windows"]["300"]["groups"]
    by_key = {(g["listener"], g["port"], g["surface"], g["target"]): g for g in groups}

    openai_g = by_key[("api", 21435, "openai", None)]
    assert openai_g["total"] == 2
    assert openai_g["by_status_class"] == {"2xx": 1, "5xx": 1}
    assert openai_g["errors"] == {"ReadTimeout": 1}

    ollama_g = by_key[("ollama", 21434, "ollama", None)]
    assert ollama_g["total"] == 1
    assert ollama_g["by_status_class"] == {"2xx": 1}
    assert ollama_g["errors"] == {}


def test_request_metrics_groups_by_target() -> None:
    metrics = RequestMetrics()
    # Same (listener, port, surface) but different targets should split into
    # separate buckets — otherwise the dashboard can't distinguish which
    # backend was slow when two share an interface.
    rid_a = _begin(metrics, started_monotonic=10.0, started_wall=1000.0)
    metrics.set_target(rid_a, "qwen-9b")
    metrics.end(rid_a, status=200, finished_monotonic=11.0)

    rid_b = _begin(metrics, started_monotonic=12.0, started_wall=1002.0)
    metrics.set_target(rid_b, "llama-13b")
    metrics.set_error_type(rid_b, "ReadTimeout")
    metrics.end(rid_b, status=502, finished_monotonic=312.0)

    rid_c = _begin(metrics, started_monotonic=20.0, started_wall=1010.0)
    # No target set — dashboard self-traffic, etc.
    metrics.end(rid_c, status=200, finished_monotonic=20.5)

    groups = metrics.stats(now_wall=1100.0)["windows"]["300"]["groups"]
    by_target = {g["target"]: g for g in groups}
    assert by_target["qwen-9b"]["total"] == 1
    assert by_target["qwen-9b"]["by_status_class"] == {"2xx": 1}
    assert by_target["llama-13b"]["total"] == 1
    assert by_target["llama-13b"]["errors"] == {"ReadTimeout": 1}
    assert by_target[None]["total"] == 1


def test_request_metrics_set_target_surfaces_in_inflight() -> None:
    metrics = RequestMetrics()
    rid = _begin(metrics, started_monotonic=10.0, started_wall=1000.0)
    assert metrics.inflight_snapshot()[0]["target"] is None
    metrics.set_target(rid, "qwen-9b")
    assert metrics.inflight_snapshot()[0]["target"] == "qwen-9b"
    # set_target on an unknown id is a no-op (the record may have ended).
    metrics.set_target(999, "nope")


def test_request_metrics_history_cap_drops_oldest() -> None:
    metrics = RequestMetrics(max_history=3)
    for i in range(5):
        rid = _begin(
            metrics, started_monotonic=float(i), started_wall=1000.0 + i,
            path=f"/r{i}",
        )
        metrics.end(rid, status=200, finished_monotonic=float(i) + 0.1)

    # Only the most recent 3 should remain.
    stats = metrics.stats(now_wall=2000.0, windows=(3600.0,))
    assert stats["history_size"] == 3
    assert stats["history_capacity"] == 3
    assert stats["windows"]["3600"]["total"] == 3


def test_dashboard_data_includes_inflight_and_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "history.json"
    settings = _settings(path)
    app = FastAPI()
    app.state.settings = settings
    app.state.dashboard_state = DashboardState()
    app.state.vram_coordinator = None
    app.state.ollama_clients = {}
    app.state.llama_cpp_clients = {}
    metrics = RequestMetrics()
    app.state.request_metrics = metrics

    rid_done = _begin(metrics, started_monotonic=10.0, started_wall=900.0)
    metrics.end(rid_done, status=200, finished_monotonic=11.0)
    _begin(metrics, started_monotonic=5.0, started_wall=950.0, path="/inflight")

    monkeypatch.setattr("fake_ollama.dashboard.time.time", lambda: 1000.0)
    data = asyncio.run(DashboardState().data(app, range_seconds=3600.0))

    assert any(r["path"] == "/inflight" for r in data["inflight_requests"])
    assert data["request_stats"]["windows"]["300"]["total"] == 1
