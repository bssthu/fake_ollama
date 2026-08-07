"""Runtime dashboard for local model and memory telemetry."""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import math
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .vram import (
    MemoryCoordinator,
    VramCoordinator,
    system_memory_status_mib as _memory_status_mib,
)


_LOG = logging.getLogger("fake_ollama")
_DASHBOARD_DATA_VERSION = 1
_DASHBOARD_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def _model_key(snapshot: dict[str, object]) -> str:
    return "|".join(
        str(snapshot.get(k) or "") for k in ("backend", "target_id", "model")
    )


def _collect_model_snapshots(app: FastAPI) -> list[dict[str, object]]:
    now = time.monotonic()
    settings = getattr(app.state, "settings", None)
    # The dashboard close button is a user-driven action and uses a much
    # looser idle threshold than the automatic LRU-reclaim path. Falls
    # back to 20s if the field is missing on older Settings instances.
    idle_reclaim_seconds = float(
        getattr(settings, "dashboard_reclaim_idle_seconds", 20.0) or 20.0
    )
    snapshots: list[dict[str, object]] = []
    for clients in (
        getattr(app.state, "ollama_clients", {}),
        getattr(app.state, "llama_cpp_clients", {}),
        getattr(app.state, "generic_openai_clients", {}),
        getattr(app.state, "comfyui_clients", {}),
    ):
        for client in list(clients.values()):
            getter = getattr(client, "loaded_model_snapshots", None)
            if not callable(getter):
                continue
            for raw in getter(now=now, idle_reclaim_seconds=idle_reclaim_seconds):
                snap = dict(raw)
                snap["key"] = _model_key(snap)
                snapshots.append(snap)
    snapshots.sort(key=lambda item: str(item.get("key") or ""))
    return snapshots


_TARGET_TELEMETRY_TIMEOUT_S = 2.0
_TARGET_TELEMETRY_STDERR_BYTES = 16384
_TARGET_TELEMETRY_STDERR_EVENTS = 40


async def _collect_target_telemetry(app: FastAPI) -> List[Dict[str, Any]]:
    """Gather per-target queue/slot/stderr telemetry for the dashboard.

    Runs each ``llama_cpp_client``'s queue/slot/stderr collectors concurrently
    with a short timeout so a slow or wedged upstream cannot block the
    dashboard data endpoint. Anything that fails to respond in time is
    surfaced as an error string instead of being silently dropped.
    """
    clients = {
        **(getattr(app.state, "llama_cpp_clients", None) or {}),
        **(getattr(app.state, "generic_openai_clients", None) or {}),
    }
    if not clients:
        return []

    async def _one(name: str, client: Any) -> Dict[str, Any]:
        # queue_wait_metrics is in-memory, no need to gate it on the timeout.
        queue: Optional[Dict[str, Any]] = None
        get_queue = getattr(client, "queue_wait_metrics", None)
        if callable(get_queue):
            try:
                queue = get_queue()
            except Exception as exc:
                queue = {"error": f"{exc.__class__.__name__}: {exc}"}

        slots_payload: Dict[str, Any] = {"slots": None, "error": "not supported"}
        fetch_slots = getattr(client, "fetch_upstream_slots", None)
        if callable(fetch_slots):
            try:
                slots_payload = await asyncio.wait_for(
                    fetch_slots(timeout=_TARGET_TELEMETRY_TIMEOUT_S),
                    timeout=_TARGET_TELEMETRY_TIMEOUT_S + 0.5,
                )
            except asyncio.TimeoutError:
                slots_payload = {"slots": None, "error": "timeout"}
            except Exception as exc:
                slots_payload = {
                    "slots": None,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }

        stderr_payload: Dict[str, Any] = {"available": False, "events": []}
        fetch_stderr = getattr(client, "fetch_recent_stderr_events", None)
        if callable(fetch_stderr):
            try:
                stderr_payload = await asyncio.wait_for(
                    fetch_stderr(
                        max_bytes=_TARGET_TELEMETRY_STDERR_BYTES,
                        limit=_TARGET_TELEMETRY_STDERR_EVENTS,
                    ),
                    timeout=_TARGET_TELEMETRY_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                stderr_payload = {"available": False, "events": [], "error": "timeout"}
            except Exception as exc:
                stderr_payload = {
                    "available": False,
                    "events": [],
                    "error": f"{exc.__class__.__name__}: {exc}",
                }

        target_id = getattr(client, "target_id", None) or f"llama.cpp:{name}"
        return {
            "name": name,
            "target_id": target_id,
            "queue": queue,
            "slots": slots_payload,
            "stderr": stderr_payload,
        }

    results = await asyncio.gather(
        *[_one(name, client) for name, client in list(clients.items())],
        return_exceptions=False,
    )
    results.sort(key=lambda r: str(r.get("target_id") or ""))
    return results


def _dashboard_model_reclaim_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "dashboard_model_reclaim_enabled", False))


def _dashboard_retention_seconds(settings: Any) -> float:
    return max(60.0, float(getattr(settings, "dashboard_retention_seconds", 0) or 0))


def _dashboard_data_path(settings: Any) -> Optional[Path]:
    raw = getattr(settings, "dashboard_data_path", None)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser()
    except (OSError, ValueError):
        _LOG.warning("ignoring invalid dashboard_data_path=%r", raw)
        return None
    if path.is_absolute():
        return path

    config_path = str(getattr(settings, "config_path", "") or "").strip()
    if config_path:
        try:
            return Path(config_path).expanduser().parent / path
        except (OSError, ValueError):
            pass
    return path


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalise_models(value: Any) -> dict[str, Optional[float]]:
    if not isinstance(value, dict):
        return {}
    models: dict[str, Optional[float]] = {}
    for raw_key, raw_vram in value.items():
        key = str(raw_key)
        if not key:
            continue
        if raw_vram is None:
            models[key] = None
            continue
        vram = _optional_float(raw_vram)
        if vram is not None:
            models[key] = vram
    return models


def _normalise_sample(
    raw: Any, *, cutoff: float, now: float
) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    ts = _optional_float(raw.get("ts"))
    if ts is None or ts < cutoff or ts > now + 300.0:
        return None
    return {
        "ts": ts,
        "memory_free_mib": _optional_float(raw.get("memory_free_mib")),
        "memory_total_mib": _optional_float(raw.get("memory_total_mib")),
        "vram_free_mib": _optional_float(raw.get("vram_free_mib")),
        "vram_total_mib": _optional_float(raw.get("vram_total_mib")),
        "models": _normalise_models(raw.get("models")),
        "models_memory": _normalise_models(raw.get("models_memory")),
    }


def _normalise_samples(
    samples: list[dict[str, Any]] | list[Any], *, retention: float, now: float
) -> list[dict[str, Any]]:
    cutoff = now - retention
    out: list[dict[str, Any]] = []
    seen_ts: set[float] = set()
    for raw in samples:
        sample = _normalise_sample(raw, cutoff=cutoff, now=now)
        if sample is None:
            continue
        ts = float(sample["ts"])
        if ts in seen_ts:
            continue
        seen_ts.add(ts)
        out.append(sample)
    out.sort(key=lambda item: float(item["ts"]))
    return out


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    out["models"] = dict(sample.get("models") or {})
    out["models_memory"] = dict(sample.get("models_memory") or {})
    return out


def _read_dashboard_samples(
    path: Path, *, retention: float, now: float
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        _LOG.warning("ignoring unreadable dashboard data file %s: %s", path, exc)
        return []

    raw_samples: Any
    if isinstance(raw, dict):
        raw_samples = raw.get("samples")
    else:
        raw_samples = raw
    if not isinstance(raw_samples, list):
        _LOG.warning("ignoring dashboard data file %s: samples is not a list", path)
        return []
    return _normalise_samples(raw_samples, retention=retention, now=now)


def _write_dashboard_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    tmp: Optional[Path] = None
    try:
        if not path.name:
            raise ValueError("dashboard_data_path must include a file name")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _DASHBOARD_DATA_VERSION,
            "samples": samples,
        }
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as fp:
            tmp = Path(fp.name)
            fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        for delay in (*_DASHBOARD_REPLACE_RETRY_DELAYS, None):
            try:
                tmp.replace(path)
                tmp = None
                break
            except OSError:
                if delay is None:
                    raise
                time.sleep(delay)
    except (OSError, TypeError, ValueError) as exc:
        _LOG.warning("failed to write dashboard data file %s: %s", path, exc)
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


_DEFAULT_STATS_WINDOWS: Tuple[float, ...] = (300.0, 3600.0)
_REQUEST_METRICS_DEFAULT_HISTORY = 5000


@dataclass
class _RequestRecord:
    req_id: int
    listener: str
    port: Any
    surface: str
    client: str
    method: str
    path: str
    started_wall: float
    started_monotonic: float
    finished_monotonic: Optional[float] = None
    status: Optional[int] = None
    error_type: Optional[str] = None
    target: Optional[str] = None


def _status_class(status: Optional[int]) -> str:
    if status is None:
        return "unknown"
    if 200 <= status < 300:
        return "2xx"
    if 300 <= status < 400:
        return "3xx"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return "other"


def _quantile_ms(sorted_ms: List[float], q: float) -> Optional[float]:
    if not sorted_ms:
        return None
    if q <= 0:
        return sorted_ms[0]
    if q >= 1:
        return sorted_ms[-1]
    # Nearest-rank: simple and stable for the dashboard.
    idx = max(0, min(len(sorted_ms) - 1, int(math.ceil(q * len(sorted_ms))) - 1))
    return sorted_ms[idx]


class RequestMetrics:
    """In-memory tracker for in-flight requests and recent completions.

    Used by the dashboard to surface request-level health that is otherwise
    only visible by grep'ing the access log. All mutating methods are
    expected to be called from a single asyncio event loop; no locking.
    """

    def __init__(
        self,
        *,
        max_history: int = _REQUEST_METRICS_DEFAULT_HISTORY,
    ) -> None:
        self._inflight: Dict[int, _RequestRecord] = {}
        self._history: Deque[_RequestRecord] = deque(maxlen=max_history)
        self._next_id = 1

    def begin(
        self,
        *,
        listener: str,
        port: Any,
        surface: str,
        client: str,
        method: str,
        path: str,
        started_monotonic: Optional[float] = None,
        started_wall: Optional[float] = None,
    ) -> int:
        rid = self._next_id
        self._next_id += 1
        self._inflight[rid] = _RequestRecord(
            req_id=rid,
            listener=listener,
            port=port,
            surface=surface,
            client=client,
            method=method,
            path=path,
            started_wall=time.time() if started_wall is None else started_wall,
            started_monotonic=(
                time.monotonic() if started_monotonic is None else started_monotonic
            ),
        )
        return rid

    def set_error_type(self, rid: int, error_type: str) -> None:
        rec = self._inflight.get(rid)
        if rec is not None and not rec.error_type:
            rec.error_type = error_type

    def set_target(self, rid: int, target: Optional[str]) -> None:
        rec = self._inflight.get(rid)
        if rec is not None:
            rec.target = target

    def end(
        self,
        rid: int,
        *,
        status: int,
        finished_monotonic: Optional[float] = None,
    ) -> None:
        rec = self._inflight.pop(rid, None)
        if rec is None:
            return
        rec.finished_monotonic = (
            time.monotonic() if finished_monotonic is None else finished_monotonic
        )
        rec.status = status
        self._history.append(rec)

    def inflight_snapshot(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        now_m = time.monotonic()
        items: List[Dict[str, Any]] = []
        for rec in self._inflight.values():
            items.append(
                {
                    "req_id": rec.req_id,
                    "listener": rec.listener,
                    "port": rec.port,
                    "surface": rec.surface,
                    "target": rec.target,
                    "client": rec.client,
                    "method": rec.method,
                    "path": rec.path,
                    "started_at": rec.started_wall,
                    "elapsed_ms": (now_m - rec.started_monotonic) * 1000.0,
                    "error_type": rec.error_type,
                }
            )
        items.sort(key=lambda x: x["elapsed_ms"], reverse=True)
        if limit is not None and limit > 0:
            items = items[:limit]
        return items

    def stats(
        self,
        *,
        windows: Iterable[float] = _DEFAULT_STATS_WINDOWS,
        now_wall: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = time.time() if now_wall is None else now_wall
        windows = sorted({float(w) for w in windows if w and w > 0})
        result: Dict[str, Any] = {}
        history = list(self._history)
        history.sort(key=lambda r: r.started_wall)
        starts = [r.started_wall for r in history]
        for window in windows:
            cutoff = now - window
            lo = bisect.bisect_left(starts, cutoff)
            recs = history[lo:]
            groups: Dict[Tuple[str, Any, str, Optional[str]], Dict[str, Any]] = {}
            for rec in recs:
                if rec.finished_monotonic is None:
                    continue
                key = (rec.listener, rec.port, rec.surface, rec.target)
                bucket = groups.get(key)
                if bucket is None:
                    bucket = {
                        "listener": rec.listener,
                        "port": rec.port,
                        "surface": rec.surface,
                        "target": rec.target,
                        "total": 0,
                        "by_status_class": {},
                        "errors": {},
                        "_durations_ms": [],
                    }
                    groups[key] = bucket
                bucket["total"] += 1
                cls = _status_class(rec.status)
                bucket["by_status_class"][cls] = bucket["by_status_class"].get(cls, 0) + 1
                if rec.error_type:
                    bucket["errors"][rec.error_type] = (
                        bucket["errors"].get(rec.error_type, 0) + 1
                    )
                duration = (rec.finished_monotonic - rec.started_monotonic) * 1000.0
                bucket["_durations_ms"].append(duration)

            group_list: List[Dict[str, Any]] = []
            for bucket in groups.values():
                durations = sorted(bucket.pop("_durations_ms"))
                bucket["p50_ms"] = _quantile_ms(durations, 0.5)
                bucket["p95_ms"] = _quantile_ms(durations, 0.95)
                bucket["max_ms"] = durations[-1] if durations else None
                group_list.append(bucket)
            group_list.sort(
                key=lambda g: (
                    str(g["listener"]),
                    str(g["port"]),
                    str(g["surface"]),
                    str(g["target"] or ""),
                )
            )
            result[str(int(window))] = {
                "window_seconds": window,
                "total": sum(g["total"] for g in group_list),
                "groups": group_list,
            }
        return {
            "now": now,
            "history_size": len(self._history),
            "history_capacity": self._history.maxlen or 0,
            "windows": result,
        }


class DashboardState:
    def __init__(self) -> None:
        self._samples: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._loaded_path_key: Optional[str] = None

    async def ensure_loaded(self, app: FastAPI) -> None:
        settings = app.state.settings
        path = _dashboard_data_path(settings)
        path_key = str(path) if path is not None else ""
        async with self._lock:
            if self._loaded_path_key == path_key:
                return

        now = time.time()
        retention = _dashboard_retention_seconds(settings)
        loaded: list[dict[str, Any]] = []
        if path is not None:
            loaded = await asyncio.to_thread(
                _read_dashboard_samples, path, retention=retention, now=now
            )

        async with self._lock:
            if self._loaded_path_key == path_key:
                return
            self._samples = _normalise_samples(
                self._samples + loaded,
                retention=retention,
                now=now,
            )
            self._loaded_path_key = path_key

    async def sample(self, app: FastAPI) -> dict[str, Any]:
        await self.ensure_loaded(app)
        settings = app.state.settings
        memory_free_mib, memory_total_mib = _memory_status_mib()
        coordinator: VramCoordinator | None = getattr(
            app.state, "vram_coordinator", None
        )
        if coordinator is None:
            vram_free_mib = None
            vram_total_mib = None
        else:
            vram_free_mib = await coordinator.free_vram_mib()
            vram_total_mib = await coordinator.total_vram_mib()

        current_models = _collect_model_snapshots(app)
        sample = {
            "ts": time.time(),
            "memory_free_mib": memory_free_mib,
            "memory_total_mib": memory_total_mib,
            "vram_free_mib": vram_free_mib,
            "vram_total_mib": vram_total_mib,
            "models": {
                str(model["key"]): model.get("estimated_vram_mib")
                for model in current_models
            },
            "models_memory": {
                str(model["key"]): model.get("estimated_memory_mib")
                for model in current_models
            },
        }
        retention = _dashboard_retention_seconds(settings)
        path = _dashboard_data_path(settings)
        async with self._lock:
            self._samples.append(sample)
            self._samples = _normalise_samples(
                self._samples,
                retention=retention,
                now=float(sample["ts"]),
            )
            samples_to_write = [_clone_sample(s) for s in self._samples]
        if path is not None:
            await asyncio.to_thread(_write_dashboard_samples, path, samples_to_write)
        return sample

    async def ensure_fresh(self, app: FastAPI) -> None:
        await self.ensure_loaded(app)
        settings = app.state.settings
        interval = max(1.0, float(settings.dashboard_sample_interval_seconds or 10.0))
        now = time.time()
        retention = _dashboard_retention_seconds(settings)
        async with self._lock:
            self._samples = _normalise_samples(
                self._samples,
                retention=retention,
                now=now,
            )
            latest = self._samples[-1]["ts"] if self._samples else 0.0
        if not latest or now - latest >= interval:
            await self.sample(app)

    async def data(self, app: FastAPI, *, range_seconds: float) -> dict[str, Any]:
        await self.ensure_fresh(app)
        now = time.time()
        cutoff = now - max(1.0, float(range_seconds))
        async with self._lock:
            samples = [_clone_sample(s) for s in self._samples if s["ts"] >= cutoff]
        settings = app.state.settings
        metrics: Optional[RequestMetrics] = getattr(
            app.state, "request_metrics", None
        )
        if metrics is None:
            inflight: List[Dict[str, Any]] = []
            request_stats: Dict[str, Any] = {
                "now": now,
                "history_size": 0,
                "history_capacity": 0,
                "windows": {},
            }
        else:
            inflight = metrics.inflight_snapshot(limit=200)
            request_stats = metrics.stats(now_wall=now)
        target_telemetry = await _collect_target_telemetry(app)
        return {
            "now": now,
            "range_seconds": range_seconds,
            "sample_interval_seconds": settings.dashboard_sample_interval_seconds,
            "retention_seconds": settings.dashboard_retention_seconds,
            "limits": {
                "vram_low_free_threshold_mib": settings.vram_low_free_threshold_mib,
            },
            "permissions": {
                "dashboard_model_reclaim_enabled": _dashboard_model_reclaim_enabled(
                    settings
                ),
            },
            "samples": samples,
            "current_models": _collect_model_snapshots(app),
            "inflight_requests": inflight,
            "request_stats": request_stats,
            "target_telemetry": target_telemetry,
        }


async def run_runtime_monitor(app: FastAPI) -> None:
    while True:
        settings = app.state.settings
        interval = max(1.0, float(settings.dashboard_sample_interval_seconds or 10.0))
        await asyncio.sleep(interval)
        try:
            if settings.dashboard_listener_enabled:
                await app.state.dashboard_state.sample(app)
            if settings.vram_low_free_reclaim_enabled:
                coordinator = getattr(app.state, "vram_coordinator", None)
                if coordinator is not None:
                    await coordinator.reclaim_if_below(
                        threshold_mib=settings.vram_low_free_threshold_mib
                    )
            if getattr(settings, "memory_low_free_reclaim_enabled", False):
                mem_coordinator = getattr(app.state, "memory_coordinator", None)
                if mem_coordinator is not None:
                    await mem_coordinator.reclaim_if_below(
                        threshold_mib=settings.memory_low_free_threshold_mib
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            import logging

            logging.getLogger("fake_ollama").exception(
                "runtime dashboard/VRAM monitor tick failed"
            )


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DASHBOARD_HTML = (_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")


def register_dashboard_routes(app: FastAPI) -> None:
    settings = app.state.settings
    if not settings.dashboard_listener_enabled:
        return

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard/", include_in_schema=False)
    async def dashboard_index() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/dashboard/data", include_in_schema=False)
    async def dashboard_data(
        request: Request,
        range_seconds: float = Query(default=3600.0, ge=1.0),
    ) -> JSONResponse:
        state: DashboardState = request.app.state.dashboard_state
        return JSONResponse(
            await state.data(request.app, range_seconds=range_seconds)
        )

    @app.post("/dashboard/reclaim-model", include_in_schema=False)
    async def dashboard_reclaim_model(request: Request) -> JSONResponse:
        settings = request.app.state.settings
        if not _dashboard_model_reclaim_enabled(settings):
            raise HTTPException(
                status_code=403,
                detail="dashboard model reclaim is disabled in settings",
            )

        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")

        key = str(payload.get("key") or "")
        if not key:
            raise HTTPException(status_code=400, detail="missing model key")

        snapshot = next(
            (
                item
                for item in _collect_model_snapshots(request.app)
                if item.get("key") == key
            ),
            None,
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="model is not currently loaded")

        target_id = str(snapshot.get("target_id") or "")
        model = str(snapshot.get("model") or "")
        if not target_id or not model:
            raise HTTPException(status_code=400, detail="invalid model snapshot")

        coordinator: VramCoordinator | None = getattr(
            request.app.state, "vram_coordinator", None
        )
        if coordinator is None:
            raise HTTPException(status_code=503, detail="VRAM coordinator unavailable")

        force = payload.get("force") is True
        result = await coordinator.reclaim_model(
            target_id=target_id,
            model=model,
            force=force,
        )
        status_code = 200 if result.get("released") else 409
        return JSONResponse(result, status_code=status_code)
