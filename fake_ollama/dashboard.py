"""Runtime dashboard for local model and memory telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .vram import VramCoordinator


_LOG = logging.getLogger("fake_ollama")
_DASHBOARD_DATA_VERSION = 1
_DASHBOARD_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def _memory_status_mib() -> tuple[Optional[float], Optional[float]]:
    if os.name == "nt":
        return _windows_memory_status_mib()
    proc = _proc_meminfo_status_mib()
    if proc != (None, None):
        return proc
    return _posix_memory_status_mib()


def _windows_memory_status_mib() -> tuple[Optional[float], Optional[float]]:
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        if not ok:
            return None, None
        return stat.ullAvailPhys / 1048576.0, stat.ullTotalPhys / 1048576.0
    except Exception:
        return None, None


def _proc_meminfo_status_mib() -> tuple[Optional[float], Optional[float]]:
    try:
        values: dict[str, float] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key not in ("MemAvailable", "MemFree", "MemTotal"):
                    continue
                raw = rest.strip().split()
                if raw:
                    values[key] = float(raw[0]) / 1024.0
        available = values.get("MemAvailable", values.get("MemFree"))
        total = values.get("MemTotal")
        return available, total
    except (OSError, ValueError):
        return None, None


def _posix_memory_status_mib() -> tuple[Optional[float], Optional[float]]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = (pages * page_size) / 1048576.0
        return None, total
    except (OSError, ValueError, AttributeError):
        return None, None


def _model_key(snapshot: dict[str, object]) -> str:
    return "|".join(
        str(snapshot.get(k) or "") for k in ("backend", "target_id", "model")
    )


def _collect_model_snapshots(app: FastAPI) -> list[dict[str, object]]:
    now = time.monotonic()
    snapshots: list[dict[str, object]] = []
    for clients in (
        getattr(app.state, "ollama_clients", {}),
        getattr(app.state, "llama_cpp_clients", {}),
    ):
        for client in list(clients.values()):
            getter = getattr(client, "loaded_model_snapshots", None)
            if not callable(getter):
                continue
            for raw in getter(now=now):
                snap = dict(raw)
                snap["key"] = _model_key(snap)
                snapshots.append(snap)
    snapshots.sort(key=lambda item: str(item.get("key") or ""))
    return snapshots


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
        except asyncio.CancelledError:
            raise
        except Exception:
            import logging

            logging.getLogger("fake_ollama").exception(
                "runtime dashboard/VRAM monitor tick failed"
            )


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fake-ollama dashboard</title>
<style>
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --ink: #172033;
  --muted: #647087;
  --line: #d8deea;
  --panel: #ffffff;
  --accent: #0f766e;
  --warn: #b45309;
  --danger: #b91c1c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 24px;
  border-bottom: 1px solid var(--line);
  background: #fff;
}
h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
}
main {
  width: min(1440px, 100%);
  margin: 0 auto;
  padding: 18px 24px 32px;
}
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
button {
  min-width: 48px;
  height: 34px;
  border: 1px solid #c8d0df;
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  cursor: pointer;
}
button.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}
button.icon {
  width: 30px;
  min-width: 30px;
  height: 30px;
  padding: 0;
  font-weight: 700;
  line-height: 1;
}
button.icon.danger {
  border-color: #fecaca;
  color: var(--danger);
}
.status {
  font-size: 13px;
  color: var(--muted);
}
.metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.metric, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.metric {
  padding: 12px 14px;
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.metric strong {
  display: block;
  margin-top: 4px;
  font-size: 22px;
  line-height: 1.2;
}
.panel {
  margin-top: 14px;
  padding: 14px;
}
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.panel h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}
canvas {
  display: block;
  width: 100%;
  height: 260px;
}
#modelChart {
  height: 320px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0 0;
}
.legend button {
  width: auto;
  min-width: 0;
  height: 30px;
  padding: 0 10px;
  border-color: var(--line);
}
.legend button.hidden {
  opacity: 0.42;
  text-decoration: line-through;
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: -1px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  padding: 9px 8px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}
th {
  color: var(--muted);
  font-weight: 600;
}
td.model {
  white-space: normal;
  overflow-wrap: anywhere;
}
td.action {
  text-align: center;
}
.empty {
  padding: 20px 6px;
  color: var(--muted);
}
@media (max-width: 760px) {
  header { align-items: flex-start; flex-direction: column; padding: 16px; }
  main { padding: 12px 12px 24px; }
  .metrics { grid-template-columns: 1fr; }
  canvas { height: 220px; }
  #modelChart { height: 260px; }
  th:nth-child(2), td:nth-child(2) { display: none; }
}
</style>
</head>
<body>
<header>
  <h1>fake-ollama dashboard</h1>
  <div class="toolbar" id="rangeButtons"></div>
</header>
<main>
  <div class="status" id="status">loading</div>
  <section class="metrics">
    <div class="metric"><span>Free RAM</span><strong id="ramNow">-</strong></div>
    <div class="metric"><span>Free VRAM</span><strong id="vramNow">-</strong></div>
    <div class="metric"><span>Loaded Models</span><strong id="modelNow">0</strong></div>
  </section>
  <section class="panel">
    <div class="panel-head"><h2>Available Memory</h2></div>
    <canvas id="memoryChart"></canvas>
  </section>
  <section class="panel">
    <div class="panel-head"><h2>Available VRAM</h2><span class="status" id="vramThreshold"></span></div>
    <canvas id="vramChart"></canvas>
  </section>
  <section class="panel">
    <div class="panel-head"><h2>Model Estimated VRAM</h2></div>
    <canvas id="modelChart"></canvas>
    <div class="legend" id="modelLegend"></div>
  </section>
  <section class="panel">
    <div class="panel-head"><h2>Current Models</h2></div>
    <div id="modelTable"></div>
  </section>
</main>
<script>
const ranges = [
  {label: '1m', seconds: 60},
  {label: '10m', seconds: 600},
  {label: '1h', seconds: 3600},
  {label: '6h', seconds: 21600},
  {label: '24h', seconds: 86400},
  {label: '7d', seconds: 604800},
];
let selectedRange = 3600;
let latest = null;
const hiddenModels = new Set();
const colors = ['#0f766e', '#2563eb', '#b45309', '#9333ea', '#dc2626', '#0891b2', '#4d7c0f', '#be185d', '#475569', '#ea580c'];

function $(id) { return document.getElementById(id); }
function escapeHtml(value) {
  const escapes = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
  return String(value ?? '').replace(/[&<>"']/g, ch => escapes[ch]);
}
function fmtGiB(mib) {
  if (mib === null || mib === undefined || Number.isNaN(Number(mib))) return '-';
  return (Number(mib) / 1024).toFixed(2) + ' GiB';
}
function fmtAge(sec) {
  if (sec === null || sec === undefined) return '-';
  if (sec < 60) return Math.round(sec) + 's';
  if (sec < 3600) return Math.round(sec / 60) + 'm';
  return (sec / 3600).toFixed(1) + 'h';
}
function modelLabel(model) {
  return `${model.model} / ${model.target_id}`;
}
function colorFor(i) { return colors[i % colors.length]; }
function maxSampleGapSeconds() {
  const interval = Number(latest && latest.sample_interval_seconds) || 10;
  return Math.max(interval * 2.5, interval + 5);
}

function setupRangeButtons() {
  const host = $('rangeButtons');
  host.innerHTML = '';
  for (const r of ranges) {
    const b = document.createElement('button');
    b.textContent = r.label;
    b.className = r.seconds === selectedRange ? 'active' : '';
    b.onclick = () => {
      selectedRange = r.seconds;
      setupRangeButtons();
      loadData();
    };
    host.append(b);
  }
}

function resizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function drawChart(canvas, samples, series, opts) {
  const {ctx, width, height} = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = {l: 56, r: 16, t: 30, b: 34};
  const plotW = Math.max(10, width - pad.l - pad.r);
  const plotH = Math.max(10, height - pad.t - pad.b);
  const now = latest ? latest.now : Date.now() / 1000;
  const minTs = now - selectedRange;
  const maxTs = now;
  let yMax = opts.yMax || 0;
  if (!yMax) {
    for (const s of samples) {
      for (const item of series) {
        const v = item.value(s);
        if (v !== null && v !== undefined) yMax = Math.max(yMax, Number(v));
      }
    }
    yMax = Math.max(1024, yMax * 1.15);
  }
  const x = ts => pad.l + ((ts - minTs) / (maxTs - minTs || 1)) * plotW;
  const y = v => pad.t + plotH - (Number(v) / (yMax || 1)) * plotH;

  ctx.strokeStyle = '#d8deea';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#647087';
  ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
  ctx.textBaseline = 'top';
  ctx.fillText('GiB', 8, 8);
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, yy);
    ctx.lineTo(width - pad.r, yy);
    ctx.stroke();
    const val = yMax * (1 - i / 4);
    ctx.fillText((val / 1024).toFixed(val >= 10240 ? 0 : 1), 8, yy);
  }
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 5; i++) {
    const ts = minTs + selectedRange * i / 5;
    const xx = x(ts);
    ctx.beginPath();
    ctx.moveTo(xx, pad.t);
    ctx.lineTo(xx, pad.t + plotH);
    ctx.stroke();
    const label = new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    ctx.fillText(label, Math.min(width - 58, Math.max(pad.l - 4, xx - 24)), pad.t + plotH + 12);
  }

  for (const item of series) {
    if (item.hidden) continue;
    const maxGap = maxSampleGapSeconds();
    const pts = samples
      .filter(s => s.ts >= minTs && s.ts <= maxTs)
      .map(s => ({ts: Number(s.ts), x: x(s.ts), y: y(item.value(s) || 0), v: item.value(s)}));
    if (!pts.length) continue;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    let last = null;
    for (const p of pts) {
      if (p.v === null || p.v === undefined) {
        started = false;
        last = null;
        continue;
      }
      if (last && p.ts - last.ts > maxGap) {
        started = false;
        last = null;
      }
      if (!started) {
        ctx.moveTo(p.x, p.y);
        started = true;
      } else if (opts.stepped && last) {
        ctx.lineTo(p.x, last.y);
        ctx.lineTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }
      last = p;
    }
    if (started) ctx.stroke();
  }
}

function renderLegend(models) {
  const legend = $('modelLegend');
  legend.innerHTML = '';
  models.forEach((m, i) => {
    const b = document.createElement('button');
    b.className = hiddenModels.has(m.key) ? 'hidden' : '';
    b.innerHTML = `<span class="swatch" style="background:${colorFor(i)}"></span>${escapeHtml(m.model)}`;
    b.title = modelLabel(m);
    b.onclick = () => {
      if (hiddenModels.has(m.key)) hiddenModels.delete(m.key);
      else hiddenModels.add(m.key);
      render();
    };
    legend.append(b);
  });
}

function renderTable(models) {
  if (!models.length) {
    $('modelTable').innerHTML = '<div class="empty">No loaded local models</div>';
    return;
  }
  const reclaimAllowed = Boolean(latest && latest.permissions && latest.permissions.dashboard_model_reclaim_enabled);
  const rows = models.map(m => {
    const canReclaim = reclaimAllowed && Boolean(m.reclaimable);
    const title = !reclaimAllowed
      ? 'Dashboard model reclaim is disabled in settings'
      : (m.reclaimable ? 'Close and reclaim this model' : 'Model is not eligible for reclaim yet');
    const disabled = canReclaim ? '' : ' disabled';
    return `<tr>
      <td class="model">${escapeHtml(m.model)}</td>
      <td>${escapeHtml(m.backend)}</td>
      <td>${escapeHtml(m.target_id)}</td>
      <td>${fmtGiB(m.estimated_vram_mib)}</td>
      <td>${m.active_requests || 0}</td>
      <td>${fmtAge(m.idle_seconds)}</td>
      <td>${m.reclaimable ? 'yes' : 'no'}</td>
      <td class="action"><button type="button" class="icon danger" data-reclaim-key="${escapeHtml(m.key)}" title="${escapeHtml(title)}"${disabled}>X</button></td>
    </tr>`;
  }).join('');
  $('modelTable').innerHTML = `<table>
    <thead><tr><th>Model</th><th>Backend</th><th>Target</th><th>Est. VRAM</th><th>Active</th><th>Idle</th><th>Reclaimable</th><th>Action</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
  for (const btn of $('modelTable').querySelectorAll('button[data-reclaim-key]')) {
    btn.addEventListener('click', () => reclaimModel(btn.getAttribute('data-reclaim-key') || ''));
  }
}

async function reclaimModel(key) {
  if (!key) return;
  const current = (latest && latest.current_models) || [];
  const model = current.find(m => m.key === key);
  const label = model ? modelLabel(model) : key;
  if (!window.confirm(`Close and reclaim ${label}?`)) return;
  $('status').textContent = `reclaiming ${label}...`;
  try {
    const resp = await fetch('/dashboard/reclaim-model', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({key}),
    });
    const text = await resp.text();
    let payload = {};
    try { payload = text ? JSON.parse(text) : {}; } catch (_) {}
    if (!resp.ok) {
      throw new Error(payload.detail || payload.reason || payload.error || text || resp.statusText);
    }
    $('status').textContent = `reclaim requested for ${label}`;
    await loadData();
  } catch (err) {
    $('status').textContent = 'reclaim failed: ' + err.message;
  }
}

function render() {
  if (!latest) return;
  const samples = latest.samples || [];
  const current = latest.current_models || [];
  const newest = samples[samples.length - 1] || {};
  $('ramNow').textContent = fmtGiB(newest.memory_free_mib);
  $('vramNow').textContent = fmtGiB(newest.vram_free_mib);
  $('modelNow').textContent = String(current.length);
  const threshold = latest.limits && latest.limits.vram_low_free_threshold_mib;
  $('vramThreshold').textContent = threshold ? `release threshold ${Math.round(threshold)} MiB` : '';

  const known = new Map();
  current.forEach(m => known.set(m.key, m));
  samples.forEach(s => {
    Object.keys(s.models || {}).forEach(k => {
      if (!known.has(k)) {
        const parts = k.split('|');
        known.set(k, {key: k, backend: parts[0], target_id: parts[1], model: parts.slice(2).join('|')});
      }
    });
  });
  const models = Array.from(known.values()).sort((a, b) => String(a.key).localeCompare(String(b.key)));
  renderLegend(models);
  renderTable(current);

  drawChart($('memoryChart'), samples, [
    {key: 'memory_free_mib', color: '#2563eb', value: s => s.memory_free_mib},
  ], {yMax: newest.memory_total_mib || 0});
  drawChart($('vramChart'), samples, [
    {key: 'vram_free_mib', color: '#0f766e', value: s => s.vram_free_mib},
  ], {yMax: newest.vram_total_mib || 0});
  drawChart($('modelChart'), samples, models.map((m, i) => ({
    key: m.key,
    color: colorFor(i),
    hidden: hiddenModels.has(m.key),
    value: s => (s.models && s.models[m.key] !== undefined) ? s.models[m.key] : 0,
  })), {stepped: true});
}

async function loadData() {
  try {
    const resp = await fetch(`/dashboard/data?range_seconds=${selectedRange}`, {cache: 'no-store'});
    if (!resp.ok) throw new Error(await resp.text());
    latest = await resp.json();
    $('status').textContent = `updated ${new Date(latest.now * 1000).toLocaleString()}`;
    render();
  } catch (err) {
    $('status').textContent = 'load failed: ' + err.message;
  }
}

setupRangeButtons();
loadData();
setInterval(loadData, 10000);
window.addEventListener('resize', render);
</script>
</body>
</html>
"""


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

        result = await coordinator.reclaim_model(target_id=target_id, model=model)
        status_code = 200 if result.get("released") else 409
        return JSONResponse(result, status_code=status_code)
