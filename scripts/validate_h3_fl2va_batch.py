"""Run and verify a real MiniMax H3 FL2VA Playground batch.

The Playground implements a multi-result request as independent, sequential
HTTP requests.  This script mirrors that behavior, keeps the first Context-IR
result for subsequent runs, samples NVML throughout the batch, and writes a
machine-readable manifest beside the output videos.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:21431/v1/videos/generations")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--size", default="768x768")
    parser.add_argument("--base-seed", type=int, default=2026080801)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument(
        "--ffmpeg", type=Path, default=Path(r"I:\Projects\Tools\ffmpeg\ffmpeg.exe")
    )
    return parser.parse_args()


def _nvml_sample() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        line = next(line for line in result.stdout.splitlines() if line.strip())
        total, used, free, utilization = [float(value.strip()) for value in line.split(",")[:4]]
        return {
            "total_mib": total,
            "used_mib": used,
            "free_mib": free,
            "utilization_percent": utilization,
        }
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        return None


def _playground_api_key(config_path: Path, model: str) -> str:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for interface in config.get("api_interfaces") or []:
        exposes_model = any(
            model in {str(item.get("model") or ""), str(item.get("alias") or "")}
            for item in interface.get("exposed_models") or []
        )
        tokens = [str(token) for token in interface.get("access_tokens") or [] if token]
        if exposes_model and tokens:
            return tokens[0]
    raise RuntimeError(
        f"no authenticated api_interface exposes {model!r} in {config_path}"
    )


def _probe_video(path: Path, ffmpeg: Path) -> dict[str, Any]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-f",
        "null",
        os.devnull,
    ]
    # Decoding the complete short result catches truncated/corrupt MP4s instead
    # of trusting container metadata alone.  The bundled ffmpeg build is used
    # because this workstation does not expose ffprobe on PATH.
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    video_match = re.search(
        r"Stream #\d+:\d+[^\r\n]*Video:[^\r\n]*?\b(\d{2,5})x(\d{2,5})\b",
        result.stderr,
    )
    if video_match is None:
        raise RuntimeError(f"ffmpeg found no video stream in {path}")
    width, height = (int(video_match.group(1)), int(video_match.group(2)))
    if width != 768 or height != 768:
        raise RuntimeError(
            f"unexpected video dimensions for {path}: {width}x{height}"
        )
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    duration = None
    if duration_match is not None:
        duration = (
            int(duration_match.group(1)) * 3600
            + int(duration_match.group(2)) * 60
            + float(duration_match.group(3))
        )
    return {
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "has_audio": bool(re.search(r"Stream #\d+:\d+[^\r\n]*Audio:", result.stderr)),
        "fully_decoded": True,
    }


def main() -> int:
    args = _args()
    if args.count != 4:
        raise ValueError("this acceptance script requires exactly four sequential outputs")
    for path in (args.first, args.last, args.prompt_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    api_key = _playground_api_key(args.config, "minimax-h3")
    if args.size != "768x768":
        raise ValueError("this acceptance script requires size=768x768")

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("prompt file is empty")
    first_bytes = args.first.read_bytes()
    last_bytes = args.last.read_bytes()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_id = str(uuid.uuid4())
    csv_path = args.output_dir / f"minimax-h3-fl2va-{stamp}-vram.csv"
    manifest_path = args.output_dir / f"minimax-h3-fl2va-{stamp}-manifest.json"
    stop_monitor = threading.Event()
    monitor_lock = threading.Lock()
    samples: list[dict[str, Any]] = []
    current_run = {"value": 0}

    def monitor() -> None:
        last_report = 0.0
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp_utc",
                    "run",
                    "total_mib",
                    "used_mib",
                    "free_mib",
                    "utilization_percent",
                ],
            )
            writer.writeheader()
            while not stop_monitor.is_set():
                sample = _nvml_sample()
                now = time.monotonic()
                if sample is not None:
                    row = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "run": current_run["value"],
                        **sample,
                    }
                    with monitor_lock:
                        samples.append(row)
                    writer.writerow(row)
                    handle.flush()
                    if now - last_report >= 30.0:
                        print(
                            f"VRAM run={row['run']}/4 used={row['used_mib']:.0f} MiB "
                            f"free={row['free_mib']:.0f} MiB gpu={row['utilization_percent']:.0f}%",
                            flush=True,
                        )
                        last_report = now
                stop_monitor.wait(1.0)

    monitor_thread = threading.Thread(target=monitor, name="h3-vram-monitor", daemon=True)
    monitor_thread.start()

    outputs: list[dict[str, Any]] = []
    revised_prompt = ""
    failure: str | None = None
    try:
        with httpx.Client(
            timeout=args.timeout,
            trust_env=False,
            headers={"x-api-key": api_key},
        ) as client:
            for run_index in range(1, args.count + 1):
                current_run["value"] = run_index
                run_started = time.monotonic()
                request_prompt = revised_prompt or prompt
                fields = {
                    "model": "minimax-h3",
                    "prompt": request_prompt,
                    "prompt_mode": "raw" if revised_prompt else "auto",
                    "context_ir_mode": "fl2va",
                    "size": args.size,
                    "n": "1",
                    "seed": str(args.base_seed + run_index - 1),
                    "response_format": "b64_json",
                    "generation_batch_id": batch_id,
                    "generation_run_index": str(run_index),
                    "generation_run_count": str(args.count),
                }
                files = [
                    ("image[]", (args.first.name, first_bytes, "image/png")),
                    ("image[]", (args.last.name, last_bytes, "image/png")),
                ]
                print(f"Starting FL2VA run {run_index}/{args.count}", flush=True)
                response = client.post(args.endpoint, data=fields, files=files)
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"run {run_index}/{args.count} returned HTTP {response.status_code}: "
                        f"{response.text[:2000]}"
                    )
                body = response.json()
                items = body.get("data") or []
                if len(items) != 1 or not items[0].get("b64_json"):
                    raise RuntimeError(f"run {run_index}/{args.count} returned no video: {body}")
                item = items[0]
                if not revised_prompt:
                    revised_prompt = str(item.get("revised_prompt") or "")
                media = base64.b64decode(item["b64_json"], validate=True)
                output_path = args.output_dir / f"minimax-h3-fl2va-{stamp}-{run_index:02d}.mp4"
                output_path.write_bytes(media)
                probe = _probe_video(output_path, args.ffmpeg)
                digest = hashlib.sha256(media).hexdigest()
                with monitor_lock:
                    run_samples = [row for row in samples if row["run"] == run_index]
                output = {
                    "run": run_index,
                    "seed": args.base_seed + run_index - 1,
                    "path": str(output_path),
                    "bytes": len(media),
                    "sha256": digest,
                    "elapsed_seconds": round(time.monotonic() - run_started, 3),
                    "minimum_free_vram_mib": min(
                        (row["free_mib"] for row in run_samples), default=None
                    ),
                    "maximum_used_vram_mib": max(
                        (row["used_mib"] for row in run_samples), default=None
                    ),
                    "media_probe": probe,
                }
                outputs.append(output)
                print(
                    f"Completed FL2VA run {run_index}/{args.count}: {output_path} "
                    f"({len(media)} bytes, sha256={digest[:12]})",
                    flush=True,
                )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=10.0)
        with monitor_lock:
            all_samples = list(samples)
        manifest = {
            "status": "success" if failure is None and len(outputs) == args.count else "failed",
            "failure": failure,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": args.endpoint,
            "model": "minimax-h3",
            "mode": "fl2va",
            "size": args.size,
            "count": args.count,
            "batch_id": batch_id,
            "first_image": str(args.first),
            "last_image": str(args.last),
            "prompt_file": str(args.prompt_file),
            "prompt": prompt,
            "revised_prompt": revised_prompt,
            "outputs": outputs,
            "vram_csv": str(csv_path),
            "minimum_free_vram_mib": min(
                (row["free_mib"] for row in all_samples), default=None
            ),
            "maximum_used_vram_mib": max(
                (row["used_mib"] for row in all_samples), default=None
            ),
            "sample_count": len(all_samples),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Manifest: {manifest_path}", flush=True)
        print(f"VRAM samples: {csv_path}", flush=True)

    if len(outputs) != args.count:
        raise RuntimeError(f"expected {args.count} outputs, got {len(outputs)}")
    if len({output["sha256"] for output in outputs}) != args.count:
        raise RuntimeError("the four output videos are not byte-distinct")
    print("SUCCESS: four verified FL2VA videos completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
