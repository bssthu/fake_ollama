"""FFmpeg duration probing, timeline selection, and bounded frame extraction."""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

from .settings import AdapterSettings, SegmentWindow


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _subprocess_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def probe_duration(ffmpeg_path: Path, video_path: Path) -> float:
    result = subprocess.run(
        [str(ffmpeg_path), "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        creationflags=_subprocess_flags(),
    )
    match = _DURATION_RE.search(result.stderr)
    if not match:
        raise RuntimeError("FFmpeg could not determine the uploaded video's duration")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0:
        raise RuntimeError("the uploaded video has no decodable duration")
    return duration


def select_segment_windows(
    duration: float, segment_seconds: float, max_segments: int
) -> tuple[list[SegmentWindow], int]:
    if duration <= 0 or segment_seconds <= 0 or max_segments <= 0:
        raise ValueError("duration, segment_seconds, and max_segments must be positive")
    # Containers commonly report a few extra hundredths of a second.  Do not
    # turn that muxing tail into a near-empty extra segment (which can then
    # displace a useful middle window when max_segments is applied).
    tail_tolerance = min(0.25, segment_seconds * 0.1)
    total = max(1, math.ceil(max(0.0, duration - tail_tolerance) / segment_seconds))
    if total <= max_segments:
        indices = list(range(total))
    elif max_segments == 1:
        indices = [0]
    else:
        indices = []
        for position in range(max_segments):
            index = round(position * (total - 1) / (max_segments - 1))
            if not indices or index != indices[-1]:
                indices.append(index)
    windows = [
        SegmentWindow(
            index=index,
            start=index * segment_seconds,
            duration=min(segment_seconds, duration - index * segment_seconds),
        )
        for index in indices
    ]
    return windows, total - len(windows)


def extract_segment_frames(
    settings: AdapterSettings,
    video_path: Path,
    window: SegmentWindow,
    frame_count: int,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rate = frame_count / max(window.duration, 0.05)
    pattern = output_dir / "frame-%03d.jpg"
    video_filter = (
        f"fps={frame_rate:.8f},scale=min({settings.max_frame_width}\\,iw):-2"
    )
    command = [
        str(settings.ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{window.start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{window.duration:.3f}",
        "-vf",
        video_filter,
        "-frames:v",
        str(frame_count),
        "-q:v",
        "3",
        str(pattern),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(60, int(window.duration * 5)),
        check=False,
        creationflags=_subprocess_flags(),
    )
    frames = sorted(output_dir.glob("frame-*.jpg"))
    if result.returncode != 0 or not frames:
        detail = result.stderr.strip()[-1000:] or f"exit status {result.returncode}"
        raise RuntimeError(f"FFmpeg failed to decode segment: {detail}")
    return frames


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
