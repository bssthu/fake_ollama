"""Configuration and request data structures for the Mage-VL service."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_LOCAL_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
) / "fake-ollama" / "mage-vl"
DEFAULT_MODEL_DIR = _DEFAULT_LOCAL_ROOT / "model"
DEFAULT_FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
DEFAULT_TEMP_ROOT = _DEFAULT_LOCAL_ROOT / "runtime"
DEFAULT_MODEL_ID = "mage-vl-local"
DEFAULT_MAX_VIDEO_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_SEGMENTS_LIMIT = 120


@dataclass(frozen=True)
class AdapterSettings:
    model_dir: Path = DEFAULT_MODEL_DIR
    ffmpeg_path: Path = DEFAULT_FFMPEG
    temp_root: Path = DEFAULT_TEMP_ROOT
    model_id: str = DEFAULT_MODEL_ID
    max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES
    default_segment_seconds: float = 8.0
    default_frames_per_segment: int = 8
    default_max_segments: int = 12
    max_segments_limit: int = DEFAULT_MAX_SEGMENTS_LIMIT
    max_frame_width: int = 768

    @classmethod
    def from_env(cls) -> "AdapterSettings":
        return cls(
            model_dir=Path(os.environ.get("MAGE_VL_MODEL_DIR", str(DEFAULT_MODEL_DIR))),
            ffmpeg_path=Path(os.environ.get("MAGE_VL_FFMPEG", str(DEFAULT_FFMPEG))),
            temp_root=Path(os.environ.get("MAGE_VL_TEMP_DIR", str(DEFAULT_TEMP_ROOT))),
            model_id=os.environ.get("MAGE_VL_MODEL_ID", DEFAULT_MODEL_ID),
            max_video_bytes=_env_int(
                "MAGE_VL_MAX_VIDEO_BYTES", DEFAULT_MAX_VIDEO_BYTES, minimum=1024
            ),
            default_segment_seconds=_env_float(
                "MAGE_VL_SEGMENT_SECONDS", 8.0, minimum=1.0
            ),
            default_frames_per_segment=_env_int(
                "MAGE_VL_FRAMES_PER_SEGMENT", 8, minimum=1
            ),
            default_max_segments=_env_int("MAGE_VL_MAX_SEGMENTS", 12, minimum=1),
            max_segments_limit=_env_int(
                "MAGE_VL_MAX_SEGMENTS_LIMIT",
                DEFAULT_MAX_SEGMENTS_LIMIT,
                minimum=1,
            ),
            max_frame_width=_env_int("MAGE_VL_MAX_FRAME_WIDTH", 768, minimum=224),
        )


@dataclass(frozen=True)
class AnalysisOptions:
    segment_seconds: float
    frames_per_segment: int
    max_segments: int
    max_new_tokens: int
    temperature: float
    include_summary: bool
    video_duration_seconds: float | None


@dataclass(frozen=True)
class SegmentWindow:
    index: int
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class PreparedRequest:
    request_dir: Path
    video_path: Path
    prompt: str
    options: AnalysisOptions


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)
