"""Serve local Mage-VL video analysis through an OpenAI-compatible endpoint.

The adapter intentionally uses Mage-VL's frame-sampled path for the first
Windows-native deployment.  FFmpeg decodes bounded timeline windows, while
Transformers runs the official remote-code model with PyTorch SDPA.  The
optional proactive StreamMind gate is not loaded here because its Mamba
dependency is not currently practical on native Windows.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Iterator, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


LOGGER = logging.getLogger("mage_vl_adapter")
DEFAULT_MODEL_DIR = Path(r"J:\Projects\LLM_Models\Mage\Mage-VL")
DEFAULT_FFMPEG = Path(r"I:\Projects\Tools\ffmpeg\ffmpeg.exe")
DEFAULT_TEMP_ROOT = Path(r"I:\Projects\fake_ollama\.tmp\mage-vl-runtime")
DEFAULT_MODEL_ID = "mage-vl-local"
DEFAULT_MAX_VIDEO_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_SEGMENTS_LIMIT = 120
SUMMARY_SEGMENTS_PER_PASS = 24
SUMMARY_INTERMEDIATE_MAX_TOKENS = 128
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)?(?P<base64>;base64)?,(?P<data>.*)$", re.DOTALL
)


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


def _bounded_number(
    payload: Mapping[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
) -> int | float:
    raw = payload.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"'{key}' must be numeric")
    if not math.isfinite(value):
        raise HTTPException(status_code=400, detail=f"'{key}' must be finite")
    value = min(maximum, max(minimum, value))
    return int(value) if integer else value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, Mapping) and part.get("type") in {
            "text",
            "input_text",
        }:
            chunks.append(str(part.get("text") or part.get("content") or ""))
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


def _video_urls(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    urls: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") not in {"video_url", "input_video"}:
            continue
        value = part.get("video_url")
        if value is None:
            value = part.get("video")
        if isinstance(value, Mapping):
            value = value.get("url") or value.get("data")
        if isinstance(value, str) and value:
            urls.append(value)
    return urls


def _parse_messages(payload: Mapping[str, Any]) -> tuple[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    system_chunks: list[str] = []
    last_user: Mapping[str, Any] | None = None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "")
        if role == "system":
            text = _content_text(message.get("content"))
            if text:
                system_chunks.append(text)
        elif role == "user":
            last_user = message
    if last_user is None:
        raise HTTPException(status_code=400, detail="a user message is required")

    content = last_user.get("content")
    urls = _video_urls(content)
    if not urls:
        raise HTTPException(
            status_code=400,
            detail="the last user message must contain one video_url part",
        )
    if len(urls) != 1:
        raise HTTPException(status_code=400, detail="exactly one video is supported")

    user_prompt = _content_text(content) or "Describe the important events in this video."
    if system_chunks:
        prompt = "System instruction:\n" + "\n".join(system_chunks)
        prompt += "\n\nUser request:\n" + user_prompt
    else:
        prompt = user_prompt
    return prompt, urls[0]


def _analysis_options(
    payload: Mapping[str, Any], settings: AdapterSettings
) -> AnalysisOptions:
    raw_summary = payload.get("include_summary", False)
    if isinstance(raw_summary, str):
        include_summary = raw_summary.strip().lower() in {"1", "true", "yes", "on"}
    else:
        include_summary = bool(raw_summary)
    return AnalysisOptions(
        segment_seconds=float(
            _bounded_number(
                payload,
                "segment_seconds",
                settings.default_segment_seconds,
                2.0,
                60.0,
            )
        ),
        frames_per_segment=int(
            _bounded_number(
                payload,
                "frames_per_segment",
                settings.default_frames_per_segment,
                2,
                32,
                integer=True,
            )
        ),
        max_segments=int(
            _bounded_number(
                payload,
                "max_segments",
                settings.default_max_segments,
                1,
                settings.max_segments_limit,
                integer=True,
            )
        ),
        max_new_tokens=int(
            _bounded_number(payload, "max_tokens", 256, 16, 512, integer=True)
        ),
        temperature=float(
            _bounded_number(payload, "temperature", 0.0, 0.0, 2.0)
        ),
        include_summary=include_summary,
    )


def _decode_data_video(data_url: str, settings: AdapterSettings, request_dir: Path) -> Path:
    if data_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="remote video URLs are disabled; upload a local video file",
        )
    match = _DATA_URI_RE.match(data_url)
    if not match or not match.group("base64"):
        raise HTTPException(
            status_code=400,
            detail="video_url must be a base64 data URL",
        )
    encoded = match.group("data")
    estimated_size = (len(encoded) * 3) // 4
    if estimated_size > settings.max_video_bytes:
        limit_mib = settings.max_video_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"video exceeds the configured {limit_mib:.0f} MiB limit",
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 video data") from exc
    if not raw:
        raise HTTPException(status_code=400, detail="video upload is empty")
    if len(raw) > settings.max_video_bytes:
        raise HTTPException(status_code=413, detail="video exceeds the configured size limit")

    mime = match.group("mime") or "video/mp4"
    suffix = mimetypes.guess_extension(mime) or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg"}:
        suffix = ".mp4"
    video_path = request_dir / f"upload{suffix}"
    video_path.write_bytes(raw)
    return video_path


def prepare_request(payload: Mapping[str, Any], settings: AdapterSettings) -> PreparedRequest:
    prompt, data_url = _parse_messages(payload)
    options = _analysis_options(payload, settings)
    settings.temp_root.mkdir(parents=True, exist_ok=True)
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=settings.temp_root))
    try:
        video_path = _decode_data_video(data_url, settings, request_dir)
    except BaseException:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    return PreparedRequest(request_dir, video_path, prompt, options)


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


class MageEngine:
    """Lazy, single-process owner of the Mage-VL model and processor."""

    def __init__(self, settings: AdapterSettings) -> None:
        self.settings = settings
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._load_lock = threading.Lock()
        self.status = "not_loaded"
        self.last_error = ""

    def validate_runtime(self) -> list[str]:
        problems: list[str] = []
        if not self.settings.model_dir.is_dir():
            problems.append(f"model directory is missing: {self.settings.model_dir}")
        else:
            required_model_files = (
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
            )
            missing_model_files = [
                name
                for name in required_model_files
                if not (self.settings.model_dir / name).is_file()
            ]
            if missing_model_files:
                problems.append(
                    "model files are missing: " + ", ".join(missing_model_files)
                )
        if not self.settings.ffmpeg_path.is_file():
            problems.append(f"FFmpeg is missing: {self.settings.ffmpeg_path}")
        return problems

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            problems = self.validate_runtime()
            if problems:
                raise RuntimeError("; ".join(problems))
            self.status = "loading"
            self.last_error = ""
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable to PyTorch")
                torch.set_float32_matmul_precision("high")
                LOGGER.info("loading Mage-VL from %s", self.settings.model_dir)
                processor = AutoProcessor.from_pretrained(
                    self.settings.model_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                ).eval()
                self._torch = torch
                self._processor = processor
                self._model = model
                self.status = "ready"
                LOGGER.info(
                    "Mage-VL ready on %s (%s)",
                    model.device,
                    torch.cuda.get_device_name(0),
                )
            except BaseException as exc:
                self.status = "error"
                self.last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("failed to load Mage-VL")
                raise

    def _generate(self, prompt: str, frame_paths: Sequence[Path], options: AnalysisOptions) -> str:
        self.ensure_loaded()
        from PIL import Image

        frames = []
        try:
            for frame_path in frame_paths:
                with Image.open(frame_path) as source:
                    frame = source.convert("RGB")
                    frame.load()
                    frames.append(frame)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], videos=[frames], return_tensors="pt", padding=True
            )
            inputs = {
                key: (value.to(self._model.device) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
            generation: dict[str, Any] = {
                "max_new_tokens": min(options.max_new_tokens, 192),
                "do_sample": options.temperature > 0,
            }
            if options.temperature > 0:
                generation["temperature"] = options.temperature
            with self._torch.inference_mode():
                output = self._model.generate(**inputs, **generation)
            answer = self._processor.tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return answer.strip() or "（该时间段未生成文字说明）"
        finally:
            for frame in frames:
                frame.close()

    def _summarize_once(
        self,
        timeline: str,
        prompt: str,
        max_new_tokens: int,
        *,
        intermediate: bool,
    ) -> str:
        self.ensure_loaded()
        instruction = (
            "请把下面一组相邻时间段压缩为忠实的中间摘要，保留时间顺序和关键事件，"
            "不要补充原文中没有的事实。"
            if intermediate
            else "请根据下面逐段视频分析给出简洁的整体结论，指出关键事件、时间顺序，"
            "不要补充逐段结果中没有的事实。"
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    f"原始问题：{prompt}\n\n逐段分析：\n{timeline}"
                ),
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {
            key: (value.to(self._model.device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 256),
                do_sample=False,
            )
        return self._processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def _summarize(
        self,
        timeline_parts: Sequence[str],
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Summarize with a bounded per-call context, independent of segment count.

        Each segment generation is capped at 192 new tokens.  Batching at most
        24 adjacent entries per intermediate pass keeps summary KV-cache demand
        below the former 30-segment one-shot path even when max_segments is
        raised.  Additional segments add sequential calls and CPU text only.
        """

        level = list(timeline_parts)
        while len(level) > SUMMARY_SEGMENTS_PER_PASS:
            next_level: list[str] = []
            for offset in range(0, len(level), SUMMARY_SEGMENTS_PER_PASS):
                batch = level[offset : offset + SUMMARY_SEGMENTS_PER_PASS]
                summary = self._summarize_once(
                    "\n\n".join(batch),
                    prompt,
                    min(max_new_tokens, SUMMARY_INTERMEDIATE_MAX_TOKENS),
                    intermediate=True,
                )
                if summary:
                    next_level.append(summary)
            if not next_level:
                return ""
            level = next_level
        return self._summarize_once(
            "\n\n".join(level),
            prompt,
            max_new_tokens,
            intermediate=False,
        )

    def analyze(self, prepared: PreparedRequest) -> Iterator[str]:
        duration = probe_duration(self.settings.ffmpeg_path, prepared.video_path)
        windows, skipped = select_segment_windows(
            duration,
            prepared.options.segment_seconds,
            prepared.options.max_segments,
        )
        intro = (
            f"视频时长 {format_timestamp(duration)}；按 {prepared.options.segment_seconds:g} 秒窗口"
            f"分析 {len(windows)} 段，每段抽取 {prepared.options.frames_per_segment} 帧。"
        )
        if skipped:
            intro += f" 视频较长，已在时间轴上均匀选取窗口，跳过 {skipped} 段。"
        yield intro + "\n\n"

        timeline_parts: list[str] = []
        for sequence, window in enumerate(windows, start=1):
            frame_dir = prepared.request_dir / f"segment-{sequence:03d}"
            frame_paths = extract_segment_frames(
                self.settings,
                prepared.video_path,
                window,
                prepared.options.frames_per_segment,
                frame_dir,
            )
            segment_prompt = (
                f"你正在分析视频时间段 {format_timestamp(window.start)} 到 "
                f"{format_timestamp(window.end)}。请围绕用户问题描述这一段中实际可见的"
                f"动作、对象、场景变化和关键事件；不确定时明确说明。\n\n"
                f"用户问题：{prepared.prompt}"
            )
            answer = self._generate(segment_prompt, frame_paths, prepared.options)
            section = (
                f"### {format_timestamp(window.start)}–{format_timestamp(window.end)}\n"
                f"{answer}"
            )
            timeline_parts.append(section)
            yield section + "\n\n"
            shutil.rmtree(frame_dir, ignore_errors=True)

        if prepared.options.include_summary and len(timeline_parts) > 1:
            summary = self._summarize(
                timeline_parts,
                prepared.prompt,
                prepared.options.max_new_tokens,
            )
            if summary:
                yield f"## 整体结论\n{summary}\n"


def _openai_chunk(
    request_id: str,
    model: str,
    *,
    content: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content:
        delta["content"] = content
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def _sse(data: Mapping[str, Any] | str) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _run_blocking(function: Any, *args: Any) -> Any:
    """Finish an in-flight GPU call before allowing another request after cancel."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException:
            pass
        raise


_ITERATOR_END = object()


async def _next_or_end(iterator: Iterator[str]) -> tuple[bool, str]:
    value = await _run_blocking(next, iterator, _ITERATOR_END)
    if value is _ITERATOR_END:
        return True, ""
    return False, str(value)


def create_app(
    settings: AdapterSettings | None = None, engine: MageEngine | None = None
) -> FastAPI:
    adapter_settings = settings or AdapterSettings.from_env()
    mage = engine or MageEngine(adapter_settings)
    app = FastAPI(title="Mage-VL Local Adapter", version="1.0")
    app.state.settings = adapter_settings
    app.state.engine = mage
    app.state.inference_lock = asyncio.Lock()
    app.state.uvicorn_server = None

    @app.get("/health")
    async def health() -> JSONResponse:
        problems = mage.validate_runtime()
        body = {
            "status": "ok" if not problems else "error",
            "model": adapter_settings.model_id,
            "model_status": mage.status,
            "model_dir": str(adapter_settings.model_dir),
            "ffmpeg": str(adapter_settings.ffmpeg_path),
        }
        if problems:
            body["problems"] = problems
            return JSONResponse(body, status_code=503)
        if mage.last_error:
            body["last_error"] = mage.last_error
        return JSONResponse(body)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": adapter_settings.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "microsoft",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="request body must be an object")
        requested_model = str(payload.get("model") or adapter_settings.model_id)
        if requested_model != adapter_settings.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {requested_model}")
        prepared = prepare_request(payload, adapter_settings)
        request_id = "chatcmpl-mage-" + uuid.uuid4().hex

        if bool(payload.get("stream", False)):
            async def stream_body() -> AsyncIterator[str]:
                try:
                    yield _sse(
                        _openai_chunk(
                            request_id,
                            adapter_settings.model_id,
                            role="assistant",
                        )
                    )
                    async with app.state.inference_lock:
                        iterator = iter(mage.analyze(prepared))
                        while True:
                            ended, content = await _next_or_end(iterator)
                            if ended:
                                break
                            yield _sse(
                                _openai_chunk(
                                    request_id,
                                    adapter_settings.model_id,
                                    content=content,
                                )
                            )
                    yield _sse(
                        _openai_chunk(
                            request_id,
                            adapter_settings.model_id,
                            finish_reason="stop",
                        )
                    )
                    yield _sse("[DONE]")
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    LOGGER.exception("video analysis failed")
                    yield _sse(
                        {
                            "error": {
                                "message": f"{type(exc).__name__}: {exc}",
                                "type": "mage_vl_error",
                            }
                        }
                    )
                    yield _sse("[DONE]")
                finally:
                    shutil.rmtree(prepared.request_dir, ignore_errors=True)

            return StreamingResponse(
                stream_body(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            async with app.state.inference_lock:
                parts = await _run_blocking(lambda: list(mage.analyze(prepared)))
        except BaseException as exc:
            LOGGER.exception("video analysis failed")
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            shutil.rmtree(prepared.request_dir, ignore_errors=True)
        content = "".join(parts)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": adapter_settings.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, Any]:
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=403, detail="shutdown is loopback-only")
        server = app.state.uvicorn_server
        if server is None:
            return {"status": "accepted", "detail": "test server has no runtime owner"}
        asyncio.get_running_loop().call_later(
            0.2, lambda: setattr(server, "should_exit", True)
        )
        return {"status": "accepted", "detail": "graceful shutdown requested"}

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8071)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn

    app = create_app()
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
        timeout_graceful_shutdown=None,
    )
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
