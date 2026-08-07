"""OpenAI payload parsing and bounded local video upload preparation."""

from __future__ import annotations

import base64
import binascii
import math
import mimetypes
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from fastapi import HTTPException

from .settings import AdapterSettings, AnalysisOptions, PreparedRequest


_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)?(?P<base64>;base64)?,(?P<data>.*)$", re.DOTALL
)


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
    raw_duration = payload.get("video_duration_seconds")
    video_duration_seconds = None
    if raw_duration is not None:
        video_duration_seconds = float(
            _bounded_number(
                payload,
                "video_duration_seconds",
                0.1,
                0.1,
                3600.0,
            )
        )
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
        video_duration_seconds=video_duration_seconds,
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
