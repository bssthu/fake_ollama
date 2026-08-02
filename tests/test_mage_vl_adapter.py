"""Tests for the lightweight Mage-VL OpenAI adapter surface."""

from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient

from mage_vl_adapter.server import (
    AdapterSettings,
    MageEngine,
    PreparedRequest,
    create_app,
    prepare_request,
    select_segment_windows,
)


class _FakeEngine:
    status = "ready"
    last_error = ""

    def validate_runtime(self) -> list[str]:
        return []

    def analyze(self, prepared: PreparedRequest) -> Iterator[str]:
        assert prepared.video_path.read_bytes() == b"fake-video"
        assert prepared.prompt == "分析关键动作"
        yield "视频时长 00:02。\n\n"
        yield "### 00:00–00:02\n测试结果\n"


def _settings(tmp_path: Path) -> AdapterSettings:
    return AdapterSettings(
        model_dir=tmp_path / "model",
        ffmpeg_path=tmp_path / "ffmpeg.exe",
        temp_root=tmp_path / "runtime",
    )


def _payload(*, stream: bool) -> dict[str, Any]:
    encoded = base64.b64encode(b"fake-video").decode("ascii")
    return {
        "model": "mage-vl-local",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "分析关键动作"},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
                    },
                ],
            }
        ],
        "stream": stream,
    }


def test_segment_windows_cover_short_video_and_sample_long_video() -> None:
    short, skipped = select_segment_windows(17.0, 8.0, 12)
    assert [(item.start, item.duration) for item in short] == [
        (0.0, 8.0),
        (8.0, 8.0),
        (16.0, 1.0),
    ]
    assert skipped == 0

    long, skipped = select_segment_windows(100.0, 8.0, 3)
    assert [item.index for item in long] == [0, 6, 12]
    assert skipped == 10

    rounded_container, skipped = select_segment_windows(4.03, 2.0, 2)
    assert [(item.start, item.duration) for item in rounded_container] == [
        (0.0, 2.0),
        (2.0, 2.0),
    ]
    assert skipped == 0


def test_runtime_validation_requires_both_weight_shards(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.model_dir.mkdir()
    settings.ffmpeg_path.write_bytes(b"")
    for name in (
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        (settings.model_dir / name).write_bytes(b"")

    engine = MageEngine(settings)
    assert engine.validate_runtime() == [
        "model files are missing: model-00002-of-00002.safetensors"
    ]

    (settings.model_dir / "model-00002-of-00002.safetensors").write_bytes(b"")
    assert engine.validate_runtime() == []


def test_request_segment_limit_defaults_to_120(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    payload = _payload(stream=False)
    payload["max_segments"] = 999
    prepared = prepare_request(payload, settings)
    try:
        assert prepared.options.max_segments == 120
    finally:
        shutil.rmtree(prepared.request_dir)


def test_large_summary_is_hierarchical_and_bounded(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = MageEngine(_settings(tmp_path))
    calls: list[tuple[list[str], int, bool]] = []

    def fake_summarize_once(
        timeline: str,
        prompt: str,
        max_new_tokens: int,
        *,
        intermediate: bool,
    ) -> str:
        assert prompt == "测试问题"
        parts = timeline.split("\n\n")
        calls.append((parts, max_new_tokens, intermediate))
        return f"summary-{len(calls)}"

    monkeypatch.setattr(engine, "_summarize_once", fake_summarize_once)
    result = engine._summarize(
        [f"segment-{index}" for index in range(120)],
        "测试问题",
        256,
    )

    assert result == "summary-6"
    assert len(calls) == 6
    assert all(len(parts) <= 24 for parts, _, _ in calls)
    assert all(tokens == 128 and intermediate for _, tokens, intermediate in calls[:5])
    assert calls[-1][1:] == (256, False)


def test_non_streaming_openai_response_and_temp_cleanup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings, _FakeEngine())  # type: ignore[arg-type]
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post("/v1/chat/completions", json=_payload(stream=False))

    assert health.status_code == 200
    assert response.status_code == 200
    assert response.json()["model"] == "mage-vl-local"
    assert "测试结果" in response.json()["choices"][0]["message"]["content"]
    assert not list(settings.temp_root.glob("request-*"))


def test_streaming_openai_response(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings, _FakeEngine())  # type: ignore[arg-type]
    with TestClient(app) as client:
        with client.stream(
            "POST", "/v1/chat/completions", json=_payload(stream=True)
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"role": "assistant"' in body
    assert "测试结果" in body
    assert "data: [DONE]" in body
    assert not list(settings.temp_root.glob("request-*"))


def test_remote_video_url_is_rejected(tmp_path: Path) -> None:
    payload = _payload(stream=False)
    payload["messages"][0]["content"][1]["video_url"]["url"] = (
        "https://example.test/video.mp4"
    )
    app = create_app(_settings(tmp_path), _FakeEngine())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "remote video URLs are disabled" in response.json()["detail"]
