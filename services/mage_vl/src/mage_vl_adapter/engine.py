"""Lazy Mage-VL model ownership and sequential video analysis."""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

from .settings import AdapterSettings, AnalysisOptions, PreparedRequest
from .video import extract_segment_frames, format_timestamp, probe_duration, select_segment_windows


LOGGER = logging.getLogger("mage_vl_adapter")
SUMMARY_SEGMENTS_PER_PASS = 24
SUMMARY_INTERMEDIATE_MAX_TOKENS = 128


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
        duration = prepared.options.video_duration_seconds or probe_duration(
            self.settings.ffmpeg_path,
            prepared.video_path,
        )
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
