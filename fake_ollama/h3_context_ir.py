"""Deterministic, provider-agnostic H3 prompt planning helpers.

The real MiniMax H3-Context-IR is a hosted, multi-stage system.  This module
implements the deliberately smaller local contract used by fake_ollama:

* ask an arbitrary configured chat/VL model for a strict JSON shot plan;
* validate the plan independently from the model;
* render the validated plan into MiniMax's public base-mode prompt format;
* fall back to a conservative one-shot prompt when a provider cannot produce
  valid JSON and the configured failure policy allows it.

Provider invocation and media wire formats live in ``server.py`` so this file
stays pure and can be unit-tested without starting model processes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


H3BaseMode = Literal["t2va", "i2va", "fl2va", "l2va"]
VALID_H3_BASE_MODES = ("auto", "t2va", "i2va", "fl2va", "l2va")


class H3BaseShot(BaseModel):
    """One shot in playback order."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    description: str

    @field_validator("description")
    @classmethod
    def _description_required(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("shot description must not be empty")
        return value


class H3BasePlan(BaseModel):
    """Small typed IR rendered to H3's public three-field base format."""

    model_config = ConfigDict(extra="forbid")

    mode: H3BaseMode
    duration_seconds: float = Field(ge=4, le=15)
    shots: List[H3BaseShot] = Field(min_length=1, max_length=12)
    overall_soundscape: str
    non_diegetic_music: str

    @field_validator("overall_soundscape", "non_diegetic_music")
    @classmethod
    def _audio_field_required(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("audio fields must contain a description or N/A")
        return value


def resolve_base_mode(requested_mode: str, image_count: int) -> H3BaseMode:
    """Resolve ``auto`` from the number of reference/keyframe images."""

    mode = (requested_mode or "auto").strip().lower()
    if mode not in VALID_H3_BASE_MODES:
        raise ValueError(
            f"mode must be one of {', '.join(VALID_H3_BASE_MODES)}, got {mode!r}"
        )
    if mode == "auto":
        if image_count == 0:
            return "t2va"
        if image_count == 1:
            return "i2va"
        if image_count == 2:
            return "fl2va"
        raise ValueError("base H3 Context-IR accepts at most two keyframe images")

    expected_images = {
        "t2va": 0,
        "i2va": 1,
        "fl2va": 2,
        "l2va": 1,
    }[mode]
    if image_count != expected_images:
        raise ValueError(
            f"mode {mode} requires {expected_images} image(s), got {image_count}"
        )
    return mode  # type: ignore[return-value]


def _strip_model_wrappers(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    fence = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1).strip()
    return value


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract one JSON object from common chat-model wrappers."""

    value = _strip_model_wrappers(text)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("provider response did not contain a JSON object")
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"provider returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("provider JSON root must be an object")
    return parsed


def parse_and_validate_plan(
    text: str,
    *,
    expected_mode: H3BaseMode,
    expected_duration_seconds: float,
) -> H3BasePlan:
    """Parse a provider response and enforce request-owned invariants."""

    plan = H3BasePlan.model_validate(extract_json_object(text))
    if plan.mode != expected_mode:
        raise ValueError(
            f"provider changed mode from {expected_mode!r} to {plan.mode!r}"
        )
    if abs(plan.duration_seconds - expected_duration_seconds) > 0.01:
        raise ValueError(
            "provider changed duration_seconds from "
            f"{expected_duration_seconds:.2f} to {plan.duration_seconds:.2f}"
        )
    starts = [float(shot.start_seconds) for shot in plan.shots]
    if abs(starts[0]) > 0.001:
        raise ValueError("the first shot must start at 0 seconds")
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError("shot start_seconds values must be strictly increasing")
    if any(start >= expected_duration_seconds for start in starts):
        raise ValueError("every shot must start before the requested duration")
    return plan


def _format_cut_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    minutes, remainder = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def render_base_prompt(plan: H3BasePlan) -> str:
    """Render a validated plan to MiniMax's documented base prompt format."""

    duration = plan.duration_seconds
    instruction = ""
    if plan.mode == "i2va":
        instruction = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        )
    elif plan.mode == "fl2va":
        instruction = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
            f"target video; Picture 2 (from Shot {len(plan.shots)}) aligns with "
            f"the {duration:.2f}-second mark of the target video.\n\n"
        )
    elif plan.mode == "l2va":
        instruction = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {len(plan.shots)}]) aligns with the "
            f"{duration:.2f}-second mark of the target video.\n\n"
        )

    shot_parts: List[str] = []
    for index, shot in enumerate(plan.shots, start=1):
        description = shot.description.strip()
        if index == 1:
            shot_parts.append(f"[Shot 1] {description}")
        else:
            shot_parts.append(
                f"[Shot {index}] At {_format_cut_time(shot.start_seconds)}, "
                f"the camera cuts to {description}"
            )
    integrated = " ".join(shot_parts)
    return (
        f"{instruction}integrated_multimodal_description: {integrated}\n"
        f"overall_soundscape: {plan.overall_soundscape.strip()}\n"
        f"non_diegetic_music: {plan.non_diegetic_music.strip()}"
    )


def fallback_plan(
    prompt: str,
    *,
    mode: H3BaseMode,
    duration_seconds: float,
) -> H3BasePlan:
    """Return a lossless one-shot plan when the planning model fails."""

    description = (prompt or "").strip()
    if mode == "i2va":
        description = (
            "Preserve the subject identity, composition, clothing, colors, and "
            f"scene shown in <Picture 1>, then develop naturally: {description}"
        )
    elif mode == "fl2va":
        description = (
            "Begin exactly from <Picture 1>, preserve subject and scene "
            "continuity, and move continuously toward <Picture 2> at the end: "
            f"{description}"
        )
    elif mode == "l2va":
        description = (
            "Infer a plausible preceding state and converge continuously to "
            f"<Picture 1> as the final frame: {description}"
        )
    return H3BasePlan(
        mode=mode,
        duration_seconds=duration_seconds,
        shots=[H3BaseShot(start_seconds=0, description=description)],
        overall_soundscape="N/A",
        non_diegetic_music="N/A",
    )


SYSTEM_PROMPT = """You are H3-Context-IR-fake, a planning stage for MiniMax H3 Base.
Convert the user's free-form request into exactly one JSON object and nothing else.

Required schema:
{
  "mode": "t2va|i2va|fl2va|l2va",
  "duration_seconds": 4.0-15.0,
  "shots": [
    {"start_seconds": 0.0, "description": "detailed natural English"}
  ],
  "overall_soundscape": "1-4 English sentences or N/A",
  "non_diegetic_music": "1-3 English sentences or N/A"
}

Rules:
- Copy mode and duration_seconds exactly from the request.
- The first shot starts at 0. Later starts are strictly increasing and below duration.
- Each description establishes composition, subjects, environment, lighting, actions,
  camera movement, dialogue, and synchronized diegetic sound that actually matter.
- Do not put [Shot N], timestamps, or top-level field labels inside descriptions.
- Write later-shot descriptions so they read naturally after "the camera cuts to";
  begin with the new view or scene, not with "As", "Then", or another transition.
- Write planning prose in English, but preserve requested dialogue, lyrics, and visible
  text verbatim in their original language and punctuation.
- Never invent dialogue, lyrics, logos, or visible text not requested by the user.
- For requested dialogue or singing, assign stable speaker IDs such as (S1), keep the
  identifying phrase outside the tag, and format the exact words as
  <d>[Language] original words</d>. Put visible on-screen text in double quotes.
- For image modes, use <Picture 1> and <Picture 2> consistently and derive observable
  appearance/composition from the supplied images when the provider can see them.
- I2VA develops forward from Picture 1. FL2VA continuously connects Picture 1 to
  Picture 2, normally in one shot unless the user explicitly requests cuts. L2VA
  converges to Picture 1 as the final frame.
- overall_soundscape summarizes ambient, action, and non-verbal sound in 1-4 English
  sentences; use N/A only when the user explicitly asks for complete silence.
- non_diegetic_music describes instrumentation, speed, rhythm, and dynamics in 1-3
  English sentences, or N/A when no audience-only background score is intended.
- Keep the plan physically achievable within the requested short duration.
- Return valid JSON only. Do not use Markdown fences and do not explain your answer.
"""


def build_planning_request(
    prompt: str,
    *,
    mode: H3BaseMode,
    duration_seconds: float,
    image_count: int,
) -> str:
    references = (
        "No reference pictures are supplied."
        if image_count == 0
        else "Reference pictures supplied in order: "
        + ", ".join(f"<Picture {index}>" for index in range(1, image_count + 1))
        + "."
    )
    return (
        f"mode: {mode}\n"
        f"duration_seconds: {duration_seconds:.2f}\n"
        f"{references}\n"
        "user_request:\n"
        f"{(prompt or '').strip()}"
    )


def build_repair_request(error: str) -> str:
    return (
        "Your previous response failed validation. Return the entire corrected JSON "
        "object only, keeping the originally requested mode and duration. "
        f"Validation error: {error}"
    )


def is_structured_base_prompt(prompt: str) -> bool:
    value = (prompt or "").lower()
    return all(
        marker in value
        for marker in (
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        )
    )
