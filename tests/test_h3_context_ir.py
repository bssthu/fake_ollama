"""Unit tests for the deterministic H3 Context-IR-fake harness."""

from __future__ import annotations

import json

import pytest

from fake_ollama.config import H3ContextIRProfile, Settings
from fake_ollama.h3_context_ir import (
    fallback_plan,
    is_structured_base_prompt,
    parse_and_validate_plan,
    render_base_prompt,
    resolve_base_mode,
)


def _plan_json(*, mode: str = "t2va", duration: float = 5.0) -> str:
    return json.dumps(
        {
            "mode": mode,
            "duration_seconds": duration,
            "shots": [
                {
                    "start_seconds": 0,
                    "description": "Cinematic medium shot of a baker opening a shop.",
                },
                {
                    "start_seconds": 2.5,
                    "description": "a close-up of warm bread releasing steam.",
                },
            ],
            "overall_soundscape": "Wooden shutters scrape and trays clink softly.",
            "non_diegetic_music": "A quiet acoustic-guitar pattern at a slow tempo.",
        }
    )


def test_resolve_base_mode_from_reference_count() -> None:
    assert resolve_base_mode("auto", 0) == "t2va"
    assert resolve_base_mode("auto", 1) == "i2va"
    assert resolve_base_mode("auto", 2) == "fl2va"
    assert resolve_base_mode("l2va", 1) == "l2va"
    with pytest.raises(ValueError, match="at most two"):
        resolve_base_mode("auto", 3)
    with pytest.raises(ValueError, match="requires 1 image"):
        resolve_base_mode("i2va", 0)


def test_parse_validate_and_render_documented_base_shape() -> None:
    plan = parse_and_validate_plan(
        f"```json\n{_plan_json()}\n```",
        expected_mode="t2va",
        expected_duration_seconds=5.0,
    )
    prompt = render_base_prompt(plan)
    assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
    assert "[Shot 2] At 00:02.500, the camera cuts to" in prompt
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt
    assert is_structured_base_prompt(prompt) is True


def test_request_owned_mode_duration_and_timeline_are_enforced() -> None:
    with pytest.raises(ValueError, match="changed mode"):
        parse_and_validate_plan(
            _plan_json(mode="i2va"),
            expected_mode="t2va",
            expected_duration_seconds=5.0,
        )
    with pytest.raises(ValueError, match="changed duration"):
        parse_and_validate_plan(
            _plan_json(duration=6.0),
            expected_mode="t2va",
            expected_duration_seconds=5.0,
        )


def test_fallback_preserves_user_text_and_keyframe_alignment() -> None:
    plan = fallback_plan(
        "一个人抬头看向雨夜天空。",
        mode="i2va",
        duration_seconds=5.0,
    )
    prompt = render_base_prompt(plan)
    assert "<Picture 1> (from [Shot 1]) is fully referenced" in prompt
    assert "一个人抬头看向雨夜天空。" in prompt
    assert "overall_soundscape: N/A" in prompt


def test_profile_lookup_ids_are_unambiguous() -> None:
    provider = {
        "name": "planner",
        "model": "planner-model",
        "target": "planner-target",
    }
    with pytest.raises(ValueError, match="alias must not contain"):
        H3ContextIRProfile(
            name="default",
            alias="ambiguous@alias",
            providers=[provider],
        )
    with pytest.raises(ValueError, match="duplicate H3 Context-IR lookup id"):
        Settings(
            openai_upstreams=[
                {
                    "name": "planner-target",
                    "base_url": "https://planner.test",
                    "models": [{"name": "planner-model"}],
                }
            ],
            h3_context_ir_profiles=[
                {"name": "one", "alias": "two", "providers": [provider]},
                {"name": "two", "alias": "three", "providers": [provider]},
            ],
        )
