"""MiniMax H3 preset selection and API-prompt binding tests."""

from __future__ import annotations

import json

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.comfyui_presets import resolve_workflows


def _config() -> dict:
    return {
        "preset": "minimax_h3",
        "output_prefix": "test/h3",
        "save_video_node_id": "9",
        "max_reference_images": 2,
    }


def test_h3_preset_ships_four_api_workflows_with_fixed_quant_files() -> None:
    workflows = resolve_workflows(_config())

    assert [mode for mode in ("video", "i2v", "fl2va", "l2va") if workflows[mode]] == [
        "video",
        "i2v",
        "fl2va",
        "l2va",
    ]
    assert workflows["video"].max_image_refs == 0
    assert workflows["i2v"].max_image_refs == 1
    assert workflows["fl2va"].max_image_refs == 2
    assert workflows["l2va"].max_image_refs == 1

    for mode in ("video", "i2v", "fl2va", "l2va"):
        spec = workflows[mode]
        prompt = json.loads(spec.path.read_text(encoding="utf-8"))
        assert prompt["1"]["inputs"]["unet_name"] == (
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        )
        assert prompt["2"]["inputs"]["clip_name"] == (
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        )
        assert spec.static_inputs["9"]["filename_prefix"] == "test/h3"


def test_h3_mode_selection_distinguishes_first_and_last_frame() -> None:
    client = ComfyUIClient("http://comfy.test", workflow_config=_config())
    try:
        assert client._select_video_workflow_mode("auto", 0) == "video"
        assert client._select_video_workflow_mode("auto", 1) == "i2v"
        assert client._select_video_workflow_mode("auto", 2) == "fl2va"
        assert client._select_video_workflow_mode("i2va", 1) == "i2v"
        assert client._select_video_workflow_mode("l2va", 1) == "l2va"
    finally:
        import asyncio

        asyncio.run(client.aclose())


def test_h3_fl2va_and_l2va_bind_the_correct_keyframe_slots() -> None:
    workflows = resolve_workflows(_config())
    client = ComfyUIClient("http://comfy.test", workflow_config=_config())
    try:
        common = {
            "prompt": "move continuously",
            "width": 1344,
            "height": 768,
            "num_frames": 124,
            "seed": 7,
            "steps": 20,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "frame_rate": 24,
        }
        fl2va = client._build_workflow(
            workflows["fl2va"],
            {**common, "image_1": "first.png", "image_2": "last.png"},
        )
        l2va = client._build_workflow(
            workflows["l2va"], {**common, "image": "last.png"}
        )
    finally:
        import asyncio

        asyncio.run(client.aclose())

    assert fl2va["20"]["inputs"]["image"] == "first.png"
    assert fl2va["21"]["inputs"]["image"] == "last.png"
    assert fl2va["5"]["inputs"]["first_frame"] == ["20", 0]
    assert fl2va["5"]["inputs"]["last_frame"] == ["21", 0]
    assert l2va["20"]["inputs"]["image"] == "last.png"
    assert "first_frame" not in l2va["5"]["inputs"]
    assert l2va["5"]["inputs"]["last_frame"] == ["20", 0]
