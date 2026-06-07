"""Tests for the declarative ComfyUI workflow presets and binding builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.comfyui_presets import (
    PRESETS,
    nearest_ratio,
    resolve_workflows,
)
from fake_ollama.config import ComfyUITarget


def _recording_client(workflow_config: Dict[str, Any]) -> tuple[ComfyUIClient, List[Dict[str, Any]]]:
    """A ComfyUIClient wired to a mock ComfyUI that records posted prompts."""
    posted: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.method == "POST" and path == "/upload/image":
            return httpx.Response(200, json={"name": "uploaded.png"})
        if request.method == "POST" and path == "/prompt":
            body = json.loads(request.read())
            posted.append(body["prompt"])
            return httpx.Response(200, json={"prompt_id": f"pid-{len(posted)}"})
        if request.method == "GET" and path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            return httpx.Response(
                200,
                json={
                    pid: {
                        "outputs": {"9": {"images": [
                            {"filename": "out.png", "subfolder": "", "type": "output"}
                        ]}},
                        "status": {"status_str": "success"},
                    }
                },
            )
        if request.method == "GET" and path == "/view":
            return httpx.Response(200, content=b"img")
        return httpx.Response(404)

    client = ComfyUIClient(
        "http://comfy.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        workflow_config={**workflow_config, "poll_interval_seconds": 0.01},
    )
    return client, posted


def _node_inputs(workflow: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    return workflow[node_id]["inputs"]


# ---------------------------------------------------------------------------
# nearest_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "w,h,expected",
    [
        (1024, 1024, "1:1"),
        (1920, 1080, "16:9"),
        (1080, 1920, "9:16"),
        (1024, 768, "4:3"),
        (768, 1024, "3:4"),
    ],
)
def test_nearest_ratio_buckets(w: int, h: int, expected: str) -> None:
    from fake_ollama.comfyui_presets import SENSENOVA_RATIOS

    assert nearest_ratio(w, h, SENSENOVA_RATIOS) == expected


# ---------------------------------------------------------------------------
# resolve_workflows wiring
# ---------------------------------------------------------------------------


def test_resolve_workflows_zimage_default_is_backward_compatible() -> None:
    target = ComfyUITarget(name="z", model="z-image-turbo")
    specs = resolve_workflows(target.workflow_config())
    t2i = specs["t2i"]
    assert t2i.binds("batch_size")
    # Legacy node-id wiring preserved.
    assert t2i.bindings["prompt"] == [("27", "text")]
    assert t2i.bindings["seed"] == [("3", "seed")]
    assert t2i.bindings["width"] == [("13", "width")]
    # Model files routed into the loader nodes via static inputs.
    assert t2i.static_inputs["28"]["unet_name"] == "z-image-turbo-fp8-e4m3fn.safetensors"
    assert t2i.static_inputs["30"]["clip_name"] == "qwen_3_4b_fp4_mixed.safetensors"
    # i2i exists and binds the uploaded image + does NOT batch (single-image edit).
    assert specs["i2i"].bindings["image"] == [("12", "image")]
    assert not specs["i2i"].binds("batch_size")


def test_resolve_workflows_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        ComfyUITarget(name="x", model="m", preset="does-not-exist")


def test_presets_have_expected_modes() -> None:
    assert PRESETS["qwen_image_edit_aio"].i2i is not None
    assert PRESETS["sensenova_u1"].i2i is not None


@pytest.mark.asyncio
async def test_joyai_echo_i2v_builtin_workflow_batches_reference_images() -> None:
    root = Path(__file__).resolve().parents[1]
    target = ComfyUITarget(
        name="joyai",
        model="joyai-echo",
        preset="custom",
        image_to_video_workflow_path=str(
            root / "fake_ollama" / "workflows" / "joyai_echo_i2v.json"
        ),
        output_prefix="fake_ollama/joyai-echo",
        save_video_node_id="9",
        max_reference_images=5,
        bindings={
            "i2v": {
                "prompt": [["3", "prompt"]],
                "images": [["19", "image"]],
                "width": [["19", "width"]],
                "height": [["19", "height"]],
                "seed": [["19", "seed"]],
                "num_frames": [["19", "num_frames"]],
                "frame_rate": [["19", "frame_rate"], ["10", "fps"]],
                "prefetch_count": [["3", "prefetch_count"], ["19", "prefetch_count"]],
            }
        },
    )
    client = ComfyUIClient("http://comfy.test", workflow_config=target.workflow_config())
    try:
        spec = client._workflows["i2v"]
        assert spec is not None
        workflow = client._build_workflow(
            spec,
            {
                "prompt": "animate the reference image",
                "images": ["uploaded-1.png", "uploaded-2.png"],
                "width": 512,
                "height": 384,
                "seed": 42,
                "num_frames": 33,
                "frame_rate": 12.0,
                "prefetch_count": 1,
            },
        )
    finally:
        await client.aclose()

    assert _node_inputs(workflow, "12")["image"] == "uploaded-1.png"
    assert _node_inputs(workflow, "20")["image"] == "uploaded-2.png"
    assert workflow["21"]["class_type"] == "ImageBatch"
    assert _node_inputs(workflow, "21")["image1"] == ["12", 0]
    assert _node_inputs(workflow, "21")["image2"] == ["20", 0]
    assert _node_inputs(workflow, "19")["image"] == ["21", 0]
    assert _node_inputs(workflow, "3")["prompt"] == "animate the reference image"
    assert _node_inputs(workflow, "19")["width"] == 512
    assert _node_inputs(workflow, "19")["height"] == 384
    assert _node_inputs(workflow, "19")["seed"] == 42
    assert _node_inputs(workflow, "19")["num_frames"] == 33
    assert _node_inputs(workflow, "10")["fps"] == 12.0
    assert _node_inputs(workflow, "9")["filename_prefix"] == "fake_ollama/joyai-echo"


# ---------------------------------------------------------------------------
# Full build chain per preset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qwen_t2i_build_fills_ksampler_and_latent() -> None:
    target = ComfyUITarget(
        name="qwen",
        model="qwen-image",
        preset="qwen_image_edit_aio",
        default_sampler_name="euler_ancestral",
        default_scheduler="beta",
        output_prefix="fake_ollama/qwen-image",
    )
    client, posted = _recording_client(target.workflow_config())
    try:
        await client.generate_image(
            model="qwen-image", prompt="a fox", width=768, height=512, n=3,
            seed=11, steps=6, cfg=1.0, sampler_name="euler_ancestral",
            scheduler="beta", denoise=1.0,
        )
    finally:
        await client.aclose()

    # batch_size bound -> one submission for n=3.
    assert len(posted) == 1
    wf = posted[0]
    assert _node_inputs(wf, "27")["text"] == "a fox"
    assert _node_inputs(wf, "3")["steps"] == 6
    assert _node_inputs(wf, "3")["sampler_name"] == "euler_ancestral"
    assert _node_inputs(wf, "3")["scheduler"] == "beta"
    assert _node_inputs(wf, "13")["width"] == 768
    assert _node_inputs(wf, "13")["height"] == 512
    assert _node_inputs(wf, "13")["batch_size"] == 3
    assert _node_inputs(wf, "9")["filename_prefix"] == "fake_ollama/qwen-image"


@pytest.mark.asyncio
async def test_qwen_i2i_uses_edit_node_and_uploaded_image() -> None:
    target = ComfyUITarget(name="qwen", model="qwen-image", preset="qwen_image_edit_aio")
    client, posted = _recording_client(target.workflow_config())
    try:
        await client.edit_image(
            model="qwen-image", prompt="add a hat", image_bytes=b"png",
            filename="in.png", width=1024, height=1024, n=1, seed=5, steps=6,
            cfg=1.0, sampler_name="euler_ancestral", scheduler="beta", denoise=1.0,
        )
    finally:
        await client.aclose()

    wf = posted[0]
    # TextEncodeQwenImageEditPlus uses "prompt" (not "text") and an image1 link.
    assert _node_inputs(wf, "27")["prompt"] == "add a hat"
    assert _node_inputs(wf, "12")["image"] == "uploaded.png"
    assert _node_inputs(wf, "3")["denoise"] == 1.0


@pytest.mark.asyncio
async def test_sensenova_t2i_maps_size_to_ratio_and_batches() -> None:
    target = ComfyUITarget(name="sn", model="sensenova", preset="sensenova_u1")
    client, posted = _recording_client(target.workflow_config())
    try:
        await client.generate_image(
            model="sensenova", prompt="a cat", width=1920, height=1080, n=2,
            seed=9, steps=8, cfg=1.0, sampler_name="x", scheduler="y", denoise=1.0,
        )
    finally:
        await client.aclose()

    assert len(posted) == 1  # batch_size bound
    wf = posted[0]
    sampler = _node_inputs(wf, "10")
    assert sampler["prompt"] == "a cat"
    assert sampler["steps"] == 8
    assert sampler["batch_size"] == 2
    assert sampler["target_pixels"] == "16:9"
    # Fixed widgets stay at the baked values.
    assert sampler["img_mode"] == "edit"
    assert sampler["timestep_shift"] == 3.0
    # Layer streaming (most layers in RAM) so the 8B model + 2048² activations
    # fit in 24 GB without OOM.
    assert sampler["prefetch_count"] == 2
    # t2i graph carries no image input.
    assert "image" not in sampler


@pytest.mark.asyncio
async def test_sensenova_i2i_wires_loadimage() -> None:
    target = ComfyUITarget(name="sn", model="sensenova", preset="sensenova_u1")
    client, posted = _recording_client(target.workflow_config())
    try:
        await client.edit_image(
            model="sensenova", prompt="make it night", image_bytes=b"png",
            filename="in.png", width=1024, height=1024, n=1, seed=1, steps=8,
            cfg=1.0, sampler_name="x", scheduler="y", denoise=1.0,
        )
    finally:
        await client.aclose()

    wf = posted[0]
    assert _node_inputs(wf, "12")["image"] == "uploaded.png"
    assert _node_inputs(wf, "10")["image"] == ["12", 0]
    assert _node_inputs(wf, "10")["target_pixels"] == "1:1"
