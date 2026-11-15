import pytest

from fake_ollama.comfyui_client import ComfyUIClient
from fake_ollama.comfyui_presets import resolve_workflows
from fake_ollama.config import ComfyUITarget
from fake_ollama.media_operations import describe_comfyui_operation


@pytest.mark.asyncio
async def test_h3_image_reference_order_and_fixed_schedule():
    target = ComfyUITarget(name="h3-image", preset="minimax_h3_image",
                          default_width=2048, default_height=2048, max_width=4096, max_height=4096)
    client = ComfyUIClient("http://unused.test", workflow_config=target.model_dump())
    try:
        specs = resolve_workflows(target.model_dump())
        params = {"prompt": "Edit <Picture 1> using <Picture 2>.", "seed": 42,
                  "width": 4096, "height": 4096, "steps": 99, "cfg": 7,
                  "reference_images": [f"ref-{i}.png" for i in range(1, 17)]}
        graph = client._build_workflow(specs["i2i"], params)
        for i in range(1, 17):
            link = graph["5"]["inputs"][f"images.image_{i}"]
            assert graph[link[0]]["inputs"]["image"] == f"ref-{i}.png"
        assert not any(n["class_type"] == "ImageBatch" for n in graph.values())
        assert graph["6"]["inputs"]["steps"] == 6
        assert graph["6"]["inputs"]["cfg"] == 1
        assert graph["5"]["inputs"]["width"] == 4096
        assert not any(k.startswith("images.") for k in client._build_workflow(specs["t2i"], params)["5"]["inputs"])
        edit = describe_comfyui_operation(target, "image_edit")
        assert edit["limits"]["max_reference_images"] == 16
        assert {p["values"]["size"] for p in edit["presets"]} == {"2048x2048", "4096x4096"}
        assert not {"steps", "cfg", "negative_prompt"} & {p["name"] for p in edit["parameters"]}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_bf16_override_keeps_image_vae_and_quantized_default():
    fields = {"preset": "minimax_h3_image", "static_inputs": {
        mode: {"1": {"unet_name": "minimax_h3_image_pruned_bf16.safetensors"},
               "2": {"clip_name": "qwen3vl_32b_minimax_h3_bf16.safetensors"}}
        for mode in ("t2i", "i2i")}}
    client = ComfyUIClient("http://unused.test", workflow_config=fields)
    try:
        for spec in resolve_workflows(fields).values():
            if spec is not None:
                graph = client._build_workflow(spec, {})
                assert graph["1"]["inputs"]["unet_name"].endswith("_bf16.safetensors")
                assert graph["2"]["inputs"]["clip_name"].endswith("_bf16.safetensors")
                assert graph["3"]["inputs"]["vae_name"] == "minimax_h3_image_vae_fp16.safetensors"
        default = client._build_workflow(resolve_workflows({"preset": "minimax_h3_image"})["t2i"], {})
        assert default["1"]["inputs"]["unet_name"].endswith("_w4a8.safetensors")
    finally:
        await client.aclose()
