"""Unit tests for workflow-backed media operation discovery schemas."""

from fake_ollama.config import ComfyUITarget
from fake_ollama.media_operations import describe_comfyui_operation


def _names(operation: dict) -> list[str]:
    return [parameter["name"] for parameter in operation["parameters"]]


def test_z_image_only_advertises_parameters_bound_by_each_workflow() -> None:
    target = ComfyUITarget(name="z", model="z-image", max_batch_size=2)

    generation = describe_comfyui_operation(target, "image_generation")
    edit = describe_comfyui_operation(target, "image_edit")

    assert _names(generation) == [
        "size",
        "n",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
    ]
    assert "denoise" not in _names(generation)
    assert _names(edit)[-1] == "denoise"
    assert edit["limits"]["max_reference_images"] == 1
    assert edit["multiple_images"] is False


def test_qwen_defaults_and_edit_contract_match_builtin_workflow() -> None:
    target = ComfyUITarget(
        name="qwen",
        model="qwen-image",
        preset="qwen_image_edit_aio",
        default_steps=6,
        default_cfg=1.0,
        default_sampler_name="euler_ancestral",
        default_scheduler="beta",
        default_edit_denoise=1.0,
    )

    edit = describe_comfyui_operation(target, "image_edit")

    assert edit["configured"] is True
    assert edit["workflow_modes"] == ["i2i"]
    assert edit["defaults"]["steps"] == 6
    assert edit["defaults"]["sampler_name"] == "euler_ancestral"
    assert edit["defaults"]["scheduler"] == "beta"
    assert edit["defaults"]["denoise"] == 1.0
    assert edit["limits"]["max_reference_images"] == 1


def test_sensenova_exposes_ratio_bucket_but_not_ignored_sampler_fields() -> None:
    target = ComfyUITarget(
        name="sense", model="sensenova", preset="sensenova_u1"
    )

    generation = describe_comfyui_operation(target, "image_generation")
    size = generation["parameters"][0]

    assert _names(generation) == ["size", "n", "seed", "steps", "cfg"]
    assert size["type"] == "select"
    assert {choice["label"] for choice in size["choices"]} >= {
        "1:1",
        "16:9",
        "9:16",
    }
    assert "sampler_name" not in generation["defaults"]
    assert "scheduler" not in generation["defaults"]


def test_joyai_safe_defaults_presets_and_nonstandard_controls() -> None:
    target = ComfyUITarget(
        name="joy",
        model="joyai-echo",
        preset="joyai_echo",
        min_width=256,
        default_width=256,
        width_modulo=32,
        min_height=256,
        default_height=256,
        height_modulo=32,
        min_num_frames=17,
        default_num_frames=17,
        max_num_frames=241,
        num_frames_offset=1,
        num_frames_modulo=8,
        min_frame_rate=8,
        default_frame_rate=8,
        default_prefetch_count=1,
        default_enable_tile=True,
        default_enable_streaming=False,
        max_reference_images=5,
        max_batch_size=1,
    )

    operation = describe_comfyui_operation(target, "video_generation")

    assert operation["workflow_modes"] == ["video", "i2v"]
    assert operation["accepts_images"] is True
    assert operation["requires_images"] is False
    assert operation["multiple_images"] is True
    assert operation["limits"]["max_reference_images"] == 5
    assert operation["limits"]["num_frames_offset"] == 1
    assert operation["limits"]["num_frames_modulo"] == 8
    assert _names(operation) == [
        "size",
        "seed",
        "num_frames",
        "fps",
        "prefetch_count",
        "enable_tile",
        "enable_streaming",
    ]
    assert "steps" not in _names(operation)
    assert "cfg" not in _names(operation)
    assert operation["defaults"]["enable_tile"] is True
    assert operation["parameters"][0]["min_width"] == 256
    assert operation["parameters"][0]["width_modulo"] == 32
    frames = next(
        parameter
        for parameter in operation["parameters"]
        if parameter["name"] == "num_frames"
    )
    assert frames["offset"] == 1
    assert frames["modulo"] == 8
    assert operation["recommended_preset"] == "safe_debug"
    assert operation["default_preset"] == "safe_debug"


def test_recommended_preset_does_not_override_an_operators_custom_defaults() -> None:
    target = ComfyUITarget(
        name="joy",
        model="joyai-echo",
        preset="joyai_echo",
        default_width=512,
        default_height=512,
        min_num_frames=17,
        default_num_frames=33,
        max_num_frames=241,
        num_frames_offset=1,
        num_frames_modulo=8,
        min_frame_rate=8,
        default_frame_rate=12,
        default_enable_tile=True,
    )

    operation = describe_comfyui_operation(target, "video_generation")

    assert operation["recommended_preset"] == "safe_debug"
    assert operation["default_preset"] is None
    assert operation["defaults"]["size"] == "512x512"
    assert operation["defaults"]["num_frames"] == 33


def test_minimax_h3_exposes_all_base_modes_and_768p_defaults() -> None:
    target = ComfyUITarget(
        name="h3",
        model="minimax-h3-768p",
        preset="minimax_h3",
        min_width=32,
        default_width=1344,
        max_width=1344,
        width_modulo=32,
        min_height=32,
        default_height=768,
        max_height=1344,
        height_modulo=32,
        default_steps=20,
        default_sampler_name="res_multistep",
        default_scheduler="simple",
        min_num_frames=5,
        default_num_frames=124,
        max_num_frames=362,
        num_frames_offset=5,
        num_frames_modulo=17,
        min_frame_rate=24,
        default_frame_rate=24,
        max_frame_rate=24,
        max_reference_images=2,
        max_batch_size=1,
    )

    operation = describe_comfyui_operation(target, "video_generation")

    assert operation["workflow_modes"] == ["video", "i2v", "fl2va", "l2va"]
    assert operation["accepts_images"] is True
    assert operation["requires_images"] is False
    assert operation["multiple_images"] is True
    assert operation["limits"]["max_reference_images"] == 2
    assert operation["limits"]["num_frames_offset"] == 5
    assert operation["limits"]["num_frames_modulo"] == 17
    assert operation["defaults"]["size"] == "1344x768"
    assert operation["defaults"]["num_frames"] == 124
    assert operation["defaults"]["fps"] == 24
    assert operation["recommended_preset"] == "h3_768p_5s"
    assert operation["default_preset"] == "h3_768p_5s"
