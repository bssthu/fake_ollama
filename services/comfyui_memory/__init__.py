"""ComfyUI stage boundary for workflows with large text encoders."""

import logging

import comfy.model_management


class ReleaseTextEncoderAfterConditioning:
    """Keep conditioning cached while releasing its encoder before sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "release"
    CATEGORY = "conditioning/memory"
    DESCRIPTION = (
        "Wait for both conditioning inputs, then unload their text encoder and "
        "its clones before sampling. Conditioning values and their cache are preserved."
    )

    def release(self, clip, positive, negative):
        comfy.model_management.unload_model_and_clones(
            clip.patcher, unload_additional_models=False, all_devices=True
        )
        logging.info("Released text encoder after positive and negative conditioning")
        return positive, negative


NODE_CLASS_MAPPINGS = {
    "ReleaseTextEncoderAfterConditioning": ReleaseTextEncoderAfterConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReleaseTextEncoderAfterConditioning": "Release Text Encoder After Conditioning",
}
