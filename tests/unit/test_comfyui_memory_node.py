"""The extension must not move or mutate cached conditioning tensors."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


def test_release_preserves_conditioning_and_targets_only_its_encoder(monkeypatch):
    management = ModuleType("comfy.model_management")
    management.unload_model_and_clones = Mock()
    comfy = ModuleType("comfy")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    path = Path(__file__).resolve().parents[2] / "services/comfyui_memory/__init__.py"
    spec = importlib.util.spec_from_file_location("memory_node_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clip = SimpleNamespace(patcher=object())
    positive, negative = [[object(), {"attention_mask": object()}]], [[object(), {}]]

    result = module.ReleaseTextEncoderAfterConditioning().release(clip, positive, negative)

    assert result[0] is positive
    assert result[1] is negative
    management.unload_model_and_clones.assert_called_once_with(
        clip.patcher, unload_additional_models=False, all_devices=True
    )
