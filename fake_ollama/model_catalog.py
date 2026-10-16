"""Protocol-specific public model metadata."""

from __future__ import annotations

from typing import Any, Dict
from .config import Settings
from .routing import (
    _profile_for_public_id,
)

def _openai_model_entry(public_id: str, created: int) -> Dict[str, Any]:
    """Return the exact OpenAI Model object shape."""

    return {
        "id": public_id,
        "object": "model",
        "created": created,
        "owned_by": "fake-ollama",
    }


def _anthropic_model_capabilities(profile: Any) -> Dict[str, Any]:
    """Map the local profile conservatively to Anthropic's capability schema."""

    capability_set = set(profile.capabilities)

    def support(enabled: bool = False) -> Dict[str, bool]:
        return {"supported": enabled}

    thinking_enabled = profile.thinking_mode == "enabled"
    return {
        "batch": support(),
        "citations": support(),
        "code_execution": support(),
        "context_management": {
            "clear_thinking_20251015": support(),
            "clear_tool_uses_20250919": support(),
            "compact_20260112": support(),
            "supported": False,
        },
        "effort": {
            "high": support(),
            "low": support(),
            "max": support(),
            "medium": support(),
            "supported": False,
            "xhigh": support(),
        },
        "image_input": support("vision" in capability_set),
        "pdf_input": support(),
        "structured_outputs": support(),
        "thinking": {
            "supported": thinking_enabled,
            "types": {
                "adaptive": support(),
                "enabled": support(thinking_enabled),
            },
        },
    }


def _anthropic_model_entry(
    settings: Settings, iface: Any, public_id: str
) -> Dict[str, Any]:
    """Return the Anthropic ModelInfo shape for one exposed model."""

    profile = _profile_for_public_id(settings, iface, public_id)
    return {
        "id": public_id,
        "capabilities": _anthropic_model_capabilities(profile),
        "created_at": "1970-01-01T00:00:00Z",
        "display_name": public_id,
        "max_input_tokens": profile.context_length,
        "max_tokens": profile.max_output_tokens or profile.context_length,
        "type": "model",
    }
