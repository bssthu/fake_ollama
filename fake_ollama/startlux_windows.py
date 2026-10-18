"""Optional Windows tensor-layout compatibility for official StartLux 1.1.0.

Importing this module does not import torch. Prompts, masks, read-out and model
self-checks remain in the official package. Only its bidirectional GQA inputs
are expanded to the equivalent repeated K/V heads for the Windows fused kernel.
"""
from __future__ import annotations

from functools import wraps
from importlib import metadata
import sys


def expand_bidirectional_kv(attention):
    @wraps(attention)
    def compatible(module, query, key, value, attention_mask, *args, **kwargs):
        # Leave the stock causal path and CPU untouched. The upstream function
        # still owns row boundaries, padding, scaling and is_causal=False.
        if kwargs.get("startlux_row_lengths") is not None and query.device.type == "cuda":
            heads, kv_heads = query.shape[1], key.shape[1]
            if heads != kv_heads and kv_heads > 0 and heads % kv_heads == 0 and value.shape[1] == kv_heads:
                groups = heads // kv_heads
                key = key.repeat_interleave(groups, dim=1)
                value = value.repeat_interleave(groups, dim=1)
        return attention(module, query, key, value, attention_mask, *args, **kwargs)

    compatible._startlux_windows_kv = True
    return compatible


def configure_windows_attention() -> bool:
    """Called only when starting the optional Transformers Instant backend."""
    if sys.platform != "win32" or metadata.version("startlux-decide") != "1.1.0":
        return False
    import torch

    if torch.backends.cuda.is_flash_attention_available():
        return False
    from startlux_decide.backends import bidirectional

    if not getattr(bidirectional._attention, "_startlux_windows_kv", False):
        bidirectional._attention = expand_bidirectional_kv(bidirectional._attention)
    return True
