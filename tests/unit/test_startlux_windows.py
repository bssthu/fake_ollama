"""Numerical CUDA acceptance for the optional Windows attention adapter.

Run in the StartLux environment with pytest and the official package installed.
The ordinary gateway test environment does not need torch or model weights.
"""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
pytest.importorskip("startlux_decide")

from startlux_decide.backends.bidirectional import _attention

# The companion environment deliberately does not install gateway dependencies.
spec = importlib.util.spec_from_file_location(
    "startlux_windows", Path(__file__).parents[2] / "fake_ollama" / "startlux_windows.py")
compatibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compatibility)
expand_bidirectional_kv = compatibility.expand_bidirectional_kv


@pytest.mark.parametrize("lengths", [[24, 24], [24, 11]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_windows_fused_attention_matches_grouped_math_with_padding(lengths, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(7)
    q = torch.randn(2, 16, 24, 256, device="cuda", dtype=dtype)
    k = torch.randn(2, 4, 24, 256, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    expected = torch.zeros(2, 24, 16, 256, device="cuda", dtype=dtype)
    with sdpa_kernel([SDPBackend.MATH]):
        for row, length in enumerate(lengths):
            out = torch.nn.functional.scaled_dot_product_attention(
                q[row:row + 1, :, :length], k[row:row + 1, :, :length], v[row:row + 1, :, :length],
                is_causal=False, enable_gqa=True, scale=0.07)
            expected[row, :length] = out[0].transpose(0, 1)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        actual, _ = expand_bidirectional_kv(_attention)(
            None, q, k, v, None, scaling=0.07, startlux_row_lengths=lengths)
    torch.testing.assert_close(actual, expected, atol=0.016 if dtype == torch.bfloat16 else 0.002, rtol=0.02)
    for row, length in enumerate(lengths):
        assert torch.count_nonzero(actual[row, length:]) == 0


def test_causal_calls_preserve_original_tensor_identity():
    seen = []
    q, k, v, mask = (object() for _ in range(4))

    def original(module, query, key, value, attention_mask, **kwargs):
        seen.append((query, key, value, attention_mask, kwargs))
        return "original-result"

    result = expand_bidirectional_kv(original)(None, q, k, v, mask, is_causal=True)
    assert result == "original-result"
    assert seen == [(q, k, v, mask, {"is_causal": True})]
