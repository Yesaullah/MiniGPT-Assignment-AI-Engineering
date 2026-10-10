"""Self-attention against the report's hand calculation (handout Section 6)."""
import math

import torch
import torch.nn.functional as F

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention

Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_hand_calculation_unmasked():
    out, w = scaled_dot_product_attention(Q, K, V)
    expected_scores = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 2.0, 1.0]]) / math.sqrt(2)
    assert torch.allclose(Q @ K.T / math.sqrt(2), expected_scores)
    assert torch.allclose(w, torch.tensor([[0.4011, 0.4011, 0.1978],
                                           [0.1978, 0.4011, 0.4011],
                                           [0.2483, 0.5035, 0.2483]]), atol=1e-4)
    assert torch.allclose(out, torch.tensor([[0.5989, 0.5989],
                                             [0.5989, 0.8022],
                                             [0.4965, 0.7517]]), atol=1e-4)


def test_hand_calculation_masked():
    out, w = scaled_dot_product_attention(Q, K, V, causal_mask(3))
    assert torch.allclose(w, torch.tensor([[1.0, 0.0, 0.0],
                                           [0.3302, 0.6698, 0.0],
                                           [0.2483, 0.5035, 0.2483]]), atol=1e-4)
    assert torch.allclose(out, torch.tensor([[1.0, 0.0], [0.3302, 0.6698], [0.4965, 0.7517]]),
                          atol=1e-4)


def test_matches_pytorch_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 7, 5) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(7))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_weights_are_distributions():
    torch.manual_seed(1)
    _, w = scaled_dot_product_attention(torch.randn(4, 6, 8), torch.randn(4, 6, 8),
                                        torch.randn(4, 6, 8), causal_mask(6))
    assert torch.allclose(w.sum(-1), torch.ones(4, 6), atol=1e-6)
    assert (w >= 0).all()


def test_multihead_split_merge_roundtrip():
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16)
    assert torch.equal(mha.merge_heads(mha.split_heads(x)), x)
