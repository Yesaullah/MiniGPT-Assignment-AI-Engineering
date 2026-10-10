"""Tensor shapes through every stage, and the parameter-count formula."""
import torch

from src.attention import MultiHeadAttention
from src.config import ModelConfig
from src.gpt import GPT
from src.positional_encoding import sinusoidal_encoding


def test_multihead_shape_trace():
    mha = MultiHeadAttention(d_model=256, n_heads=4)
    mha.trace = []
    out = mha(torch.randn(2, 10, 256))
    shapes = dict(mha.trace)
    assert shapes["Q = x W_Q"] == (2, 10, 256)
    assert shapes["Q split into heads"] == (2, 4, 10, 64)
    assert shapes["attention weights per head"] == (2, 4, 10, 10)
    assert shapes["per-head output"] == (2, 4, 10, 64)
    assert shapes["heads concatenated"] == (2, 10, 256)
    assert out.shape == (2, 10, 256)


def test_model_output_shape(tiny_model):
    logits, loss = tiny_model(torch.randint(0, 11, (3, 16)), torch.randint(0, 11, (3, 16)))
    assert logits.shape == (3, 16, 11)
    assert loss.ndim == 0


def test_baseline_parameter_count():
    cfg = ModelConfig(vocab_size=65, context_length=128, d_model=256, n_heads=4, n_layers=4,
                      ffn_dim=1024)
    assert GPT(cfg).num_params() == 3_192_832


def test_positional_encoding_values():
    pe = sinusoidal_encoding(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8))  # sin(0)
    assert torch.allclose(pe[0, 1::2], torch.ones(8))  # cos(0)
    assert torch.allclose(pe[3, 0], torch.sin(torch.tensor(3.0)))


def test_initial_loss_near_expected(tiny_cfg):
    torch.manual_seed(0)
    model = GPT(tiny_cfg).eval()
    _, loss = model(torch.randint(0, 11, (8, 16)), torch.randint(0, 11, (8, 16)))
    assert abs(loss.item() - torch.log(torch.tensor(11.0)).item()) < 0.3
