"""Sinusoidal positional encoding (Vaswani et al., 2017), interleaved sin/cos.

PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_encoding(max_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: d_model // 2])
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (not learned) encoding. Positions are local to each input window: 0..T-1."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model), persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.pe.size(0):
            raise ValueError(f"Sequence length {T} exceeds context length {self.pe.size(0)}")
        return self.pe[:T]
