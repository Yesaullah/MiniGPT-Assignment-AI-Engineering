"""Pre-LN Transformer decoder block: x + Attn(LN(x)), then x + FFN(LN(x))."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Linear(d, f) -> GELU -> Linear(f, d) -> dropout."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear_in = nn.Linear(d_model, ffn_dim)
        self.act = nn.GELU()
        self.linear_out = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear_out(self.act(self.linear_in(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0,
                 causal: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), key_padding_mask)  # residual around attention
        x = x + self.ffn(self.ln_2(x))  # residual around the feed-forward network
        return x
