"""Scaled dot-product attention and multi-head attention, written out explicitly.

Shapes: B = batch, T = sequence length, d = d_model, h = heads, d_h = d / h.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Boolean (T, T) mask, True = allowed. Position i may attend to positions j <= i."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor | None = None,
                                 dropout: nn.Module | None = None):
    """softmax(Q K^T / sqrt(d_k)) V.

    q, k: (..., T, d_k); v: (..., T, d_v); mask: broadcastable (..., T, T), True = allowed.
    Returns (output (..., T, d_v), weights (..., T, T)). Weights are returned before dropout.
    Scores and softmax are computed in float32.
    """
    d_k = q.size(-1)
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    applied = dropout(weights) if dropout is not None else weights
    out = applied @ v.float()
    return out.to(v.dtype), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.causal = causal
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.last_weights: torch.Tensor | None = None
        self.trace: list[tuple[str, tuple]] | None = None  # set to [] to record shapes

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, h, T, d_h)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, T, d_h) -> (B, T, d): concatenate the heads."""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def _record(self, name: str, t: torch.Tensor) -> None:
        if self.trace is not None:
            self.trace.append((name, tuple(t.shape)))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        self._record("input x", x)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        self._record("Q = x W_Q", q)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        self._record("Q split into heads", q)
        mask = causal_mask(T, x.device) if self.causal else None
        if key_padding_mask is not None:  # (B, T), True = real token
            kp = key_padding_mask[:, None, None, :]
            mask = kp if mask is None else (mask & kp)
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        self.last_weights = weights.detach()
        self._record("attention weights per head", weights)
        self._record("per-head output", out)
        out = self.merge_heads(out)
        self._record("heads concatenated", out)
        out = self.resid_dropout(self.w_o(out))
        self._record("output projection W_O", out)
        return out
