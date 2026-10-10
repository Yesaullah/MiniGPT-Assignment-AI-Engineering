"""MiniGPT: token embedding + sinusoidal positions -> N Pre-LN blocks -> LayerNorm -> LM head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.positional_encoding import SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be set from the tokenizer")
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.context_length, d)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(d, cfg.n_heads, cfg.ffn_dim, cfg.dropout, cfg.causal)
            for _ in range(cfg.n_layers)
        )
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)  # untied from tok_emb
        self._init_weights()

    def _init_weights(self) -> None:
        d, L = self.cfg.d_model, self.cfg.n_layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Token embeddings: std 1/sqrt(d), scaled by sqrt(d) in forward -> unit variance.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(d))
        # GPT-2: residual output projections scaled by 1/sqrt(2L).
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(block.ffn.linear_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * L))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model) + self.pos_enc(T)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x, key_padding_mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    def optimizer_groups(self, weight_decay: float) -> list[dict]:
        """Weight decay on Linear weight matrices only (not embeddings, biases or LayerNorm)."""
        decay, no_decay = [], []
        for module in self.modules():
            for name, p in module.named_parameters(recurse=False):
                if isinstance(module, nn.Linear) and name == "weight":
                    decay.append(p)
                else:
                    no_decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """Autoregressive sampling. Each step recomputes the last context_length tokens."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.context_length :]
            logits, _ = self(window)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx
