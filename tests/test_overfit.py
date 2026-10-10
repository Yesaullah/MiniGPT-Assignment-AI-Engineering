"""Tiny overfit gate: loss on one repeated batch falls substantially (handout Section 16)."""
import torch

from src.config import ModelConfig
from src.gpt import GPT


def test_overfits_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=20, context_length=16, d_model=64, n_heads=4, n_layers=2,
                      ffn_dim=128, dropout=0.0)
    model = GPT(cfg)
    stream = torch.randint(0, 20, (4, 17))
    x, y = stream[:, :-1], stream[:, 1:]  # shifted targets
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    _, first = model(x, y)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first.item(), (first.item(), loss.item())
