import pytest
import torch

from src.config import ModelConfig
from src.gpt import GPT


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=11, context_length=16, d_model=32, n_heads=4, n_layers=2,
                       ffn_dim=64, dropout=0.0)


@pytest.fixture
def tiny_model(tiny_cfg) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_cfg).eval()
