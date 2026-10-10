"""API smoke tests with a tiny random-weight checkpoint created on the fly."""
import importlib
import sys

import pytest

from src.config import Config, ModelConfig
from src.gpt import GPT
from src.tokenizer import CharTokenizer
from src.utils import load_checkpoint, save_checkpoint


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    tok = CharTokenizer.from_text("ROMEO:\n abcdefghij")
    cfg = Config(model=ModelConfig(vocab_size=tok.vocab_size, context_length=16, d_model=32,
                                   n_heads=2, n_layers=1, ffn_dim=64))
    path = tmp_path / "model.pt"
    save_checkpoint(str(path), GPT(cfg.model), cfg, tok)
    monkeypatch.setenv("MODEL_PATH", str(path))
    sys.modules.pop("src.serve.app", None)
    app = importlib.import_module("src.serve.app").app
    return TestClient(app)


def test_checkpoint_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("abc")
    cfg = Config(model=ModelConfig(vocab_size=3, context_length=8, d_model=16, n_heads=2,
                                   n_layers=1, ffn_dim=32))
    save_checkpoint(str(tmp_path / "m.pt"), GPT(cfg.model), cfg, tok)
    model, cfg2, tok2, _ = load_checkpoint(str(tmp_path / "m.pt"))
    assert cfg2.model == cfg.model and tok2.chars == tok.chars


def test_health_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model-info").json()["vocab_size"] > 0


def test_generate(client):
    r = client.post("/generate", json={"prompt": "ROMEO:\n", "max_new_tokens": 10, "seed": 1})
    assert r.status_code == 200
    assert len(r.json()["generated"]) == 10


def test_unknown_character_422(client):
    assert client.post("/generate", json={"prompt": "ROMEO 😀"}).status_code == 422


def test_unknown_field_422(client):
    assert client.post("/generate", json={"prompt": "a", "temprature": 1}).status_code == 422
