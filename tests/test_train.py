"""Training-loop failure paths: interrupted runs, no improvement, changed configs.

The corpus is replaced by a small synthetic one, so these tests need no download.
"""
import json

import pytest
import torch

import src.train as train_mod
from src.config import from_dict
from src.tokenizer import CharTokenizer

TEXT = "ROMEO:\nO, what a rogue and peasant slave am I!\nFirst Citizen:\nMy lord, I come.\n" * 100


@pytest.fixture(autouse=True)
def small_corpus(monkeypatch):
    tok = CharTokenizer.from_text(TEXT)
    data = torch.tensor(tok.encode(TEXT), dtype=torch.long)
    monkeypatch.setattr(train_mod, "prepare_data",
                        lambda cfg: (tok, data[:6000], data[6000:7000], data[7000:]))


def make_cfg(out_dir, **train):
    return from_dict({
        "model": {"context_length": 16, "d_model": 32, "n_heads": 2, "n_layers": 2, "dropout": 0.1},
        "train": {"batch_size": 4, "max_steps": 40, "warmup_steps": 5, "eval_interval": 10,
                  "sample_tokens": 5, "device": "cpu", "out_dir": str(out_dir), **train},
    })


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_interrupted_run_resumes_exactly(tmp_path):
    """Stopping at step 20 and resuming gives the same weights and optimizer state as one
    uninterrupted run (dropout on, so the restored RNG states matter)."""
    train_mod.train(make_cfg(tmp_path / "full"))
    assert train_mod.train(make_cfg(tmp_path / "split"), stop_after=20) is None
    train_mod.train(make_cfg(tmp_path / "split"), resume=True)

    a, b = load(tmp_path / "full" / "last.pt"), load(tmp_path / "split" / "last.pt")
    for name, tensor in a["model"].items():
        assert torch.equal(tensor, b["model"][name]), name
    for key, state in a["optimizer"]["state"].items():
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(state[field], b["optimizer"]["state"][key][field])
    assert a["extra"]["val_loss"] == b["extra"]["val_loss"]
    lines = (tmp_path / "split" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == [0, 10, 20, 30, 40]  # no duplicated log rows after the resume


def test_run_without_improvement_still_has_best_checkpoint(tmp_path):
    """With lr = 0 the model never changes, so only the step-0 checkpoint can be best."""
    summary = train_mod.train(make_cfg(tmp_path, lr=0.0, max_steps=20))
    assert summary["best_step"] == 0
    assert load(tmp_path / "best.pt")["extra"]["step"] == 0


def test_resume_rejects_a_changed_config(tmp_path):
    train_mod.train(make_cfg(tmp_path), stop_after=10)
    changed = make_cfg(tmp_path)
    changed = from_dict({**changed.to_dict(), "model": {**changed.to_dict()["model"], "dropout": 0.2}})
    with pytest.raises(ValueError, match="model.dropout"):
        train_mod.train(changed, resume=True)


def test_finished_run_is_reused_only_with_the_same_config(tmp_path):
    train_mod.train(make_cfg(tmp_path, max_steps=10))
    assert train_mod.finished_run(make_cfg(tmp_path, max_steps=10))
    assert not train_mod.finished_run(make_cfg(tmp_path / "elsewhere", max_steps=10))
    with pytest.raises(ValueError, match="train.max_steps"):
        train_mod.finished_run(make_cfg(tmp_path, max_steps=20))
