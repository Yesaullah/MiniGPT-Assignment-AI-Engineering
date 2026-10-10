"""Tokenizer, split and target-shift checks."""
import pytest
import torch

from src.dataset import eval_windows, get_batch, split_text
from src.tokenizer import CharTokenizer, UnknownCharacterError


def test_tokenizer_roundtrip_and_determinism():
    tok = CharTokenizer.from_text("hello world\n")
    assert tok.chars == sorted(set("hello world\n"))
    assert tok.decode(tok.encode("hello")) == "hello"
    assert CharTokenizer.from_dict(tok.to_dict()).chars == tok.chars


def test_unknown_character_is_rejected():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(UnknownCharacterError) as e:
        tok.encode("abz")
    assert e.value.chars == ["z"]


def test_split_sizes_match_hf_loader():
    train, val, test = split_text("x" * 1_115_394)
    assert (len(train), len(val), len(test)) == (1_003_854, 55_770, 55_770)


def test_batch_targets_are_shifted_by_one():
    stream = torch.arange(1000)
    x, y = get_batch(stream, 8, 16, torch.Generator().manual_seed(0))
    for row_x, row_y in zip(x, y):
        start = int(row_x[0])
        assert torch.equal(row_x, stream[start : start + 16])
        assert torch.equal(row_y, stream[start + 1 : start + 17])  # y[t] == stream[start+t+1]


def test_eval_windows_cover_every_target_once():
    stream = torch.arange(100)
    x, y = eval_windows(stream, 16)
    targets = y[y != -100]
    assert sorted(targets.tolist()) == list(range(1, 100))
