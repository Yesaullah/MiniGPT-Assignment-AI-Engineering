"""Sliding-window (strided) evaluation: coverage, context, and agreement with a brute force."""
import pytest
import torch

from src.train import evaluate_split, evaluate_split_strided, strided_windows


@pytest.mark.parametrize("n,T,stride", [(100, 16, 4), (100, 16, 16), (97, 16, 5), (10, 16, 4), (17, 16, 3)])
def test_every_target_scored_once(n, T, stride):
    scored = []
    for start, n_new in strided_windows(n, T, stride):
        length = min(T, n - 1)
        targets = list(range(start + 1, start + 1 + length))
        scored += targets[length - n_new :]
    assert scored == list(range(1, n))


def test_context_after_first_window():
    T, stride = 16, 4
    for start, n_new in strided_windows(200, T, stride)[1:]:
        assert n_new <= stride  # scored targets see at least T - stride + 1 tokens


def test_matches_brute_force(tiny_model):
    """Each target's loss equals scoring it alone from the window that strided evaluation uses."""
    torch.manual_seed(0)
    data = torch.randint(0, 11, (60,))
    T, stride = 16, 3
    total, count = 0.0, 0
    with torch.no_grad():
        for start, n_new in strided_windows(len(data), T, stride):
            length = min(T, len(data) - 1)
            logits, _ = tiny_model(data[start : start + length][None])
            for j in range(length - n_new, length):
                total += torch.nn.functional.cross_entropy(logits[0, j], data[start + j + 1]).item()
                count += 1
    expected = total / count
    assert evaluate_split_strided(tiny_model, data, T, "cpu", stride, batch_size=4) == pytest.approx(expected)


def test_stride_equal_to_context_matches_non_overlapping(tiny_model):
    """When the split is a whole number of windows, stride T is the non-overlapping protocol."""
    torch.manual_seed(1)
    data = torch.randint(0, 11, (16 * 5 + 1,))
    assert evaluate_split_strided(tiny_model, data, 16, "cpu", stride=16) == pytest.approx(
        evaluate_split(tiny_model, data, 16, "cpu"))
