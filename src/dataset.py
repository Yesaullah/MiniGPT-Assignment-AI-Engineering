"""Tiny Shakespeare download, verified split, and next-token batches.

Split (matches the Hugging Face loader): 90% train, then the rest halved into
validation and test, as contiguous character ranges.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

import torch

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt"
)
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DATA_BYTES = 1_115_394


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_data(path: str = "data/input.txt") -> str:
    """Download the pinned corpus if missing; always verify the checksum. Returns the text."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            data = response.read()
        if sha256_bytes(data) != DATA_SHA256:
            raise RuntimeError("Downloaded corpus failed its SHA-256 check")
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        data = f.read()
    if sha256_bytes(data) != DATA_SHA256:
        raise RuntimeError(f"{path} does not match the pinned SHA-256; delete it and re-run")
    return data.decode("utf-8")


def split_text(text: str) -> tuple[str, str, str]:
    n = len(text)
    train_end = int(n * 0.9)
    val_end = train_end + int((n - train_end) * 0.5)
    return text[:train_end], text[train_end:val_end], text[val_end:]


def get_batch(data: torch.Tensor, batch_size: int, context_length: int,
              generator: torch.Generator, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Random contiguous windows. Targets are the inputs shifted by one: y[t] = x[t+1]."""
    starts = torch.randint(0, len(data) - context_length - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + context_length] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + 1 + context_length] for s in starts.tolist()])
    return x.to(device), y.to(device)


def eval_windows(data: torch.Tensor, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-overlapping windows covering every target of a split exactly once.

    The last window is right-padded; padded targets are -100 (ignored by the loss).
    """
    n_targets = len(data) - 1
    xs, ys = [], []
    for start in range(0, n_targets, context_length):
        x = data[start : start + context_length]
        y = data[start + 1 : start + 1 + context_length]
        pad = context_length - len(y)
        if pad:
            x = torch.cat([x[: len(y)], torch.zeros(pad, dtype=data.dtype)])
            y = torch.cat([y, torch.full((pad,), -100, dtype=data.dtype)])
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)
