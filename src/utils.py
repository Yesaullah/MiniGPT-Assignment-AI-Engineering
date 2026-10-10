"""Seeds, devices, checkpoints and verified artifact downloads."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import urllib.request

import torch

from src.config import Config, from_dict
from src.gpt import GPT
from src.tokenizer import CharTokenizer

SCHEMA_VERSION = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_generator(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def rng_state(batch_gen: torch.Generator) -> dict:
    """Every random stream training uses: CPU (and CUDA) default generators and the batch sampler."""
    state = {"cpu": torch.get_rng_state(), "batch": batch_gen.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()  # dropout on the GPU draws from these
    return state


def restore_rng_state(state: dict, batch_gen: torch.Generator) -> None:
    """The saved states must be CPU tensors: load checkpoints with map_location='cpu'."""
    torch.set_rng_state(state["cpu"])
    batch_gen.set_state(state["batch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str, model: GPT, cfg: Config, tokenizer: CharTokenizer,
                    extra: dict | None = None, optimizer: torch.optim.Optimizer | None = None,
                    rng: dict | None = None) -> None:
    """Only tensors and plain Python types, so it loads with weights_only=True.

    Written to a temporary file and then renamed, so an interrupted save never
    replaces the previous checkpoint with a partial file.
    """
    state = {
        "schema_version": SCHEMA_VERSION,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if rng is not None:
        state["rng"] = rng
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model, cfg, tokenizer, raw_state). Architecture comes from the checkpoint."""
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {state.get('schema_version')}")
    cfg = from_dict(state["config"])
    tokenizer = CharTokenizer.from_dict(state["tokenizer"])
    if cfg.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocab_size does not match its tokenizer")
    model = GPT(cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, cfg, tokenizer, state


def export_model(best_path: str, out_path: str) -> str:
    """Strip optimizer state from a checkpoint for release. Returns the SHA-256."""
    state = torch.load(best_path, map_location="cpu", weights_only=True)
    state.pop("optimizer", None)
    torch.save(state, out_path)
    digest = sha256_file(out_path)
    with open(out_path + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(out_path)}\n")
    return digest


def download_verified(url: str, path: str, sha256: str) -> str:
    """Download `url` to `path` unless a file with the right checksum is already there."""
    if os.path.exists(path) and sha256_file(path) == sha256:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
        f.write(response.read())
    actual = sha256_file(tmp)
    if actual != sha256:
        os.remove(tmp)
        raise RuntimeError(f"Checksum mismatch for {url}: expected {sha256}, got {actual}")
    os.replace(tmp, path)
    return path


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
