"""Train MiniGPT on Tiny Shakespeare.

python -m src.train --config configs/tiny_gpt.yaml [--config configs/experiments/depth_2.yaml]
                    [--set train.max_steps=200] [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import torch
import yaml

from src.config import Config, load_config, with_vocab
from src.dataset import DATA_SHA256, download_data, eval_windows, get_batch, split_text
from src.gpt import GPT
from src.tokenizer import CharTokenizer
from src.utils import (get_device, git_commit, load_checkpoint, make_generator, save_checkpoint,
                       set_seed, write_json)


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr."""
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
    min_lr = t.lr * t.min_lr_ratio
    return min_lr + 0.5 * (t.lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_split(model: GPT, data: torch.Tensor, context_length: int, device: str,
                   batch_size: int = 64, causal_override: bool | None = None) -> float:
    """Mean next-token loss (nats) over every target of the split, non-overlapping windows."""
    was_training = model.training
    model.eval()
    xs, ys = eval_windows(data, context_length)
    total, count = 0.0, 0
    for i in range(0, len(xs), batch_size):
        x, y = xs[i : i + batch_size].to(device), ys[i : i + batch_size].to(device)
        kpm = None if model.cfg.causal else (y != -100)
        if causal_override is not None:
            saved = [b.attn.causal for b in model.blocks]
            for b in model.blocks:
                b.attn.causal = causal_override
        logits, _ = model(x, key_padding_mask=kpm)
        if causal_override is not None:
            for b, c in zip(model.blocks, saved):
                b.attn.causal = c
        valid = y != -100
        loss = torch.nn.functional.cross_entropy(logits[valid].float(), y[valid], reduction="sum")
        total += loss.item()
        count += int(valid.sum())
    model.train(was_training)
    return total / count


def train_subset(train_data: torch.Tensor, size: int, pieces: int = 8) -> torch.Tensor:
    """Evenly spaced excerpts of the training split, same total size as validation."""
    piece = size // pieces
    step = (len(train_data) - piece) // (pieces - 1)
    return torch.cat([train_data[i * step : i * step + piece] for i in range(pieces)])


def prepare_data(cfg: Config):
    text = download_data(cfg.data.path)
    train_text, val_text, test_text = split_text(text)
    tokenizer = CharTokenizer.from_text(train_text)
    enc = lambda s: torch.tensor(tokenizer.encode(s), dtype=torch.long)
    return tokenizer, enc(train_text), enc(val_text), enc(test_text)


def train(cfg: Config, resume: bool = False) -> dict:
    t = cfg.train
    set_seed(t.seed)
    device = get_device(t.device)
    tokenizer, train_data, val_data, _ = prepare_data(cfg)
    cfg = with_vocab(cfg, tokenizer.vocab_size)
    subset = train_subset(train_data, len(val_data))
    os.makedirs(t.out_dir, exist_ok=True)

    model = GPT(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(t.weight_decay), lr=t.lr,
                                  betas=(t.beta1, t.beta2), eps=1e-8)
    batch_gen = torch.Generator().manual_seed(t.seed + 1)
    start_step, best_val, elapsed_before = 0, float("inf"), 0.0
    last_path, best_path = os.path.join(t.out_dir, "last.pt"), os.path.join(t.out_dir, "best.pt")
    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        extra = state["extra"]
        start_step, best_val = extra["step"], extra["best_val_loss"]
        elapsed_before = extra.get("train_seconds", 0.0)
        batch_gen.set_state(state["batch_rng"])
        torch.set_rng_state(state["cpu_rng"])
        print(f"Resumed from step {start_step}")

    n_params = model.num_params()
    print(f"device={device} params={n_params:,} vocab={tokenizer.vocab_size}")
    with open(os.path.join(t.out_dir, "config.resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    manifest = {
        "params": n_params, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "data_sha256": DATA_SHA256, "git": git_commit(),
        "train_targets": len(train_data) - 1, "val_targets": len(val_data) - 1,
    }
    metrics_path = os.path.join(t.out_dir, "metrics.jsonl")
    samples_path = os.path.join(t.out_dir, "samples.jsonl")
    if not resume:
        for p in (metrics_path, samples_path):
            if os.path.exists(p):
                os.remove(p)

    def log(path: str, row: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def run_eval(step: int, train_loss: float | None) -> float:
        val_loss = evaluate_split(model, val_data, cfg.model.context_length, device)
        sub_loss = evaluate_split(model, subset, cfg.model.context_length, device)
        row = {"step": step, "train_batch_loss": train_loss, "train_subset_loss": sub_loss,
               "val_loss": val_loss, "val_bpc": val_loss / math.log(2), "lr": lr_at(step, cfg),
               "tokens_seen": step * t.batch_size * cfg.model.context_length,
               "elapsed_s": elapsed_before + time.time() - t0}
        log(metrics_path, row)
        gen = make_generator(device, 1234)
        prompt = torch.tensor([tokenizer.encode(t.sample_prompt)], device=device)
        out = model.generate(prompt, t.sample_tokens, cfg.generate.temperature, None, gen)
        log(samples_path, {"step": step, "prompt": t.sample_prompt,
                           "text": tokenizer.decode(out[0, prompt.size(1):].tolist())})
        print(f"step {step:5d} | train {train_loss if train_loss is None else round(train_loss, 4)} "
              f"| subset {sub_loss:.4f} | val {val_loss:.4f} (bpc {val_loss / math.log(2):.3f})")
        return val_loss

    t0 = time.time()
    if start_step == 0:
        best_val = run_eval(0, None)
    model.train()
    running = None
    for step in range(start_step, t.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(train_data, t.batch_size, cfg.model.context_length, batch_gen, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()
        running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
        done = step + 1
        if done % 50 == 0:
            log(metrics_path, {"step": done, "train_batch_loss": loss.item(),
                               "grad_norm": float(grad_norm), "lr": lr_at(step, cfg)})
        if done % t.eval_interval == 0 or done == t.max_steps:
            val_loss = run_eval(done, running)
            extra = {"step": done, "val_loss": val_loss, "best_val_loss": min(best_val, val_loss),
                     "train_seconds": elapsed_before + time.time() - t0}
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, cfg, tokenizer, extra)
            state_extra = dict(extra)
            save_checkpoint(last_path, model, cfg, tokenizer, state_extra, optimizer)
            ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
            ckpt["batch_rng"] = batch_gen.get_state()
            ckpt["cpu_rng"] = torch.get_rng_state()
            torch.save(ckpt, last_path)

    best = torch.load(best_path, map_location="cpu", weights_only=True)["extra"]
    summary = {**manifest, "best_step": best["step"], "best_val_loss": best["val_loss"],
               "best_val_bpc": best["val_loss"] / math.log(2),
               "best_val_perplexity_per_char": math.exp(best["val_loss"]),
               "train_seconds": elapsed_before + time.time() - t0,
               "tokens_per_second": (t.max_steps - start_step) * t.batch_size
               * cfg.model.context_length / max(1e-9, time.time() - t0),
               "peak_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                                  if device == "cuda" else None),
               "config": cfg.to_dict()}
    write_json(os.path.join(t.out_dir, "summary.json"), summary)
    print(f"Best val loss {best['val_loss']:.4f} at step {best['step']} "
          f"({summary['train_seconds'] / 60:.1f} min). Saved {best_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniGPT.")
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config or ["configs/tiny_gpt.yaml"], args.overrides)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
