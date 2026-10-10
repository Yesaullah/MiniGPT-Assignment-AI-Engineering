"""Generate text from a trained checkpoint.

python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 [--temperature 0.8] [--top-k 20]
                       [--seed 0] [--checkpoint checkpoints/model.pt] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from src.tokenizer import UnknownCharacterError
from src.utils import download_verified, get_device, load_checkpoint, make_generator, sha256_file

DEFAULT_CHECKPOINT = "checkpoints/model.pt"
RELEASE_FILE = "checkpoints/RELEASE.json"  # {"url": ..., "sha256": ...}


def resolve_checkpoint(path: str | None) -> str:
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if os.path.exists(RELEASE_FILE):
        # The released model: a cached copy is used only if it matches the published SHA-256.
        with open(RELEASE_FILE, encoding="utf-8") as f:
            release = json.load(f)
        if not (os.path.exists(DEFAULT_CHECKPOINT) and sha256_file(DEFAULT_CHECKPOINT) == release["sha256"]):
            print(f"Downloading model from {release['url']} ...", file=sys.stderr)
        return download_verified(release["url"], DEFAULT_CHECKPOINT, release["sha256"])
    if os.path.exists(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT
    raise FileNotFoundError("No checkpoint: pass --checkpoint or train a model first")


def run_generation(model, cfg, tokenizer, prompt: str, max_new_tokens: int, temperature: float,
                   top_k: int | None, seed: int, device: str) -> dict:
    used_prompt = prompt if prompt else "\n"  # empty prompt -> newline, which starts speeches
    ids = tokenizer.encode(used_prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = make_generator(device, seed)
    start = time.perf_counter()
    out = model.generate(idx, max_new_tokens, temperature, top_k, gen)
    seconds = time.perf_counter() - start
    text = tokenizer.decode(out[0, len(ids):].tolist())
    return {
        "prompt": prompt, "generated": text, "prompt_truncated": len(ids) > cfg.model.context_length,
        "empty_prompt_fallback": not prompt, "max_new_tokens": max_new_tokens,
        "temperature": temperature, "top_k": top_k, "seed": seed,
        "model_parameters": model.num_params(), "context_length": cfg.model.context_length,
        "vocab_size": tokenizer.vocab_size, "seconds": round(seconds, 3),
        "chars_per_second": round(len(text) / seconds, 1) if seconds > 0 else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text with MiniGPT.")
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", action="store_true", help="print only JSON on stdout")
    args = p.parse_args()

    try:
        path = resolve_checkpoint(args.checkpoint)
        device = get_device(args.device)
        model, cfg, tokenizer, _ = load_checkpoint(path, device)
        result = run_generation(model, cfg, tokenizer, args.prompt.replace("\\n", "\n"),
                                args.max_new_tokens, args.temperature, args.top_k, args.seed, device)
    except UnknownCharacterError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    result["checkpoint_sha256"] = sha256_file(path)[:12]

    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['prompt']}{result['generated']}")
    print()
    print(f"model parameters: {result['model_parameters'] / 1e6:.2f}M ({result['model_parameters']:,})")
    print(f"context length: {result['context_length']}")
    print(f"temperature: {result['temperature']}")
    print(f"top-k: {result['top_k']}")
    print(f"seed: {result['seed']}")
    print(f"max new tokens: {result['max_new_tokens']}")
    print(f"vocab size: {result['vocab_size']} (character-level)")
    print(f"checkpoint: {path} (sha256 {result['checkpoint_sha256']}...)")
    print(f"time: {result['seconds']} s ({result['chars_per_second']} chars/s)")


if __name__ == "__main__":
    main()
