"""Final evaluation: validation and (once) test loss/BPC, fixed-prompt samples, greedy vs sampling.

python -m experiments.evaluate [--checkpoint outputs/runs/baseline/best.pt] [--test]
"""
from __future__ import annotations

import argparse
import json
import math
import re

import torch

from experiments.common import save_text, text_figure
from src.config import load_config
from src.train import evaluate_split, prepare_data
from src.utils import load_checkpoint, make_generator, sha256_file

PROMPTS = ["ROMEO:\n", "First Citizen:\n", "KING RICHARD III:\n", "O, what a", "My lord, I"]
OOD_PROMPT = "The future of artificial intelligence"


def diversity(text: str) -> dict:
    words = re.findall(r"[A-Za-z']+", text)
    bigrams = list(zip(words, words[1:]))
    grams = [text[i : i + 10] for i in range(len(text) - 9)]
    return {"distinct_1": len(set(words)) / max(1, len(words)),
            "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
            "repeated_char10_rate": 1 - len(set(grams)) / max(1, len(grams))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/runs/baseline/best.pt")
    p.add_argument("--test", action="store_true", help="also report the test split (do this once)")
    p.add_argument("--length", type=int, default=300)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, tok, state = load_checkpoint(args.checkpoint, device)
    tokenizer, train_data, val_data, test_data = prepare_data(load_config("configs/tiny_gpt.yaml"))
    assert tokenizer.chars == tok.chars, "checkpoint tokenizer differs from the training split"
    train_words = set(re.findall(r"[A-Za-z']+", tok.decode(train_data.tolist())))
    T = cfg.model.context_length

    result = {"checkpoint": args.checkpoint, "sha256": sha256_file(args.checkpoint),
              "params": model.num_params(), "best_step": state["extra"].get("step")}
    val = evaluate_split(model, val_data, T, device)
    result.update(val_loss=val, val_bpc=val / math.log(2), val_char_perplexity=math.exp(val))
    if args.test:
        test = evaluate_split(model, test_data, T, device)
        result.update(test_loss=test, test_bpc=test / math.log(2), test_char_perplexity=math.exp(test))

    samples, md = [], []
    for prompt in PROMPTS + [OOD_PROMPT]:
        for mode, temp in (("temperature 0.8", 0.8), ("greedy", 0.0)):
            idx = torch.tensor([tok.encode(prompt)], device=device)
            out = model.generate(idx, args.length, temp, None, make_generator(device, 0))
            text = tok.decode(out[0, idx.size(1):].tolist())
            words = re.findall(r"[A-Za-z']+", text)
            stats = diversity(text)
            stats["train_word_coverage"] = sum(w in train_words for w in words) / max(1, len(words))
            samples.append({"prompt": prompt, "mode": mode, "text": text, **stats})
            md.append(f"### {prompt!r} ({mode}, seed 0)\n```\n{prompt}{text}\n```\n"
                      + ", ".join(f"{k} {v:.3f}" for k, v in stats.items()) + "\n")
    for mode in ("temperature 0.8", "greedy"):
        rows = [s for s in samples if s["mode"] == mode and s["prompt"] != OOD_PROMPT]
        result[f"mean_{mode.split()[0]}"] = {k: sum(r[k] for r in rows) / len(rows) for k in
                                             ("distinct_1", "distinct_2", "repeated_char10_rate",
                                              "train_word_coverage")}
    result["samples"] = samples
    with open("outputs/evidence/evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    save_text("samples.md", "\n".join(md))
    shown = [s for s in samples if s["mode"] == "temperature 0.8" and s["prompt"] in PROMPTS]
    fig_text = "\n\n".join(f">>> {s['prompt']!r}\n{s['prompt']}{s['text'][:240]}" for s in shown)
    text_figure(fig_text, "fig8_samples.png", "Figure 8. Generated samples (temperature 0.8, seed 0)")
    summary = {k: v for k, v in result.items() if k != "samples"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
