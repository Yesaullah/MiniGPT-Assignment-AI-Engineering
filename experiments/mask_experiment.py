"""Lesson 06: why GPT needs the causal mask (target-leakage demonstration).

Two short runs with the identical schedule, optimizer, data and seed; only the mask differs:
  causal : the normal GPT model
  leak   : no causal mask, still trained on next-token targets, so position t can see x[t+1],
           which is its own target (BERT-style bidirectional attention used for the wrong task).
The leaky model is then evaluated with and without the mask, and asked to generate.

python -m experiments.mask_experiment [--steps 1000]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from experiments.common import plt, save_fig, save_text
from src.config import load_config
from src.train import evaluate_split, prepare_data, train
from src.utils import load_checkpoint, make_generator


def run(steps: int, configs: list[str]) -> dict:
    common = [f"train.max_steps={steps}", f"train.warmup_steps={max(1, steps // 20)}",
              f"train.eval_interval={max(1, steps // 4)}"]
    results = {}
    for name, causal in (("mask_causal", True), ("mask_leak", False)):
        out_dir = f"outputs/runs/{name}"
        if not os.path.exists(os.path.join(out_dir, "summary.json")):
            cfg = load_config(configs,
                              common + [f"model.causal={causal}", f"train.out_dir={out_dir}"])
            train(cfg)
        results[name] = os.path.join(out_dir, "best.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, val_data, _ = prepare_data(load_config("configs/tiny_gpt.yaml"))
    causal_model, cfg, tok, _ = load_checkpoint(results["mask_causal"], device)
    leak_model, _, _, _ = load_checkpoint(results["mask_leak"], device)
    T = cfg.model.context_length
    out = {
        "steps": steps, "params": causal_model.num_params(),
        "causal_model_val_loss": evaluate_split(causal_model, val_data, T, device),
        "leak_model_val_loss_no_mask": evaluate_split(leak_model, val_data, T, device),
        "leak_model_val_loss_with_causal_mask": evaluate_split(leak_model, val_data, T, device,
                                                               causal_override=True),
    }
    prompt = torch.tensor([tok.encode("ROMEO:\n")], device=device)
    for name, model in (("causal", causal_model), ("leak", leak_model)):
        ids = model.generate(prompt, 200, 0.8, None, make_generator(device, 0))
        out[f"{name}_sample"] = tok.decode(ids[0].tolist())
    return out


def report(out: dict) -> None:
    text = "\n".join([
        f"Both models: same architecture ({out['params']:,} parameters), {out['steps']} steps, identical schedule, data and seed; only the mask differs.",
        f"causal model, validation loss:                     {out['causal_model_val_loss']:.4f}",
        f"leak model (no mask), validation loss without mask: {out['leak_model_val_loss_no_mask']:.4f}",
        f"leak model, validation loss WITH causal mask:       "
        f"{out['leak_model_val_loss_with_causal_mask']:.4f}",
        "", "Causal model sample:", out["causal_sample"], "", "Leak model sample:", out["leak_sample"],
    ])
    save_text("mask_experiment.txt", text)
    with open("outputs/evidence/mask_experiment.json", "w") as f:
        json.dump(out, f, indent=2)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["causal model", "leak model\n(no mask, cheating)", "leak model\n(causal mask at test)"]
    vals = [out["causal_model_val_loss"], out["leak_model_val_loss_no_mask"],
            out["leak_model_val_loss_with_causal_mask"]]
    bars = ax.bar(labels, vals, color=["#7a2e2e", "#d9a", "#555"])
    ax.bar_label(bars, fmt="%.3f")
    ax.axhline(math.log(65), ls="--", c="gray", lw=1)
    ax.text(2.4, math.log(65), "uniform guess ln 65", va="bottom", ha="right", fontsize=8)
    ax.set_ylabel("validation cross-entropy (nats)")
    ax.set_title("Target leakage: training without the causal mask")
    save_fig(fig, "fig_mask_experiment.png")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--config", action="append", default=None)
    args = p.parse_args()
    report(run(args.steps, args.config or ["configs/tiny_gpt.yaml"]))


if __name__ == "__main__":
    main()
