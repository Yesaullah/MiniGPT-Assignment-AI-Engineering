"""Required ablations (depth, heads, context), one change at a time from the baseline.

python -m experiments.ablation --run        # train or resume every unfinished run, then summarise
python -m experiments.ablation              # summarise existing runs only
"""
from __future__ import annotations

import argparse
import json
import os

from experiments.common import ABLATIONS, RUNS_DIR, save_text
from src.config import load_config
from src.train import finished_run, train

LABELS = {"baseline": "baseline (L4 h4 T128)", "depth_2": "2 layers", "depth_6": "6 layers",
          "heads_2": "2 heads", "heads_8": "8 heads", "context_64": "context 64",
          "context_256": "context 256"}


def config_paths(run: str) -> list[str]:
    paths = ["configs/tiny_gpt.yaml"]
    if run != "baseline":
        paths.append(f"configs/experiments/{run}.yaml")
    return paths


def collect() -> list[dict]:
    rows = []
    for run in ABLATIONS:
        path = os.path.join(RUNS_DIR, run, "summary.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        m = s["config"]["model"]
        rows.append({"run": run, "label": LABELS[run], "n_layers": m["n_layers"],
                     "n_heads": m["n_heads"], "context": m["context_length"], "params": s["params"],
                     "best_val_loss": s["best_val_loss"], "best_val_bpc": s["best_val_bpc"],
                     "best_step": s["best_step"], "train_minutes": s["train_seconds"] / 60,
                     "tokens_per_second": s.get("train_tokens_per_second"),
                     "peak_memory_mb": s.get("peak_memory_mb")})
    return rows


def summarise() -> None:
    rows = collect()
    header = ("| run | layers | heads | context | params | best val loss | best val BPC | best step "
              "| train min | tokens/s | peak MB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines = [header]
    for r in rows:
        mem = f"{r['peak_memory_mb']:.0f}" if r["peak_memory_mb"] else "-"
        tps = f"{r['tokens_per_second']:,.0f}" if r["tokens_per_second"] else "-"
        lines.append(f"| {r['label']} | {r['n_layers']} | {r['n_heads']} | {r['context']} | "
                     f"{r['params']:,} | {r['best_val_loss']:.4f} | {r['best_val_bpc']:.4f} | "
                     f"{r['best_step']} | {r['train_minutes']:.1f} | {tps} | {mem} |")
    note = ("\n\nSingle seed (0) per configuration: differences are exploratory, with no seed-variance "
            "estimate. LR 1e-3 and dropout 0.1 are shared by all runs. tokens/s counts training steps "
            "only; train min is wall clock including evaluation; peak MB is the GPU peak of that run.\n")
    save_text("ablation_results.md", "\n".join(lines) + note)
    with open(os.path.join("outputs", "evidence", "ablation_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="train or resume runs that are not finished")
    p.add_argument("--only", default=",".join(ABLATIONS))
    args = p.parse_args()
    if args.run:
        for run in args.only.split(","):
            cfg = load_config(config_paths(run))
            if finished_run(cfg):  # raises if the saved run used a different config
                print(f"{run}: already done")
                continue
            print(f"=== training {run} ===")
            train(cfg, resume=True)
    summarise()


if __name__ == "__main__":
    main()
