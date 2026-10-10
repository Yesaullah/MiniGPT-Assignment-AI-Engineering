"""Re-score every ablation checkpoint (best.pt) on validation with overlapping windows, stride 32.

Training selected checkpoints with non-overlapping windows (D25), where longer-context models also
give each scored character more context on average. With stride 32 every scored character after
the first window sees at least T - 31 characters, so this separates "longer context helps" from
"the evaluation protocol favours longer context". A separate, labelled protocol: it does not change
which checkpoint was selected. Validation only; the test split is untouched.

python -m experiments.stride_eval [--stride 32]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import torch

from experiments.ablation import ABLATIONS, LABELS, SEEDS, run_dir
from experiments.common import save_text
from src.config import load_config
from src.train import evaluate_split_strided, prepare_data
from src.utils import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=32)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _, val_data, _ = prepare_data(load_config("configs/tiny_gpt.yaml"))

    rows = []
    for run in ABLATIONS:
        for seed in SEEDS:
            path = os.path.join(run_dir(run, seed), "best.pt")
            summary_path = os.path.join(run_dir(run, seed), "summary.json")
            if not os.path.exists(summary_path):
                continue
            model, cfg, tok, _ = load_checkpoint(path, device)
            assert tok.chars == tokenizer.chars, f"{path}: tokenizer differs from the data split"
            with open(summary_path, encoding="utf-8") as f:
                non_overlap = json.load(f)["best_val_bpc"]
            loss = evaluate_split_strided(model, val_data, cfg.model.context_length, device, args.stride)
            rows.append({"run": run, "seed": seed, "context": cfg.model.context_length,
                         "non_overlapping_bpc": non_overlap, "strided_bpc": loss / math.log(2)})
            print(f"{run} seed {seed}: non-overlapping {non_overlap:.4f}, "
                  f"stride {args.stride} {loss / math.log(2):.4f}")

    base = {r["seed"]: r for r in rows if r["run"] == "baseline"}
    lines = [f"| configuration | non-overlapping BPC | stride-{args.stride} BPC "
             f"| difference vs baseline, non-overlapping | difference vs baseline, stride {args.stride} |",
             "|---|---:|---:|---:|---:|"]

    def pm(values: list[float], sign: bool = False) -> str:
        if not values:
            return "-"
        mean = statistics.mean(values)
        text = f"{mean:+.4f}" if sign else f"{mean:.4f}"
        return text + (f" ± {statistics.stdev(values):.4f}" if len(values) > 1 else "")

    for run in ABLATIONS:
        mine = [r for r in rows if r["run"] == run]
        if not mine:
            continue
        paired = [r for r in mine if r["seed"] in base and run != "baseline"]
        lines.append(f"| {LABELS[run]} | {pm([r['non_overlapping_bpc'] for r in mine])} | "
                     f"{pm([r['strided_bpc'] for r in mine])} | "
                     f"{pm([r['non_overlapping_bpc'] - base[r['seed']]['non_overlapping_bpc'] for r in paired], True)} | "
                     f"{pm([r['strided_bpc'] - base[r['seed']]['strided_bpc'] for r in paired], True)} |")
    note = (f"\n\nValidation split, best.pt of each run (selected under the non-overlapping protocol). "
            f"Mean ± sample std over seeds; differences are paired with the baseline at the same seed. "
            f"Stride {args.stride}: every scored character after the first window sees at least "
            f"T - {args.stride - 1} characters of context.\n")
    save_text("stride_eval.md", "\n".join(lines) + note)
    with open(os.path.join("outputs", "evidence", "stride_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"stride": args.stride, "runs": rows}, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
