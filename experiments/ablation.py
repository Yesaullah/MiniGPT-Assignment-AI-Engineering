"""Required ablations (depth, heads, context), one change at a time from the baseline,
each trained with seeds 0, 1 and 2 (D26).

python -m experiments.ablation --run        # train or resume every unfinished run, then summarise
python -m experiments.ablation              # summarise existing runs only
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from experiments.common import ABLATIONS, RUNS_DIR, save_text
from src.config import load_config
from src.train import finished_run, train

SEEDS = [0, 1, 2]  # fixed before any ablation result was seen (D26)
LABELS = {"baseline": "baseline (L4 h4 T128)", "depth_2": "2 layers", "depth_6": "6 layers",
          "heads_2": "2 heads", "heads_8": "8 heads", "context_64": "context 64",
          "context_256": "context 256"}


def config_paths(run: str) -> list[str]:
    paths = ["configs/tiny_gpt.yaml"]
    if run != "baseline":
        paths.append(f"configs/experiments/{run}.yaml")
    return paths


def run_dir(run: str, seed: int) -> str:
    """Seed 0 keeps the configs' own folder, so the section-4 baseline run is reused."""
    return os.path.join(RUNS_DIR, run if seed == 0 else f"{run}_s{seed}")


def run_config(run: str, seed: int):
    overrides = [] if seed == 0 else [f"train.seed={seed}", f"train.out_dir={run_dir(run, seed)}"]
    return load_config(config_paths(run), overrides)


def collect() -> list[dict]:
    """One row per finished (configuration, seed) run."""
    rows = []
    for run in ABLATIONS:
        for seed in SEEDS:
            path = os.path.join(run_dir(run, seed), "summary.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            m = s["config"]["model"]
            rows.append({"run": run, "seed": s["config"]["train"]["seed"], "label": LABELS[run],
                         "n_layers": m["n_layers"], "n_heads": m["n_heads"],
                         "context": m["context_length"], "params": s["params"],
                         "best_val_loss": s["best_val_loss"], "best_val_bpc": s["best_val_bpc"],
                         "best_step": s["best_step"], "train_minutes": s["train_seconds"] / 60,
                         "ms_per_step": s.get("ms_per_step"),
                         "tokens_per_second": s.get("train_tokens_per_second"),
                         "peak_memory_mb": s.get("peak_memory_mb")})
    return rows


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample standard deviation (None with a single value)."""
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def aggregate(rows: list[dict]) -> list[dict]:
    """Per configuration: mean ± sample std over seeds, and the paired difference from the
    baseline at the same seed (variant BPC − baseline BPC)."""
    base = {r["seed"]: r["best_val_bpc"] for r in rows if r["run"] == "baseline"}
    out = []
    for run in ABLATIONS:
        mine = [r for r in rows if r["run"] == run]
        if not mine:
            continue
        bpc_mean, bpc_std = _mean_std([r["best_val_bpc"] for r in mine])
        diffs = [r["best_val_bpc"] - base[r["seed"]] for r in mine if r["seed"] in base]
        diff_mean, diff_std = _mean_std(diffs) if diffs and run != "baseline" else (None, None)
        timed = [r for r in mine if r["ms_per_step"]]
        mem = [r["peak_memory_mb"] for r in mine if r["peak_memory_mb"]]
        out.append({"run": run, "label": mine[0]["label"], "params": mine[0]["params"],
                    "seeds": [r["seed"] for r in mine], "bpc_per_seed": [r["best_val_bpc"] for r in mine],
                    "bpc_mean": bpc_mean, "bpc_std": bpc_std,
                    "diff_vs_baseline_mean": diff_mean, "diff_vs_baseline_std": diff_std,
                    "ms_per_step": statistics.mean(r["ms_per_step"] for r in timed) if timed else None,
                    "tokens_per_second": (statistics.mean(r["tokens_per_second"] for r in timed)
                                          if timed else None),
                    "peak_memory_mb": max(mem) if mem else None,
                    "train_minutes": statistics.mean(r["train_minutes"] for r in mine)})
    return out


def _pm(mean: float | None, std: float | None, digits: int = 4, sign: bool = False) -> str:
    if mean is None:
        return "-"
    text = f"{mean:+.{digits}f}" if sign else f"{mean:.{digits}f}"
    return text + (f" ± {std:.{digits}f}" if std is not None else "")


def summarise() -> None:
    rows = collect()
    agg = aggregate(rows)
    lines = ["| configuration | params | seeds | best val BPC (mean ± std) | difference vs baseline (paired) "
             "| ms/step | train tokens/s | peak MB | train min |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for a in agg:
        tps = f"{a['tokens_per_second']:,.0f}" if a["tokens_per_second"] else "-"
        ms = f"{a['ms_per_step']:.1f}" if a["ms_per_step"] else "-"
        mem = f"{a['peak_memory_mb']:.0f}" if a["peak_memory_mb"] else "-"
        lines.append(f"| {a['label']} | {a['params']:,} | {','.join(map(str, a['seeds']))} | "
                     f"{_pm(a['bpc_mean'], a['bpc_std'])} | "
                     f"{_pm(a['diff_vs_baseline_mean'], a['diff_vs_baseline_std'], sign=True)} | "
                     f"{ms} | {tps} | {mem} | {a['train_minutes']:.1f} |")
    lines += ["", "Per seed:", "", "| configuration | seed | best val loss | best val BPC | best step |",
              "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['label']} | {r['seed']} | {r['best_val_loss']:.4f} | {r['best_val_bpc']:.4f} | "
              f"{r['best_step']} |" for r in rows]
    note = ("\n\nSeeds 0, 1 and 2 per configuration (fixed in advance, D26); std is the sample standard "
            "deviation over seeds. The paired difference compares each variant with the baseline at the "
            "same seed. Three seeds give a rough, not precise, estimate of seed-to-seed variation. LR 1e-3 "
            "and dropout 0.1 are shared by all runs. ms/step and tokens/s count training steps only; train "
            "min is wall clock including evaluation; peak MB is the largest GPU peak over the seeds.\n")
    save_text("ablation_results.md", "\n".join(lines) + note)
    with open(os.path.join("outputs", "evidence", "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "configurations": agg}, f, indent=2)
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="train or resume runs that are not finished")
    p.add_argument("--only", default=",".join(ABLATIONS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = p.parse_args()
    if args.run:
        # All configurations at one seed before the next seed, so every finished seed is a full set.
        for seed in map(int, args.seeds.split(",")):
            for run in args.only.split(","):
                cfg = run_config(run, seed)
                if finished_run(cfg):  # raises if the saved run used a different config
                    print(f"{run} seed {seed}: already done")
                    continue
                print(f"=== training {run}, seed {seed} ===")
                train(cfg, resume=True)
    summarise()


if __name__ == "__main__":
    main()
