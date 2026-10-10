"""Ablation bookkeeping: run folders per seed and the seed statistics in the results table."""
import os
import statistics

import pytest

from experiments.ablation import aggregate, run_config, run_dir


def row(run, seed, bpc):
    return {"run": run, "seed": seed, "label": run, "params": 1, "best_val_bpc": bpc,
            "ms_per_step": 10.0, "tokens_per_second": 100.0, "peak_memory_mb": None,
            "train_minutes": 1.0}


def test_seed_zero_reuses_the_config_folder_and_other_seeds_get_their_own():
    assert run_config("baseline", 0).train.out_dir == "outputs/runs/baseline"
    assert run_config("depth_2", 0).train.out_dir == "outputs/runs/depth_2"
    cfg = run_config("depth_2", 2)
    assert (cfg.train.seed, cfg.train.out_dir) == (2, run_dir("depth_2", 2))
    assert cfg.model.n_layers == 2
    assert run_dir("depth_2", 2) == os.path.join("outputs/runs", "depth_2_s2")


def test_aggregate_mean_std_and_paired_difference():
    rows = [row("baseline", 0, 1.50), row("baseline", 1, 1.52), row("baseline", 2, 1.51),
            row("depth_6", 0, 1.45), row("depth_6", 1, 1.49), row("depth_6", 2, 1.44)]
    by = {a["run"]: a for a in aggregate(rows)}
    d6 = by["depth_6"]
    assert d6["bpc_mean"] == pytest.approx(statistics.mean([1.45, 1.49, 1.44]))
    assert d6["bpc_std"] == pytest.approx(statistics.stdev([1.45, 1.49, 1.44]))
    diffs = [1.45 - 1.50, 1.49 - 1.52, 1.44 - 1.51]  # paired by seed
    assert d6["diff_vs_baseline_mean"] == pytest.approx(statistics.mean(diffs))
    assert d6["diff_vs_baseline_std"] == pytest.approx(statistics.stdev(diffs))
    assert by["baseline"]["diff_vs_baseline_mean"] is None


def test_single_seed_has_no_std():
    a = aggregate([row("baseline", 0, 1.5)])[0]
    assert a["bpc_std"] is None
