"""Report figures 1-6 and 9 (handout Section 12), built from code and run logs.

python -m experiments.make_figures [--only 1,2,3]
Figure 7: experiments.attention_visualization; figure 8: experiments.evaluate;
figure 10: real CLI/API screenshots.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from experiments.common import (RUNS_DIR, fmt_matrix, plt, save_fig, save_text, text_figure)
from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from src.config import load_config, with_vocab
from src.gpt import GPT
from src.positional_encoding import sinusoidal_encoding

Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def fig1_hand_calculation() -> None:
    scores = Q @ K.T
    scaled = scores / math.sqrt(2)
    weights = torch.softmax(scaled, dim=-1)
    out = weights @ V
    masked = scaled.masked_fill(~causal_mask(3), float("-inf"))
    m_weights = torch.softmax(masked, dim=-1)
    m_out = m_weights @ V
    impl_out, _ = scaled_dot_product_attention(Q, K, V, causal_mask(3))
    assert torch.allclose(impl_out, m_out)
    text = "\n".join([
        "3 tokens, d_k = d_v = 2",
        "Q =\n" + fmt_matrix(Q, 0), "K =\n" + fmt_matrix(K, 0), "V =\n" + fmt_matrix(V, 0),
        "Step 1  Q K^T =\n" + fmt_matrix(scores, 0),
        "Step 2  Q K^T / sqrt(2) =\n" + fmt_matrix(scaled),
        "Step 3  softmax (row-wise) =\n" + fmt_matrix(weights),
        "Step 4  weights @ V =\n" + fmt_matrix(out),
        "Step 5  causal mask applied to the scaled scores =\n" + fmt_matrix(masked),
        "Step 6  masked softmax =\n" + fmt_matrix(m_weights),
        "Step 7  masked output =\n" + fmt_matrix(m_out),
        "Implementation (src.attention) reproduces step 7 exactly: True",
    ])
    save_text("hand_calculation.txt", text)
    text_figure(text, "fig1_hand_calculation.png", "Figure 1. Hand-calculated attention")


def fig2_causal_mask() -> None:
    T = 8
    mask = causal_mask(T)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(mask, cmap="Greens")
    axes[0].set_title("GPT causal mask (green = allowed)")
    axes[1].imshow(torch.ones(T, T), cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("BERT-style bidirectional (all allowed)")
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 1).eval()
    mha.store_weights = True
    mha(torch.randn(1, T, 32))
    im = axes[2].imshow(mha.last_weights[0, 0], cmap="viridis")
    axes[2].set_title("Causal attention weights (upper triangle = 0)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("key position j")
        ax.set_ylabel("query position i")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
    save_fig(fig, "fig2_causal_mask.png")


def fig3_shape_trace() -> None:
    mha = MultiHeadAttention(d_model=256, n_heads=4).eval()
    mha.trace = []
    mha(torch.randn(2, 128, 256))
    lines = ["MultiHeadAttention(d_model=256, n_heads=4, d_head=64), batch B=2, T=128", ""]
    lines += [f"{name:<32} {str(shape)}" for name, shape in mha.trace]
    text = "\n".join(lines)
    save_text("shape_trace.txt", text)
    text_figure(text, "fig3_shape_trace.png", "Figure 3. Multi-head tensor-shape trace")


def fig4_positional_encoding() -> None:
    pe = sinusoidal_encoding(128, 256)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    im = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu")
    axes[0].set_xlabel("position")
    axes[0].set_ylabel("dimension")
    axes[0].set_title("Sinusoidal encoding (128 positions x 256 dims)")
    fig.colorbar(im, ax=axes[0])
    sim = pe @ pe.T
    im = axes[1].imshow(sim, cmap="magma")
    axes[1].set_title("Dot product between positions")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("position")
    fig.colorbar(im, ax=axes[1])
    save_fig(fig, "fig4_positional_encoding.png")


def fig5_model_summary() -> None:
    cfg = with_vocab(load_config("configs/tiny_gpt.yaml"), 65)
    model = GPT(cfg.model)
    rows = [f"{'module':<34}{'shape':<22}{'params':>10}"]
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and not name.startswith("blocks.0."):
            continue
        rows.append(f"{name:<34}{str(tuple(p.shape)):<22}{p.numel():>10,}")
    per_block = sum(p.numel() for p in model.blocks[0].parameters())
    rows += ["", f"(blocks.1-{cfg.model.n_layers - 1} repeat blocks.0: {per_block:,} params each)",
             f"total parameters: {model.num_params():,}",
             f"config: {json.dumps({k: v for k, v in cfg.model.__dict__.items()})}"]
    text = "\n".join(rows)
    save_text("model_summary.txt", text)
    text_figure(text, "fig5_model_summary.png", "Figure 5. Model summary")


def read_metrics(run: str) -> list[dict]:
    path = os.path.join(RUNS_DIR, run, "metrics.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fig6_loss_curve(run: str = "baseline") -> None:
    rows = read_metrics(run)
    evals = [r for r in rows if "val_loss" in r]
    batches = [r for r in rows if "val_loss" not in r and r.get("train_batch_loss") is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["step"] for r in batches], [r["train_batch_loss"] for r in batches], alpha=0.35,
            label="training minibatch loss (dropout on)")
    ax.plot([r["step"] for r in evals], [r["train_subset_loss"] for r in evals], "o-",
            label="train subset (eval mode)")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], "o-", label="validation")
    best = min(evals, key=lambda r: r["val_loss"])
    ax.axvline(best["step"], ls="--", c="gray", lw=1)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats per character)")
    ax.set_title(f"Training vs validation loss ({run}); best val {best['val_loss']:.3f} "
                 f"at step {best['step']}")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"fig6_loss_curve_{run}.png")


def fig9_ablations() -> None:
    from experiments.ablation import collect

    rows = collect()
    if not rows:
        print("no ablation summaries found; skipping figure 9")
        return
    groups = {"depth": ["depth_2", "baseline", "depth_6"], "heads": ["heads_2", "baseline", "heads_8"],
              "context": ["context_64", "baseline", "context_256"]}
    by = {r["run"]: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, runs) in zip(axes, groups.items()):
        present = [r for r in runs if r in by]
        vals = [by[r]["best_val_bpc"] for r in present]
        labels = [by[r]["label"] for r in present]
        bars = ax.bar(labels, vals, color=["#9bb", "#7a2e2e", "#9bb"][: len(present)])
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylim(min(vals) * 0.97 if vals else 0, max(vals) * 1.02 if vals else 1)
        ax.set_title(f"{name} ablation")
        ax.set_ylabel("best validation BPC (lower is better)")
    save_fig(fig, "fig9_ablations.png")


FIGS = {"1": fig1_hand_calculation, "2": fig2_causal_mask, "3": fig3_shape_trace,
        "4": fig4_positional_encoding, "5": fig5_model_summary, "6": fig6_loss_curve,
        "9": fig9_ablations}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=",".join(FIGS))
    args = p.parse_args()
    for key in args.only.split(","):
        try:
            FIGS[key]()
        except FileNotFoundError as e:
            print(f"figure {key} skipped: {e}")


if __name__ == "__main__":
    main()
