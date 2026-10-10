"""Figure 7: attention weights of the trained model on a real prompt.

python -m experiments.attention_visualization [--checkpoint outputs/runs/baseline/best.pt]
"""
from __future__ import annotations

import argparse

import torch

from experiments.common import plt, save_fig
from src.utils import load_checkpoint

PROMPT = "ROMEO:\nBut, soft! what light through yonder window breaks?"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/runs/baseline/best.pt")
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()
    model, cfg, tok, _ = load_checkpoint(args.checkpoint, "cpu")
    prompt = args.prompt[-cfg.model.context_length :]
    ids = torch.tensor([tok.encode(prompt)])
    for block in model.blocks:
        block.attn.store_weights = True
    with torch.no_grad():
        model(ids)
    labels = [repr(c)[1:-1] for c in prompt]
    for layer in (0, cfg.model.n_layers - 1):
        weights = model.blocks[layer].attn.last_weights[0]  # (heads, T, T)
        h = weights.size(0)
        fig, axes = plt.subplots(1, h, figsize=(5 * h, 5))
        for i, ax in enumerate(axes):
            ax.imshow(weights[i], cmap="viridis")
            ax.set_title(f"layer {layer + 1}, head {i + 1}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=5, rotation=90)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_xlabel("attended-to character (key)")
        axes[0].set_ylabel("predicting from (query)")
        save_fig(fig, f"fig7_attention_layer{layer + 1}.png")


if __name__ == "__main__":
    main()
