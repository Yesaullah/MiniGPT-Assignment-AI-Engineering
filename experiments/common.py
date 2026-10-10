"""Shared helpers for experiment scripts."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = "outputs/figures"
EVIDENCE_DIR = "outputs/evidence"
RUNS_DIR = "outputs/runs"
ABLATIONS = ["baseline", "depth_2", "depth_6", "heads_2", "heads_8", "context_64", "context_256"]


def save_fig(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def save_text(name: str, text: str) -> str:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(EVIDENCE_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {path}")
    return path


def text_figure(text: str, name: str, title: str | None = None) -> str:
    lines = text.splitlines() or [""]
    width = max(len(line) for line in lines)
    fig = plt.figure(figsize=(max(6, width * 0.085), max(1.5, len(lines) * 0.22 + 0.6)))
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.98 if not title else 0.92, text, family="monospace", fontsize=9, va="top")
    return save_fig(fig, name)


def fmt_matrix(m, digits: int = 4) -> str:
    rows = []
    for row in m.tolist():
        cells = []
        for v in row:
            cells.append("  -inf " if v == float("-inf") else f"{v:7.{digits}f}")
        rows.append("[" + " ".join(cells) + " ]")
    return "\n".join(rows)
