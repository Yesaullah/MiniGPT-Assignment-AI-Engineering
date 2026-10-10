"""Lesson 01: sequential vs parallel depth. Our attention layer vs PyTorch's nn.LSTM.

Same width (256), batch 8, forward pass only, eval mode. The LSTM must process T steps one
after another (sequential depth T); attention computes all positions at once (depth 1) but
stores a T x T score matrix per head (memory grows with T^2).

python -m experiments.lstm_benchmark
"""
from __future__ import annotations

import json

import torch
import torch.utils.benchmark as benchmark

from experiments.common import plt, save_fig, save_text
from src.attention import MultiHeadAttention

LENGTHS = [64, 128, 256, 512, 1024]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, B = 256, 8
    attn = MultiHeadAttention(d, 4).to(device).eval()
    lstm = torch.nn.LSTM(d, d, batch_first=True).to(device).eval()
    rows = []
    for T in LENGTHS:
        x = torch.randn(B, T, d, device=device)
        row = {"T": T}
        for name, fn in (("attention", lambda: attn(x)), ("lstm", lambda: lstm(x))):
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    fn()
                    torch.cuda.synchronize()
                    row[f"{name}_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
                timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
                row[f"{name}_ms"] = timer.blocked_autorange(min_run_time=0.5).median * 1000
        rows.append(row)
        print(row)
    params = {"attention": sum(p.numel() for p in attn.parameters()),
              "lstm": sum(p.numel() for p in lstm.parameters())}
    result = {"device": device, "batch": B, "width": d, "params": params, "rows": rows}
    with open("outputs/evidence/lstm_benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    lines = [f"device {device}, batch {B}, width {d}; params: attention {params['attention']:,}, "
             f"LSTM {params['lstm']:,}", "T      attention ms   LSTM ms"]
    lines += [f"{r['T']:<6} {r['attention_ms']:>12.3f} {r['lstm_ms']:>9.3f}" for r in rows]
    save_text("lstm_benchmark.txt", "\n".join(lines))

    fig, axes = plt.subplots(1, 2 if device == "cuda" else 1, figsize=(12 if device == "cuda" else 6, 4))
    axes = axes if device == "cuda" else [axes]
    axes[0].plot(LENGTHS, [r["attention_ms"] for r in rows], "o-", label="our multi-head attention")
    axes[0].plot(LENGTHS, [r["lstm_ms"] for r in rows], "o-", label="nn.LSTM (optimized)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("sequence length T")
    axes[0].set_ylabel("forward time (ms, median)")
    axes[0].set_title(f"Forward latency on {device}")
    axes[0].legend()
    if device == "cuda":
        axes[1].plot(LENGTHS, [r["attention_peak_mb"] for r in rows], "o-", label="attention")
        axes[1].plot(LENGTHS, [r["lstm_peak_mb"] for r in rows], "o-", label="LSTM")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("sequence length T")
        axes[1].set_ylabel("peak memory (MB)")
        axes[1].set_title("Peak memory")
        axes[1].legend()
    save_fig(fig, "fig_lstm_vs_attention.png")


if __name__ == "__main__":
    main()
