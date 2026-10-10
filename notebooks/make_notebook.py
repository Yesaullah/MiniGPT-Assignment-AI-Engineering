"""Builds notebooks/minigpt_colab.ipynb (run: python notebooks/make_notebook.py)."""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# MiniGPT — a small GPT trained from scratch on Tiny Shakespeare

Runs the whole project on Colab: tests, training, ablations, the target-leakage demonstration,
the LSTM-vs-attention benchmark, evaluation and all report figures.

**Runtime → Change runtime type → T4 GPU** for training. Inference only (`RUN_TRAINING = False`)
also works on a CPU runtime.

**If the runtime disconnects:** reconnect and run the cells again from the top. Run folders live
on Google Drive, so every training command continues from its last checkpoint (saved every
250 steps) instead of starting over, and finished runs are skipped.
""")
md("## 0. Setup")
code("""
REPO_URL = "https://github.com/Yesaullah/MiniGPT-Assignment-AI-Engineering.git"
RUN_TRAINING = True   # False: only download the released model and run inference (no GPU or Drive)
DRIVE_DIR = "/content/drive/MyDrive/minigpt"  # run folders (checkpoints, logs) are kept here

import os, shutil, subprocess
if not os.path.exists("/content/minigpt"):
    subprocess.run(["git", "clone", REPO_URL, "/content/minigpt"], check=True)
%cd /content/minigpt
!pip install -q -r requirements.txt
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if RUN_TRAINING:
    assert torch.cuda.is_available(), "Training needs a GPU runtime (Runtime -> Change runtime type -> T4 GPU)"
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(f"{DRIVE_DIR}/runs", exist_ok=True)
    if not os.path.islink("outputs/runs"):  # outputs/runs -> Drive, so checkpoints survive a disconnect
        shutil.rmtree("outputs/runs", ignore_errors=True)
        os.symlink(f"{DRIVE_DIR}/runs", "outputs/runs")
    print("run folders:", os.path.realpath("outputs/runs"), sorted(os.listdir("outputs/runs")))
""")
md("## 1. Tests (attention, mask, shapes, overfit, data, training, API)")
code("!python -m pytest")
md("## 2. Quick inference with the released model (no training needed)")
code("""
if not RUN_TRAINING:
    !python -m src.artifacts fetch
    !python -m src.generate --prompt "ROMEO:" --max-new-tokens 300
""")
md("""
## 3. Pilot: time 300 baseline steps first

Measures this GPU's time per step and per evaluation, and estimates the time for the full plan.
Run it once before section 4; set `RUN_PILOT = False` afterwards.
""")
code("""
RUN_PILOT = True
if RUN_TRAINING and RUN_PILOT:
    !python -m src.train --config configs/tiny_gpt.yaml --set train.max_steps=300 --set train.eval_interval=100 --set train.out_dir=outputs/runs/pilot
    import json
    from src.config import load_config
    s = json.load(open("outputs/runs/pilot/summary.json"))
    evals = [r for r in map(json.loads, open("outputs/runs/pilot/metrics.jsonl")) if "eval_seconds" in r]
    eval_s = sum(r["eval_seconds"] for r in evals) / len(evals)
    base = load_config("configs/tiny_gpt.yaml").train
    full = base.max_steps * s["ms_per_step"] / 1000 + (base.max_steps // base.eval_interval + 1) * eval_s
    print(f"{s['gpu']}: {s['ms_per_step']:.1f} ms/step, {eval_s:.1f} s per evaluation, "
          f"peak memory {s['peak_memory_mb']:.0f} MB")
    print(f"Baseline estimate: {full / 60:.0f} min. Seven configurations: roughly {7 * full / 60:.0f} min "
          "(depth 6 and context 256 cost more than the baseline, depth 2 and context 64 less).")
""")
md("""
## 4. Train the baseline (4 layers, d_model 256, 4 heads, context 128; 3.19M parameters)
`--resume` continues from the last checkpoint on Drive if the session restarted.
""")
code("""
if RUN_TRAINING:
    !python -m src.train --config configs/tiny_gpt.yaml --resume
""")
md("## 5. Required ablations: depth 2/6, heads 2/8, context 64/256 (one change at a time)")
code("""
if RUN_TRAINING:
    !python -m experiments.ablation --run
""")
md("## 6. Lesson 06: target leakage (no causal mask) vs causal, same schedule")
code("""
if RUN_TRAINING:
    !python -m experiments.mask_experiment --steps 1500
""")
md("## 7. Lesson 01: attention vs LSTM timing and memory")
code("!python -m experiments.lstm_benchmark")
md("## 8. Evaluation (test split once), samples, figures")
code("""
CKPT = "outputs/runs/baseline/best.pt" if RUN_TRAINING else "checkpoints/model.pt"
if RUN_TRAINING:
    !python -m experiments.evaluate --checkpoint {CKPT} --test
    !python -m experiments.make_figures
    !mkdir -p {DRIVE_DIR} && cp -r outputs/evidence outputs/figures {DRIVE_DIR}/
!python -m experiments.attention_visualization --checkpoint {CKPT}
""")
md("## 9. Export the release model and try the CLI")
code("""
if RUN_TRAINING:
    !python -m src.artifacts export {CKPT} checkpoints/model.pt
    !cp checkpoints/model.pt checkpoints/model.pt.sha256 {DRIVE_DIR}/
!python -m src.generate --checkpoint checkpoints/model.pt --prompt "The future of artificial intelligence" --max-new-tokens 80
""")
md("## 10. Download everything (figures, evidence, run logs, model)")
code("""
!zip -qr minigpt_results.zip outputs -x "*.pt"
!zip -q minigpt_results.zip checkpoints/model.pt checkpoints/model.pt.sha256
from google.colab import files
files.download("minigpt_results.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("notebooks/minigpt_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/minigpt_colab.ipynb")
