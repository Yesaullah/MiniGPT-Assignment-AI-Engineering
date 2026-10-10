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

**Runtime → Change runtime type → T4 GPU** for training. Inference (section 2) also works on CPU.
""")
md("## 0. Setup")
code("""
REPO_URL = "https://github.com/YOUR-USER/YOUR-REPO.git"  # <- set this
RUN_TRAINING = True       # False: only download the released model and run inference
SAVE_TO_DRIVE = True      # keeps outputs safe if the Colab session disconnects

import os, subprocess
if not os.path.exists("/content/minigpt"):
    subprocess.run(["git", "clone", REPO_URL, "/content/minigpt"], check=True)
%cd /content/minigpt
!pip install -q -r requirements.txt
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if RUN_TRAINING:
    assert torch.cuda.is_available(), "Training needs a GPU runtime (Runtime -> Change runtime type -> T4 GPU)"
""")
code("""
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/minigpt_outputs"
    os.makedirs(DRIVE_DIR, exist_ok=True)

def backup():
    if SAVE_TO_DRIVE:
        !rsync -a outputs/ {DRIVE_DIR}/outputs/
        print("backed up to", DRIVE_DIR)
""")
md("## 1. Tests (attention, mask, shapes, overfit, data, API)")
code("!python -m pytest")
md("## 2. Quick inference with the released model (no training needed)")
code("""
if not RUN_TRAINING:
    !python -m src.artifacts fetch
    !python -m src.generate --prompt "ROMEO:" --max-new-tokens 300
""")
md("""
## 3. Train the baseline (4 layers, d_model 256, 4 heads, context 128; 3.19M parameters)
Resumes from `outputs/runs/baseline/last.pt` if the session restarted (`--resume`).
""")
code("""
if RUN_TRAINING:
    !python -m src.train --config configs/tiny_gpt.yaml --resume
    backup()
""")
md("## 4. Required ablations: depth 2/6, heads 2/8, context 64/256 (one change at a time)")
code("""
if RUN_TRAINING:
    !python -m experiments.ablation --run
    backup()
""")
md("## 5. Lesson 06: target leakage (no causal mask) vs causal, same schedule")
code("""
if RUN_TRAINING:
    !python -m experiments.mask_experiment --steps 1500
    backup()
""")
md("## 6. Lesson 01: attention vs LSTM timing and memory")
code("!python -m experiments.lstm_benchmark")
md("## 7. Evaluation (test split once), samples, figures")
code("""
CKPT = "outputs/runs/baseline/best.pt"
!python -m experiments.evaluate --checkpoint {CKPT} --test
!python -m experiments.attention_visualization --checkpoint {CKPT}
!python -m experiments.make_figures
backup()
""")
md("## 8. Export the release model and try the CLI")
code("""
!python -m src.artifacts export {CKPT} checkpoints/model.pt
!python -m src.generate --checkpoint checkpoints/model.pt --prompt "The future of artificial intelligence" --max-new-tokens 80
""")
md("## 9. Download everything (figures, evidence, run logs, model)")
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
