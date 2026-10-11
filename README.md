# MiniGPT: a small GPT built from scratch

A character-level, GPT-style causal language model (3.19M parameters) implemented from scratch in
PyTorch and trained on Tiny Shakespeare. It has a CLI, a FastAPI service deployed on Render, and a
Colab notebook that reproduces every result. No pretrained model or `transformers` model classes are used.

- **Live demo:** https://minigpt-assignment-ai-engineering.onrender.com (free tier: the first request after 15 idle minutes takes about a minute to wake)
- **Colab:** [`notebooks/minigpt_colab.ipynb`](notebooks/minigpt_colab.ipynb)
- **Report:** [Medium article](https://medium.com/@yesaullahsheikh/what-3-2-million-parameters-learn-from-shakespeare-building-a-small-gpt-from-scratch-7e551858f867) · design log: [`docs/decisions.md`](docs/decisions.md)

## Where things live

| What | Where |
|---|---|
| Model (embedding + sinusoidal positions → Pre-LN blocks → LM head) | [`src/gpt.py`](src/gpt.py) |
| Scaled dot-product + multi-head attention, causal mask | [`src/attention.py`](src/attention.py) |
| Positional encoding | [`src/positional_encoding.py`](src/positional_encoding.py) |
| Transformer block, feed-forward network | [`src/transformer_block.py`](src/transformer_block.py) |
| Tokenizer, data pipeline | [`src/tokenizer.py`](src/tokenizer.py), [`src/dataset.py`](src/dataset.py) |
| Training, generation CLI, config | [`src/train.py`](src/train.py), [`src/generate.py`](src/generate.py), [`src/config.py`](src/config.py) |
| REST API + web page | [`src/serve/`](src/serve/) |
| Tests | [`tests/`](tests/) |
| Experiments, figures | [`experiments/`](experiments/), [`outputs/figures/`](outputs/figures/), [`outputs/evidence/`](outputs/evidence/) |
| Configs | [`configs/tiny_gpt.yaml`](configs/tiny_gpt.yaml) (baseline), [`configs/experiments/`](configs/experiments/) (ablations) |

## Setup

```bash
git clone https://github.com/Yesaullah/MiniGPT-Assignment-AI-Engineering.git
cd MiniGPT-Assignment-AI-Engineering
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build for a GPU
pip install -r requirements.txt
python -m pytest                                       # 46 tests, CPU
```

## Dataset

Tiny Shakespeare (the Hugging Face `tiny_shakespeare` source), downloaded automatically from a pinned
commit and checked against its SHA-256 ([`data/README.md`](data/README.md)). Contiguous split
(90/5/5): 1,003,854 / 55,770 / 55,770 characters. Tokenizer: character-level, 65 symbols built from the
training split, no special tokens. Characters outside the vocabulary are rejected with an error.

## Train

```bash
python -m src.train --config configs/tiny_gpt.yaml                    # baseline, about 7 min on a T4
python -m src.train --config configs/tiny_gpt.yaml --config configs/experiments/tiny_debug.yaml   # small CPU smoke run
python -m src.train --config configs/tiny_gpt.yaml --resume           # continue from last.pt
python -m experiments.ablation --run                                  # depth, heads, context ablations, seeds 0-2
python -m src.config --config configs/tiny_gpt.yaml --set model.n_layers=6   # print a resolved config
```

Every parameter in handout Section 11 is configurable in YAML or with `--set section.key=value`.
Each run writes `config.resolved.yaml`, `metrics.jsonl` (train/val loss, LR, step), `samples.jsonl`
(a sample at every evaluation), `best.pt`, `last.pt` and `summary.json` to `outputs/runs/<name>/`.

## Generate (CLI)

```bash
python -m src.artifacts fetch            # download the released model (checks SHA-256)
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80
```

Expected output format:

```
Prompt: The future of artificial intelligence
Generated: The future of artificial intelligence ...
model parameters: 3.19M (3,192,832)
context length: 128
temperature: 0.8
...
```

Options: `--temperature` (0 = greedy), `--top-k`, `--seed`, `--checkpoint`, `--json`.

## API

```bash
uvicorn src.serve.app:app --port 8000      # needs checkpoints/model.pt
curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:\n", "max_new_tokens": 200, "temperature": 0.8, "seed": 0}'
```

Endpoints: `GET /` (web page), `GET /health`, `GET /model-info`, `POST /generate`, `GET /docs`.
Requests are validated (prompt ≤ 1000 characters, `max_new_tokens` 1–500, temperature 0–2, unknown
fields or characters → 422). One generation runs at a time.

**Deploy (Render, free):** New → Web Service → this repo → Runtime *Docker*. The Docker build
downloads the model listed in `checkpoints/RELEASE.json` and verifies its SHA-256.

## Results

All numbers come from [`outputs/evidence/`](outputs/evidence/) and the run logs in `outputs/runs/`.
BPC = bits per character (lower is better); uniform guessing over 65 characters is 6.02.

**Baseline** (4 layers, 4 heads, d_model 256, context 128; 3,192,832 parameters; 5,000 steps,
6.6 minutes on a Colab T4, fp32): best validation loss 1.385 nats/char = **1.999 BPC** at step 5,000
(seeds 0, 1, 2: 1.999 / 1.998 / 1.999). The test split, evaluated once after everything was fixed:
1.559 nats/char = **2.249 BPC**. Loss curve: [`fig6`](outputs/figures/fig6_loss_curve_baseline.png);
samples from fixed prompts: [`samples.md`](outputs/evidence/samples.md).

**Ablations** (one change at a time from the baseline, seeds 0-2 each, differences paired by seed;
[`ablation_results.md`](outputs/evidence/ablation_results.md), [`fig9`](outputs/figures/fig9_ablations.png)):

| Configuration | Parameters | Val BPC | vs baseline | Val BPC, stride 32 | vs baseline | ms/step | Peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3,192,832 | 1.999 | | 1.963 | | 75 | 794 |
| 2 layers | 1,613,312 | 2.062 | +0.064 | 2.031 | +0.068 | 38 | 439 |
| 6 layers | 4,772,352 | 2.006 | +0.008 | 1.970 | +0.007 | 113 | 1148 |
| 2 heads | 3,192,832 | 1.993 | -0.006 | 1.957 | -0.006 | 72 | 722 |
| 8 heads | 3,192,832 | 2.009 | +0.011 | 1.974 | +0.012 | 84 | 938 |
| context 64 | 3,192,832 | 2.029 | +0.031 | 1.960 | -0.003 | 71 | 722 |
| context 256 | 3,192,832 | 1.983 | -0.016 | 1.957 | -0.005 | 86 | 938 |

- Depth: 2 layers is clearly worse; 6 layers is slightly worse at 1.5 times the cost, and its
  validation loss turned up after step 3,500 to 4,250 on every seed (overfitting at the shared
  dropout of 0.1).
- Heads: small quality differences; 8 heads costs 12% more time per step and 18% more memory.
- Context: under the training-time protocol (non-overlapping windows) longer context looks better,
  but re-scoring the same checkpoints with overlapping windows (stride 32,
  [`stride_eval.md`](outputs/evidence/stride_eval.md)) removes most of the difference. The protocol
  gave longer-context models more context per scored character.
- Peak memory follows batch x heads x T^2: settings with the same number of attention scores per
  layer used the same memory (722 MB and 938 MB pairs).

**Other experiments:** training without the causal mask reaches a validation loss of 0.016 by copying
the next character, rises to 12.2 with the mask restored, and generates blank lines
([`mask_experiment.txt`](outputs/evidence/mask_experiment.txt)). Our attention layer is faster than
`nn.LSTM` at every length from 64 to 1024 on a T4, but its memory grows with T^2 (16 MB to 317 MB
against 28 MB to 70 MB; [`lstm_benchmark.txt`](outputs/evidence/lstm_benchmark.txt)). Greedy decoding
repeats 28% of its 10-character sequences, against 0% at temperature 0.8
([`evaluation.json`](outputs/evidence/evaluation.json)).

**Deployment** (Render free tier, 0.1 CPU, 512 MB; [`latency_render_free.md`](outputs/evidence/latency_render_free.md)):
100 new characters take 7.3 s median from a short prompt and 12.1 s from a prompt longer than the
context; 300 characters take 31 to 36 s.

## Known limitations

- Three training seeds per ablation configuration (D26): enough for a rough estimate of
  seed-to-seed variation, not a precise one. LR (1e-3) and dropout (0.1) were fixed, not tuned
  per architecture.
- Checkpoints were selected with non-overlapping validation windows, which favour longer contexts;
  the stride-32 re-scoring above is reported separately.
- Test BPC (2.249) is clearly worse than validation (1.999). Both held-out sets are short
  passages from the end of the corpus; we think the test passage is harder, but did not measure it.
- Character-level, 1 MB of text, 3M parameters: output is Shakespeare-like in form, not meaning.
- Free hosting is slow (0.1 CPU) and sleeps when idle.
- Planned but not done before the deadline (see `docs/decisions.md`, D25): LR × dropout grid,
  BPE tokenizer comparison, n-gram baselines, blind human rating, streaming API.

## Contributions and AI assistance

Design review, the decision log, code drafts and the article draft were produced with help from Claude (Anthropic).
The team reviewed, ran and verified the code and results. `docs/decisions.md` records the design
reasoning, including AI errors that were caught and corrected.

Contributions: Yesaullah Sheikh built the attention module, the Transformer block and the test
suite, and wrote the hand calculation. Murtaza Jafri ran the training, the ablations and the
deployment. Both reviewed each other's work and can explain every part of the model.

## References

Vaswani et al., *Attention Is All You Need* (2017) · Radford et al., GPT (2018) and GPT-2 (2019) ·
Devlin et al., BERT (2019) · Xiong et al., *On Layer Normalization in the Transformer Architecture*
(2020) · Hendrycks & Gimpel, GELU (2016) · Loshchilov & Hutter, AdamW (2019) · Karpathy, char-rnn
(dataset source) and nanoGPT (consulted for conventions; no code copied).
