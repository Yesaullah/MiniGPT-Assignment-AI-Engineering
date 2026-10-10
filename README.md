# MiniGPT: a small GPT built from scratch

A character-level, GPT-style causal language model (3.19M parameters) implemented from scratch in
PyTorch and trained on Tiny Shakespeare. It has a CLI, a FastAPI service deployed on Render, and a
Colab notebook that reproduces every result. No pretrained model or `transformers` model classes are used.

- **Live demo:** https://minigpt-assignment-ai-engineering.onrender.com (free tier: the first request after 15 idle minutes takes about a minute to wake)
- **Colab:** [`notebooks/minigpt_colab.ipynb`](notebooks/minigpt_colab.ipynb)
- **Report:** Medium article (link) · design log: [`docs/decisions.md`](docs/decisions.md)

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
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build for a GPU
pip install -r requirements.txt
python -m pytest                                       # 34 tests, CPU
```

## Dataset

Tiny Shakespeare (the Hugging Face `tiny_shakespeare` source), downloaded automatically from a pinned
commit and checked against its SHA-256 ([`data/README.md`](data/README.md)). Contiguous split
(90/5/5): 1,003,854 / 55,770 / 55,770 characters. Tokenizer: character-level, 65 symbols built from the
training split, no special tokens. Characters outside the vocabulary are rejected with an error.

## Train

```bash
python -m src.train --config configs/tiny_gpt.yaml                    # baseline, ~X min on a T4
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

*(filled in from `outputs/evidence/` after training)*

## Known limitations

- Three training seeds per ablation configuration (D26): enough for a rough estimate of
  seed-to-seed variation, not a precise one. LR (1e-3) and dropout (0.1) were fixed, not tuned
  per architecture.
- Validation uses non-overlapping windows, so early positions in each window have little context.
- Character-level, 1 MB of text, 3M parameters: output is Shakespeare-like in form, not meaning.
- Free hosting is slow (0.1 CPU) and sleeps when idle.
- Planned but not done before the deadline (see `docs/decisions.md`, D25): LR × dropout grid,
  BPE tokenizer comparison, n-gram baselines, blind human rating, streaming API.

## AI assistance and contributions

Design review, the decision log and code drafts were produced with help from Claude (Anthropic).
The team reviewed, ran and verified the code and results. `docs/decisions.md` records the design
reasoning, including AI errors that were caught and corrected. Contributions: *(member A / member B)*.

## References

Vaswani et al., *Attention Is All You Need* (2017) · Radford et al., GPT (2018) and GPT-2 (2019) ·
Devlin et al., BERT (2019) · Xiong et al., *On Layer Normalization in the Transformer Architecture*
(2020) · Hendrycks & Gimpel, GELU (2016) · Loshchilov & Hutter, AdamW (2019) · Karpathy, char-rnn
(dataset source) and nanoGPT (consulted for conventions; no code copied).
