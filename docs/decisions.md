# MiniGPT Design Decisions

A log of the design decisions for the MiniGPT assignment (AI Engineering, Assignment 1).
Each entry records what was decided, why, and which alternatives were rejected, so every
choice can be explained and defended in the report and the oral demo.

## Context

- **Local machine:** no NVIDIA GPU; PyTorch 2.11 (CPU build), Python 3.13. Local work is
  CPU-only: tests, debugging, small runs, inference.
- **Training hardware:** Google Colab GPU (expected T4) for full training and ablations.
- **Dataset:** Tiny Shakespeare (required by the handout, Section 9).
- **Design status: D1–D24 locked on 2026-10-10.** Further changes only through a new,
  dated amendment entry. The next evidence comes from implementation tests and the Render
  and Colab pilots, not new design rounds.
- **Budget: zero spending (hard constraint, set 2026-10-10).** Training, hosting and
  storage must all be free. Paid plans are never the fallback. Free services are used
  **without a payment method** wherever possible, so running out of quota suspends a
  service instead of charging for it.

## Decisions

<!-- Entries are appended in the order they are decided. -->

### D1. Tokenization scheme

**Status:** Accepted · 2026-10-10

**Decision:** Character-level tokenization for the core model and all required ablations,
plus a planned comparison against our own byte-level BPE. The deployed model is chosen by a
ship rule fixed in advance.

#### Core tokenizer: character-level
- Text is kept exactly as-is: case, whitespace, punctuation and newlines.
- The vocabulary is built from the **training split only**.
- **Held-out coverage check before training:** list every character in the validation and
  test splits that is missing from the training vocabulary. If any exist, report them and
  settle the evaluation policy explicitly. Never drop or replace them silently, because that
  would distort the BPC comparison.
- **Unknown-character policy (CLI and API):** inputs containing characters outside the
  vocabulary are rejected with a clear validation error naming the offending characters.
  There is no untrained `<unk>` token and no silent stripping.
- No special tokens: training uses contiguous text windows, so BOS/EOS/PAD are not needed.

#### Experimental tokenizer: byte-level BPE (our own implementation)
- **1,024 tokens** = 256 byte tokens + 768 learned merges, with no special tokens.
- Merges are learned on the **training split only**, using GPT-2-style regex
  pre-tokenization so merges do not cross word/whitespace boundaries, and **deterministic
  tie-breaking** for equal pair counts.
- The vocabulary and the **ordered merge list** are saved to a file shipped with the checkpoint.
- Byte fallback can encode any input, but bytes never seen in training have untrained
  embeddings. If BPE ships, the API still rejects characters outside the training character set.

#### Comparison protocol
- Both tokenizers are evaluated on **identical held-out text**.
- **Bits per character (BPC)** is the shared metric. For BPE:
  `BPC = (summed token negative log-likelihood) / (ln 2 × number of evaluated characters)`.
  Each evaluated target is counted once. (Tiny Shakespeare is ASCII, so bytes = characters.)
- Also reported: measured compression (characters per BPE token on the training set),
  **non-embedding and total** parameter counts, training time, generation speed in
  characters/second, fixed-prompt samples and repetition rate.
- **Training exposure and compute** are documented for both runs (tokens, characters,
  steps, FLOPs or wall-clock time). Equal step counts alone do not make the comparison fair.
- **Matched-context control:** one extra character-level run with context ≈
  `BPE context length × training-set characters per BPE token`. BPE tokens vary in length,
  so this match is approximate; the actual character coverage of both models is reported.
  This separates "BPE helps because of tokenization" from "BPE helps because it sees more text".

#### Ship rule (fixed before any results)
- Selection uses the **validation split**. The **test split** is used only for final reporting.
- BPE ships only if it has **lower validation BPC** and its samples are **not worse** on the
  predetermined prompts, judged on readability, repetition and Shakespeare-like formatting
  (speaker names, line breaks).
- ~~If the BPC gap is smaller than about 2× the seed-to-seed standard deviation, repeat
  over additional seeds.~~ **Superseded by D24d**, which is the only BPE promotion rule: 3
  matched seeds, a gap of at least 0.02 BPC, the D15e human gate and the deployment gates.
- The tokenizer file's required contents are defined in D24b.

#### Interface and checkpoints
- One shared tokenizer interface: `encode`, `decode`, `vocab_size`, `save`, `load`.
- Every checkpoint stores, or points to, its exact tokenizer. The interface makes the
  **code** reusable across tokenizers; the **weights** are not interchangeable, and
  switching tokenizer means retraining.
- The MLM-vs-CLM experiment, if it needs a `[MASK]` token, gets its **own documented
  vocabulary extension** and does not change either tokenizer above.

#### Alternatives rejected
- **Character-level only:** simpler, but gives up a measured chance to improve the shipped model.
- **Pretrained GPT-2 tokenizer (50k vocab):** a 50,257 × d embedding would dwarf a ~3M-parameter
  Transformer trained on ~1 MB of text.
- **BPE as the main tokenizer by default:** better word-level coherence is a hypothesis on a
  corpus this small, not a given. It has to win under the ship rule.

#### Notes for the report and demo
- Character-level does not add information compared with BPE. It gives more prediction
  positions over the same text.
- nanoGPT's character-level Shakespeare result (val loss ≈ 1.47) is a **reference point only**:
  it uses 6 layers, d=384, context 256, ~10.7M parameters and its own split.

**Depends on:** D2 (train/validation/test split).

**Coverage check result (2026-10-10):** the training split contains all 65 distinct
characters; validation and test introduce no unseen characters. No special evaluation
policy is needed for the D2 split.

### D2. Data source and train/validation/test split

**Status:** Accepted · 2026-10-10

**Decision:** Download the canonical Tiny Shakespeare text from a **commit-pinned** URL,
verify its SHA-256 on the raw bytes, and reproduce the Hugging Face dataset's sequential
90/5/5 split exactly in our own code. The Hugging Face `datasets` loader is not used.

#### Source
- **Pinned URL:**
  `https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt`
  (commit `370cbcd`, 2015-05-21; the file there is byte-identical to `master`)
- **SHA-256:** `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- **Size:** 1,115,394 bytes = 1,115,394 characters (pure ASCII)
- The checksum is verified on the **downloaded bytes, before decoding**. Decoding is strict
  UTF-8. The text is **not** stripped, normalized or otherwise altered.
- **Why not the loader:** the Hugging Face dataset `karpathy/tiny_shakespeare` is
  script-based, and `datasets` ≥ 4.0.0 no longer runs dataset scripts. Pinning an older
  `datasets` just to load 1 MB would add a fragile dependency. The dataset card is still
  cited as the dataset source, and its script was the reference for the split.

#### Split (reproduces the Hugging Face script's arithmetic exactly)
```python
train_end = int(len(text) * 0.9)
val_end = train_end + int((len(text) - train_end) * 0.5)

train      = text[:train_end]         # 1,003,854 chars
validation = text[train_end:val_end]  #    55,770 chars
test       = text[val_end:]           #    55,770 chars
```
- The split is applied to the **raw text, before tokenization and window construction**.
- Every window's inputs **and** shifted targets stay inside their own split; no window
  crosses a split boundary.
- Roles (from D1): **validation** for model selection and ablation comparisons; **test**
  only for final reporting.

#### Repository handling
- A small committed **data manifest** (`data/manifest.json`) records the pinned URL,
  expected SHA-256, byte length, decoding policy and expected split lengths. The download
  script and tests check against it.
- `data/README.md` and the manifest are tracked. Downloaded text and processed caches are
  gitignored.
- Fallback only, if the pinned URL ever fails: commit `input.txt` itself (small and public
  domain), still verified against the same checksum.

#### Licensing note
- The Hugging Face dataset card does not specify a license.
- The text is Shakespeare (public domain); the file comes from Karpathy's char-rnn
  repository, which is MIT-licensed. The README documents these two facts separately and
  does not claim a license for the dataset card.

#### Evaluation and uncertainty
- Validation loss is computed **deterministically over the full validation region**, not
  on randomly sampled batches. The windowing details are decided in D3.
- No standard error is claimed from the window count: neighbouring passages are correlated,
  so window losses are not independent.
- When a ranking is close: use **paired comparisons over contiguous text blocks** (same
  blocks scored by both models) together with **multiple training seeds**. These measure
  different sources of uncertainty and both are reported.

#### Known limitation (for the report)
- Validation and test are **held-out contiguous regions, not held-out plays**. The
  boundaries cut mid-play and even mid-word: validation begins inside *The Taming of the
  Shrew* (GREMIO/BAPTISTA), and test begins mid-word ("…rance ta'en"). The held-out text
  therefore comes largely from one play's ending.
- The size of any distribution shift between training and held-out text has **not been
  measured**; it is stated as a limitation, not as a quantified effect.
- Matching the split does **not** by itself make our losses comparable to published numbers;
  evaluation context length and scoring protocol must also match.

#### Alternatives rejected
- **Hugging Face loader with a pinned old `datasets`:** fragile dependency, breaks clean-checkout reproducibility.
- **Block-interleaved split:** less distribution shift, but non-standard and no longer
  matches the dataset's published split.
- **nanoGPT's 90/10 split:** has no test split, which D1's selection/reporting rule requires.

**Depends on:** D1 (tokenizer roles for validation/test).

### D3. Training windows and evaluation protocol

**Status:** Accepted · 2026-10-10

**Decision:** Epoch-based training over shuffled blocks with randomly shifted boundaries
and complete target coverage; held-out loss measured by sliding-window evaluation with a
**fixed stride of 32 tokens** everywhere; a train-subset curve measured under the same
protocol; and a matched-target context diagnostic.

Terminology: a split of N tokens has **eligible targets** at positions 1…N−1. Token 0 is
context-only. Input position t in a window sees **t+1 tokens** (itself and everything before
it), so context ranges from 1 to T.

#### A. Training sampling: shifted, shuffled blocks with complete coverage
- Each epoch, the eligible target positions are partitioned into consecutive blocks of T
  targets, with block boundaries shifted by a random offset in 0…T−1 (seeded generator).
- A full block uses **T+1 stream tokens**: inputs `x = s[i : i+T]`, targets
  `y = s[i+1 : i+T+1]`. Each block's input includes the token immediately before its first
  target, so there are no gaps between blocks.
- The short **prefix and suffix blocks are kept**, not dropped and not wrapped around.
  They are **right-padded**; padded targets use `ignore_index` so they contribute nothing to
  the loss. Right-padding needs no attention mask: under the causal mask, real positions
  never attend to later pad positions.
- Blocks are shuffled each epoch. **Every eligible target is trained on exactly once per
  epoch.** How blocks are grouped into updates: **D24a** (full blocks shuffled; the short
  boundary blocks go into the epoch's final update).
- Exposure is reported as epochs, steps, and **non-pad targets seen**, as D1 requires.
- Why not random offsets per batch (nanoGPT style): total sampled targets can be counted
  exactly there too, but coverage is uneven. Some text is seen more often than other text
  by chance. The reason for this design is uniform once-per-epoch coverage.

#### B. Held-out evaluation: sliding windows, fixed stride 32
- Window length = the model's context length T. Windows advance by **s = 32 tokens**. The
  first window scores all of its targets; every later window scores only its **last 32
  targets**. The final partial block is included, and scores **only its new targets**.
  Windows are defined by absolute target indices (**D24b**).
- **Every eligible target is scored exactly once.** The first targets in a split use
  whatever earlier context exists inside the split, and never context from another split.
- **Loss = total negative log-likelihood ÷ actual number of scored targets.** It is not an
  unweighted average of per-window losses.
- **BPC** = total NLL ÷ (ln 2 × number of characters represented by the scored targets).
- **Why a fixed stride rather than T/2:** with T/2, increasing T changes both the maximum
  context and how much context is discarded between scored targets. A fixed stride keeps
  the second factor constant across the context-length ablation:

  | Context T | Stride | Context for scored targets (after the first window) |
  |---:|---:|---:|
  | 64 | 32 | 33–64 tokens |
  | 128 | 32 | 97–128 tokens |
  | 256 | 32 | 225–256 tokens |

- Cost is small: at T=256, about 0.45M token-positions per validation pass.
- Character vs BPE comparison (D1): both models score **exactly the same raw text**, from
  character offset *b* to the end of the split, where *b* is the end of the first BPE token
  in the split. **Amended by D24b:** *b* is frozen **before the grid**, and **every**
  reported held-out number (selection included) starts at *b*, not just the BPE comparison.
- The **same protocol** is used for in-training validation, ablations, final validation
  and the final test report.
- Optional robustness check: report the final model once at stride 1 as well, so the report
  states how sensitive the number is to stride.
- Sliding evaluation **reduces** context-truncation artifacts. It does not remove the
  low-context predictions at the start of a split, and scores still depend on the protocol.
  The report states this.

#### C. Train-subset curve for Figure 6
- At each evaluation step, the model is also scored on a **fixed training subset**: 8
  contiguous excerpts of about 7k characters each (about 56k in total, matching validation),
  evenly spaced across the training split. The exact offsets are recorded in the data manifest.
- Each excerpt is evaluated **separately** with protocol B in `eval()` mode (dropout off).
  Excerpts are never concatenated, so no artificial transitions are created.
- The plotted curve is labelled **"train subset, evaluation mode"**. The running minibatch
  loss is logged separately and plotted as its own curve.
- Caveats for the report: the 8 excerpts have 8 low-context starts against validation's
  one, a slight upward bias. The train/validation gap can also reflect differences in
  passage difficulty and distribution (D2), not only overfitting.

#### D. Context diagnostics
- **Primary, matched targets:** a fixed set of about 2,000 validation target positions,
  sampled once with a seed. Each target is scored with prefix lengths
  k ∈ {1, 2, 4, …, T}. Because the targets are identical at every k, the curve isolates the
  effect of extra context.
- **Secondary, per-position loss:** average loss at each input position over
  non-overlapping windows. It is **descriptive only**: each position covers different
  target tokens, so it shows loss associated with position, not the isolated benefit of
  each extra character.

#### Alternatives rejected
- **Fixed non-overlapping windows:** the same alignment every epoch.
- **Random shift that keeps only full windows:** drops a prefix and suffix each epoch.
- **Wrap-around:** creates an artificial end-to-start transition.
- **Non-overlapping evaluation:** inflated loss from low-context positions, and a bias that
  depends on T.
- **Stride T/2:** changes discarded context together with T.
- **Running minibatch loss as the "train" curve:** dropout on and a different protocol, so
  not comparable to validation.

**Depends on:** D1 (BPC definition, tokenizer comparison), D2 (splits, manifest).

### D4. Normalization placement and type

**Status:** Accepted · 2026-10-10

**Decision:** Pre-LN Transformer blocks using `nn.LayerNorm(d_model)` with learned scale and
bias and `eps=1e-5`, plus a final LayerNorm after the block stack, before the LM head.

#### Architecture
```python
# Each Transformer block: two separate LayerNorm instances
x = x + attention(ln_attention(x))
x = x + ffn(ln_ffn(x))

# After all Transformer blocks
logits = lm_head(ln_final(x))
```
- Normalization placement follows GPT-2 (Radford et al., 2019, §2.3): LayerNorm at the input
  of each sub-block, and an extra LayerNorm after the final block.
- Every LayerNorm instance is separate, with its own parameters. Scale γ starts at 1 and
  bias β at 0 (PyTorch defaults).

#### Normalization axis
- LayerNorm normalizes over the **`d_model` features of each token independently**:
  `nn.LayerNorm(d_model)` applied to a `(B, T, d_model)` tensor.
- It must **never** normalize over the sequence dimension. That would mix statistics across
  positions and leak future-token information into earlier positions. The end-to-end
  causal perturbation test would catch this kind of leak.
- Formula for the report: `y = γ · (x − μ) / sqrt(σ² + ε) + β`, where μ and σ² are the mean
  and (biased) variance over the `d_model` features of a single token.

#### Rationale (stated precisely)
- **Pre-LN improves optimization conditions; it does not guarantee stable training.** Xiong
  et al. (2020) show better-behaved gradients at initialization and less dependence on
  learning-rate warmup than Post-LN. The residual stack gives gradients an identity path to
  earlier layers. Learning rate, initialization and residual scaling still matter, so a
  **short warmup is kept** as the initial training choice. Its length is decided with the
  optimizer settings.
- **The final LayerNorm controls the scale of the representation given to the output
  head.** In Pre-LN the residual stream is not normalized inside the stack, and its
  magnitude tends to grow with depth. We include the final LayerNorm in our architecture;
  we do not claim that every Pre-LN model mathematically requires it. Gradients from the
  loss still pass through this final normalization before reaching the residual stack.
- Pre-LN keeps the depth ablation (2/4/6 layers) focused on depth rather than on how hard
  each depth is to optimize.

#### Implementation
- PyTorch's `nn.LayerNorm` is used; the handout explicitly allows it. The report explains the
  formula and the normalization axis. A custom reimplementation would add little.
- LayerNorm parameters are excluded from weight decay. This is confirmed with the optimizer
  settings.

#### Alternatives rejected
- **Post-LN** (`x = LN(x + Sublayer(x))`): more sensitive to warmup and depth at
  initialization. Its final-quality difference on this model is **unknown without
  testing**; it is not assumed small.
- **RMSNorm:** a valid pre-norm option, but its **benefit here is unestablished**, and it
  would need explicit justification under the handout for no demonstrated gain.

#### Open
- Whether to run a Pre-LN vs Post-LN experiment is decided with the experiment budget.
  (**Resolved in D16:** optional extra 2, depth 6, single seed, illustrative.)

**Depends on:** none. **Related:** D3 (causal-leak testing), the optimizer settings
(warmup, weight-decay groups).

### D5. Positional encoding and token-embedding scale

**Status:** Accepted · 2026-10-10

**Decision:** Token embeddings initialized with std `1/√d_model` and multiplied by `√d_model`
in the forward pass, plus our own sinusoidal positional encoding (interleaved sin/cos,
base 10000). The output head is **untied** for the baseline. Weight tying was later closed
(**not used**) in D8.

#### Input representation
```python
# E: nn.Embedding(vocab_size, d_model), initialized N(0, 1/d_model)
h0 = math.sqrt(d_model) * E(token_ids) + PE[0:T]   # PE: (T, d_model), positions 0..T-1
```
- **Scale matching at initialization:** the scaled token vector has a typical norm of about
  `√d`, and each sinusoidal position vector has an exact norm of `√(d/2)` (even d). Their
  ratio is about √2 at **every width**, so neither term swamps the other.
  - For comparison, GPT-2-style std 0.02 embeddings with raw sinusoids would give a ratio of
    about 1:35 at d = 256 (0.32 vs 11.3). Position would dominate token identity at the start.
- **Attribution:** Vaswani et al. (2017, §§3.4–3.5) multiply the embedding weights by
  `√d_model` and add sinusoidal encodings. The paper does **not** specify the Gaussian
  initialization. `std = 1/√d_model` is **our choice**, made to match scales.

#### Sinusoidal encoding implementation
- `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(pos / 10000^(2i/d))`,
  built in **float32** using the exp/log form.
- `d_model` must be even, enforced by an assertion. This is a constraint of **our
  implementation**; sinusoidal encoding can be adapted to odd widths.
- Stored as a **non-persistent buffer** for positions `0…context_length−1`, rebuilt on load.
  The config records the PE method, base and context length so reconstruction is exact.
  This avoids buffer-shape errors when loading checkpoints. It does **not** imply that a
  checkpoint trained at one context length works equally well at another.
- **Local positions** `0…length−1` are used consistently in training, evaluation and
  cropped generation. Every window starts at position 0.
- **Mixed precision:** the PE buffer stays float32 and is explicitly cast to the embedding
  dtype at the addition, so autocast never silently changes it.

#### Output head: untied baseline (tying closed in D8)
- The baseline LM head is an **independently initialized linear layer**, not tied to E.
- **Why tying is not assumed safe.** With tied weights and this input scaling, Pre-LN's
  identity path keeps a large component of the current token's own embedding in the final
  hidden state. Its dot product with that same embedding strongly favors predicting the
  current token again.
  - Verified in a simplified initialization check (d = 256, vocab 65, the embeddings and PE
    above, final LayerNorm, residual branches set to zero, 4,096 samples):
    - current-token logit averaged **13.66**, while other logits had std **1.02**
    - mean **P(repeat current token) = 0.9998**
  - Analytically: LayerNorm keeps a component of about `√d / √1.5 ≈ 13` along `E[x]`, whose
    norm is about 1.
  - This is a diagnostic, not a result from the final Transformer, but it rules out a
    blanket "tied logits are well-scaled" justification.
- *Resolved in D8:* tying is not used. This analysis is the reason why.

#### Evidence for the report (Figure 4 and text)
- PE heatmap and position-similarity (dot-product) plot.
- Norms of `√d·E[x]` and `PE[pos]`, measured **before dropout**, at initialization and after
  training. Report **averages across tokens and positions**, not a single selected
  embedding.
- Equal norms are evidence about initialization scale only. They do **not** show that token
  and position information have equal influence on predictions.

#### Alternatives rejected
- **GPT-2 initialization (std 0.02) with raw sinusoids:** position dominates token identity
  by about 35× at initialization.
- **Learned positions as the main method:** the handout requires sinusoidal; learned
  positions remain available as an experiment.
- **RoPE:** extra credit only. It changes the attention code rather than the input, so it is
  left to the experiment-budget decision.

**Depends on:** D4 (Pre-LN identity path and final LayerNorm, used in the tying analysis).
**Open:** weight tying and output-head initialization (**resolved in D8**: untied, no tying);
learned-position and RoPE experiments (**resolved in D16**: frozen vs trainable
sinusoid-initialized positions as optional extra 1; RoPE skipped).

### D6. Attention module design

**Status:** Accepted · 2026-10-10

**Decision:** Hand-written scaled dot-product attention in one shared function, used by both
the multi-head module and the hand-calculation test. Separate biased Q/K/V projections,
explicit head split and merge with `1/√d_head` scaling, a boolean causal mask, attention
numerics in float32 with autocast disabled, and PyTorch's SDPA math backend as a test-only
reference.

#### 6a. One core function, reused everywhere
- `scaled_dot_product_attention(q, k, v, mask=None, dropout=None) -> (output, probs)` is
  called by the multi-head module **and** by the hand-calculation test. The code that
  reproduces the hand-calculated numbers is exactly the code the model runs.

#### 6b. Separate Q/K/V projections
- Three `nn.Linear(d_model, d_model)` layers, one each for Q, K and V. They map one-to-one
  onto the hand calculation and the report's equations.
- A fused `nn.Linear(d_model, 3·d_model)` is mathematically identical. Any speed difference
  is **measured, not assumed**, during the training-configuration benchmark.

#### 6c. Projection biases
- `bias=True` on the Q, K, V and output projections (GPT-2 convention).
- Report point: in standard dot-product attention the **K bias has no effect on the output**.
  It adds `q·b_k` to every score in a query's row, a per-row constant that softmax ignores.
  The Q and V biases do affect the output. This argument must be revisited if a
  transformation such as RoPE is added, because the bias would then be rotated with the keys.

#### 6d. Explicit head split and merge
```
(B,T,d) → view (B,T,h,d_h) → transpose (B,h,T,d_h)       # split
scores (B,h,T,T) → probs (B,h,T,T) → out (B,h,T,d_h)     # per-head attention
transpose (B,T,h,d_h) → contiguous().view (B,T,d) → W_O  # merge + output projection
```
- `assert d_model % n_heads == 0`; `d_h = d_model / n_heads`.
- Scores are scaled by **`1/√d_h`**, not `1/√d_model`. The heads ablation changes `d_h`,
  so using `d_model` would be a hidden bug.

#### 6e. Masking and numerical precision
- **Causal mask:** a boolean lower-triangular buffer (`True` = allowed) over
  `context_length × context_length`, non-persistent, sliced to `[:T, :T]`.
  `masked_fill(~mask, -inf)` is applied **before** softmax, so masked probabilities are exactly 0.
- **Padding contract (D3's right-padded boundary blocks).** **In causal mode**, the causal mask alone
  is sufficient (bidirectional `leak`/`mlm` runs need a key-padding mask: **D24c**), and no key-padding mask or special pad token is needed, provided that:
  - real tokens come before all padding, and no example is empty;
  - padding positions use a valid vocabulary ID as input;
  - padded targets are excluded from the loss (`ignore_index`);
  - real queries cannot see padding to their right (causal mask); padded-query outputs are
    discarded through the loss.

  The guarantee that every row has a visible key (the diagonal) holds under the causal mask
  alone. It does not automatically hold once other masks are combined with it.
- **Fully masked rows:** if the standalone function receives a mask with any row that has
  no allowed key, it **raises an error** instead of silently returning NaNs.
- **Precision policy (if mixed precision is used):** inside the attention function,
  1. promote Q and K to float32;
  2. compute scores, scaling, masking and softmax in an **autocast-disabled region**
     (`torch.autocast(..., enabled=False)`);
  3. cast the probabilities to V's dtype for the multiplication with V.

  Casting scores to float32 *after* `Q @ Kᵀ` cannot repair an fp16 overflow that already
  happened, and calling `.float()` inside an active autocast region does not guarantee an
  fp32 matrix multiply. This policy costs speed, so it is benchmarked when the training
  configuration is chosen.
- **Note for the training-configuration decision:** at about 3M parameters, training costs
  roughly 6·N FLOPs per token (about 2×10¹³ per epoch of 1M tokens), so **plain fp32
  training may be fast enough** that AMP is unnecessary. AMP is adopted only if a benchmark
  shows a meaningful gain.

#### 6f. Manual attention in the model; SDPA as a test-only reference
- The model always uses our attention function. There is no switchable fused path, so the
  deployed model runs only code we wrote and can explain.
- `F.scaled_dot_product_attention` is a dispatcher with several backends; it is not the same
  thing as FlashAttention. Tests use it with the **math backend selected and dropout
  disabled** as a numerical reference, comparing **outputs and Q/K/V gradients** within
  tolerance. Its boolean mask uses the same convention as ours (`True` = allowed).
- **Memory:** one fp32 attention-probability tensor at batch 64, 4 heads, T = 256 is
  64 MiB (about 67 MB). Training also keeps probabilities, other activations and backward
  state for every layer, and more heads means more attention matrices even when the
  parameter count is unchanged. Actual memory use is **measured** (D3 / context ablation),
  not assumed.

#### 6g. Introspection
- The function returns the **pre-dropout softmax probabilities** for interpretation.
  Attention dropout, if used, is applied to a **separate tensor** that multiplies V, so the
  returned probabilities always sum to 1 along each row.
- `return_attn=False` by default. When enabled, the model returns weights only for the
  **requested layers and heads**, detached and copied for plotting. Figure 7 is generated
  in `eval()` mode.
- The shape trace (Figure 3) goes through an optional callback or logger, not `print`s.
- With introspection disabled, attention tensors are never stored on module attributes or
  collected across layers. The normal autograd graph keeps only what backward needs.

#### Required tests
1. **Hand calculation:** the function reproduces the hand-calculated output **and** the
   masked attention probabilities from the report.
2. **Reference agreement:** output and Q/K/V gradients match SDPA (math backend, dropout
   off), with and without the causal mask.
3. **Causal invariance (full model, `eval()`):** changing tokens after position t leaves
   logits at positions ≤ t unchanged.
4. **Right-padding invariance:** appending padding leaves logits at real positions unchanged.
5. **Edge cases:** single-token input (T = 1) works; a mask with a fully masked row raises an error.
6. **Shapes:** split/merge shapes at every step, for several `(n_heads, d_model)` pairs;
   `d_model % n_heads != 0` is rejected.

#### Alternatives rejected
- **Fused QKV projection:** equivalent, with a less direct mapping to the report.
  Reconsidered only if the benchmark shows a meaningful gain.
- **`F.scaled_dot_product_attention` or `nn.MultiheadAttention` inside the model:** hides
  the operations the assignment assesses.
- **No projection biases:** valid, but differs from the cited GPT-2 convention for no
  measured benefit.

**Depends on:** D3 (padding scheme), D5 (RoPE note).
**Resolved later:** attention-dropout rate (D9/D10 grid); AMP on/off (D12c, D24f).

### D7. Feed-forward network design

**Status:** Accepted · 2026-10-10

**Decision:** A two-layer, position-wise MLP with biases and exact GELU, hidden width
`ffn_dim = 4 × d_model` by default, and dropout at its output before the residual addition.

#### Sublayer (within D4's Pre-LN block)
```python
x = x + dropout(linear_out(gelu(linear_in(ln_ffn(x)))))
# linear_in:  nn.Linear(d_model, ffn_dim, bias=True)
# gelu:       nn.GELU(approximate="none")   # exact erf form, PyTorch's default
# linear_out: nn.Linear(ffn_dim, d_model, bias=True)
```
- Applied to each token independently: attention mixes information **across** positions,
  and the FFN transforms each position's vector **within** itself.
- This satisfies the handout's requirement of "two linear transformations with a
  nonlinearity", with GELU as recommended.

#### Width
- If `ffn_dim` is not specified, it is set to `4 * d_model` at config-resolution time.
  Otherwise any positive integer is accepted. It does not need to be divisible by the head count.
- The **resolved** value is saved in the checkpoint config.
- The 4× default follows the original Transformer (2048 for d = 512) and GPT-2.

#### Parameter count (for the report)
- `N_FFN = 2·d·f + f + d`, with `f = ffn_dim`. At `f = 4d`: `8d² + 5d`.
- For comparison, attention has `4d² + 4d` (four biased `d × d` projections), so the FFN
  holds about **two-thirds** of each block's parameters.

#### Rationale (stated precisely)
- **GELU vs ReLU:** GELU is a smooth, input-dependent gate (`x · Φ(x)`) that keeps some
  negative-valued signal and has nonzero gradient for negative inputs. ReLU can produce
  "dead" units, meaning units that become **persistently inactive** across inputs and stop
  learning. That is different from a single negative activation.
- **Exact vs tanh-approximate GELU:** their pointwise difference is small, but that does
  **not** prove identical training outcomes. Exact GELU is chosen as the true definition
  and PyTorch's default. No activation experiment is planned.
- **Dropout position:** at the FFN output, before the residual addition (GPT-2 placement).
  The rate is set in the dropout decision.

#### Alternatives rejected
- **SwiGLU:** three projections with a gated activation. It does not match the handout's
  two-projection GELU design, and its benefit at this scale is unestablished.
- **ReLU:** simpler, but not the handout's recommendation, and prone to dead units.
- **Tanh-approximate GELU:** a speed approximation with no need for it here.

**Depends on:** D4 (Pre-LN sublayer structure), D6 (bias convention).
**Resolved later:** dropout rate (D9/D10 grid).

### D8. Weight initialization and output head

**Status:** Accepted · 2026-10-10

**Decision:** GPT-2-style initialization with depth-aware scaling of the residual output
projections, and an **untied, bias-free LM head** for both the character and BPE models.
Weight tying is **closed**: not used and not planned. This resolves D5's open item.

#### Initialization recipe
| Parameters | Initialization |
|---|---|
| Token embeddings | Normal, std `1/√d_model` (D5; multiplied by `√d_model` in forward) |
| Q, K, V and FFN input weights | Normal, std `0.02` |
| Attention output (`W_O`) and FFN output (`linear_out`) weights | Normal, std `0.02/√(2L)` |
| Untied LM-head weights | Normal, std `0.02` (no bias) |
| Linear biases | Zero |
| LayerNorm scale / bias | One / zero |

- **Residual scaling:** each block adds **two** residual branches (attention and FFN), so an
  L-layer model has `N = 2L` residual contributions. Following GPT-2 (Radford et al., 2019,
  §2.3), their output projections are scaled by `1/√N`.
- **Implementation order:** the general linear-layer initialization must **not overwrite**
  D5's embedding initialization or the specially scaled residual projections. Apply the
  general rule first and the special cases last, or initialize by parameter role.
  An automated test checks the result.

#### Output head
- `nn.Linear(d_model, vocab_size, bias=False)`, independently initialized, applied to the
  final LayerNorm output (D4).
- **Why untied:** it keeps input-embedding and output-head scales independent (D5 scales
  the input embedding by `√d`), and it keeps the character-vs-BPE comparison (D1)
  consistent. D5's diagnostic, a simplified setup with residual branches set to zero, showed
  a strong current-token bias under tying. That result **motivated** the choice. It is
  **not** a measured property of the full initialized Transformer and is not reported as one.
- **Cost of not tying:** 65 × 256 ≈ 17k parameters for the character model (about 0.5%);
  1,024 × 256 = 262k for the BPE model. Non-embedding and total counts are reported
  separately (D1).

#### Expected initial loss
- The head's input has roughly unit variance per feature after LayerNorm, so logits are
  approximately `N(0, 0.02²·d)`. Random nonzero logits are not exactly uniform, so:
  `E[loss] ≈ ln V + ½ · (0.02²·d) · (1 − 1/V)`.
- At d = 256, V = 65: **about 4.22 nats**, against the uniform baseline `ln 65 = 4.17`.
  A simulation gave 4.223 (formula 4.225).
- This is a **broad sanity check**, not a precise prediction. The test tolerance is ±0.3 nats.

#### Checks
**Automated tests**, run with dropout disabled and over several fixed initialization seeds:
1. Each parameter group has the intended initialization: std per role (within statistical
   tolerance), zero biases, LayerNorm one/zero, embeddings not overwritten.
2. Initial loss is finite and within ±0.3 nats of the expected baseline (about 4.22 at
   d = 256, V = 65).
3. No extreme current-token preference: mean P(current token repeated) at initialization is
   far below 1 (threshold fixed in the test, near the 1/V scale).

**Initialization report** (diagnostic evidence, not pass/fail), for depths 2, 4 and 6 with
dropout disabled:
- actual initial loss;
- mean current-token probability;
- residual-stream feature standard deviation after each layer.

If the residual-stream measurement is ever turned into a test, it gets an explicit
numerical threshold.

#### Rationale (stated precisely)
- Residual scaling **reduces** depth-dependent scale differences; it does not eliminate
  them, because branch outputs can be correlated and have different variances. The claim is
  that every depth uses the **same depth-aware initialization rule**, not that all depths
  start from identical conditions.
- A fixed std of 0.02 does not confound any planned ablation, because `d_model` is not varied.

#### Alternatives rejected
- **Weight tying:** closed (see above).
- **Zero-initialized head:** exactly uniform initial predictions, but no gradient reaches
  the network body on the first step. An unnecessary oddity to explain.
- **Head bias set to log unigram frequencies:** reproduces unigram probabilities exactly only
  when the other logits are zero, and its speed benefit is unmeasured. Omitted.
- **PyTorch default (Kaiming-uniform) initialization:** similar scale and valid, but gives
  no depth-aware residual scaling and no single cited scheme.

**Depends on:** D4 (final LayerNorm), D5 (embedding initialization and the tying
diagnostic), D6–D7 (which projections are residual outputs).

### D9. Dropout placement and rate policy

**Status:** Accepted · 2026-10-10 · The rate value comes from D10's joint grid.

**Decision:** Dropout at GPT-2's four locations, controlled by one shared rate, chosen by a
controlled sweep over {0.0, 0.1, 0.2} on the baseline configuration, then frozen for all
ablations.

#### 9a. Placement
Separate `nn.Dropout` modules at:
1. **Embedding output:** after `√d·E[x] + PE` (D5).
2. **Attention probabilities:** on the separate tensor that multiplies V. The returned
   probabilities stay pre-dropout (D6g).
3. **Attention output:** after `W_O`, before the residual addition.
4. **FFN output:** after `linear_out`, before the residual addition (D7).

Wording for the report: embedding dropout **does** act on the initial residual-stream
representation. **Inside each Transformer block**, dropout acts only on the residual
branches and never interrupts the identity skip connection (D4). Dropout is off in
`eval()` mode, which all evaluation (D3) and tests (D6, D8) use.

#### 9b. One shared rate
- One `dropout` config value sets all four modules by default.
- Optional per-location overrides (`dropout_embedding`, `dropout_attention`,
  `dropout_residual`) exist for experiments only.
- The **resolved** value of every rate is saved in the checkpoint config.

#### 9c. Rate selection procedure
1. **Candidates:** `{0.0, 0.1, 0.2}` on the baseline configuration.
2. **Budget:** each candidate is trained on the **intended baseline training budget**.
   Shorter runs may screen candidates but never settle the final rate, because low dropout
   learns faster at first, before overfitting appears.
3. **Matched conditions:** identical initialization seed, data order, training exposure,
   learning-rate schedule and evaluation frequency. D3's sampler uses its **own seeded
   generator**, separate from the RNG dropout uses, so changing the dropout rate cannot
   change the order of training windows.
4. **Comparison:** each run's **best validation checkpoint** within the common budget,
   matching the checkpoint-selection policy for the final model. Validation only; test is
   never used (D1).
5. **Close finalists:** confirmed with additional **paired seeds** (same seeds for each rate)
   before choosing. A small single-seed difference is not enough evidence.
6. **Freeze** the selected rate for all controlled ablations.

#### Interpretation (stated precisely)
- Dropout is described as **one regularization mechanism**, not "the main regularizer",
  until experiments show its effect.
- The **0.0 run is a control**, not guaranteed overfitting evidence. Other regularization
  (e.g. weight decay) may still apply, and a train/validation gap alone also includes the
  distribution differences discussed in D2–D3. Overfitting is shown by training loss still
  improving while validation loss gets worse, with dropout reducing that.
- **Ablation caveat:** with a frozen rate, the ablations measure architectural changes
  under a **shared regularization setting**, not each architecture's individually tuned
  performance. Deeper variants may be under-regularized and shallower ones over-regularized.

#### Alternatives rejected
- **Fixed rate by convention (e.g. 0.1):** acceptable under the handout, which asks for a
  **justified** configuration (Section 11), not a sweep. The team chose to provide evidence
  instead, because the best rate for this data-to-parameter ratio is uncertain.
- **Separate tuned rates per location:** too many tuning dimensions for the budget.
- **Re-tuning dropout for each ablation configuration:** changes two variables at once
  and multiplies the number of runs.

**Depends on:** D3 (sampler, evaluation protocol), D4 (residual structure), D6–D7 (module
locations). **Resolved later:** baseline (D11), budget (D12), checkpoint selection (D13);
the rate comes from D10's grid.

**Amended by D10:** the dropout sweep is run as the dropout axis of D10's joint
LR × dropout grid. It **replaces** a separate dropout-only sweep. All procedure rules in 9c
(budget, matched conditions, best-validation comparison, paired seeds, freezing) still apply.

### D10. Optimizer, weight decay, LR schedule and gradient clipping

**Status:** Accepted · 2026-10-10 · **Peak LR and dropout are pending the joint grid;
the warmup fraction is set before the grid.**

**Decision:** AdamW with β = (0.9, 0.99); weight decay 0.1 on Linear weight matrices only;
linear warmup then cosine decay to 0.1× peak; global gradient clipping at 1.0; peak LR and
dropout chosen together from a 3 × 3 grid.

#### 10a. Optimizer
- `torch.optim.AdamW(betas=(0.9, 0.99), eps=1e-8)`.
- β₂ = 0.99 averages squared gradients over about **100 updates** (about 1,000 for 0.999),
  so it adapts faster to the noisy gradients of small batches. nanoGPT's Shakespeare
  character config uses the same value. This is a **cited heuristic**, not proof it is optimal here.

#### 10b. Weight-decay parameter groups
| Group | Weight decay |
|---|---|
| Linear weight matrices (Q, K, V, O, FFN in/out, LM head) | 0.1 |
| Token embedding | 0 |
| All biases | 0 |
| LayerNorm scale and bias | 0 |
| Trainable position table (optional D16f extra only) | 0 (D24m) |

- **Why embeddings are excluded:** the forward pass is `√d·E[x] + PE` with PE fixed (D5).
  Excluding E removes **direct weight-decay pressure** on the token term relative to the
  position term. It does **not** preserve the initial balance, because gradient updates can
  still grow or shrink E. D5's norm measurements track what actually happens.
- **Decay strength across the grid:** AdamW's per-update shrinkage is proportional to
  **LR × weight decay**. Holding the coefficient at 0.1 across the LR grid therefore does
  **not** hold cumulative shrinkage constant. The report states that the **coefficient**
  is fixed, not that regularization strength is identical.
- A test checks that **every trainable parameter belongs to exactly one** optimizer group.

#### 10c. Learning-rate schedule
- Linear warmup from 0 to `peak_lr`, then cosine decay to `min_lr = 0.1 × peak_lr` at the
  end of the fixed training budget.
- **Warmup fraction:** one value in the 2–5% range, **chosen before the grid and then held
  fixed** for every run. The concrete value is recorded with the training-budget decision.
- The schedule is defined in **optimizer updates**, not gradient-accumulation microbatches.
- ~~If mixed precision skips an update, the scheduler does not advance.~~ **Superseded by
  D24f:** the schedule is indexed by **attempted** updates, so a fixed number of epochs
  always ends at `min_lr`. Skips are logged, and too many skips count as a numerical
  failure.
- The learning rate is logged at every update (handout requirement).

#### 10d. Gradient clipping
- Global gradient-norm clipping at **1.0**, applied once per optimizer update, in this order:
  1. loss normalized by the number of **valid (non-pad) targets** (D3's partial blocks);
  2. gradient accumulation finished;
  3. gradients unscaled, if `GradScaler` is in use (PyTorch requires unscaling before
     inspecting or clipping);
  4. global norm measured and logged (pre-clip), then clipped;
  5. optimizer update.
- Logged: pre-clip global norm per update and the fraction of updates clipped.
- **Interpretation:** frequent clipping is a reason to **investigate** (gradient scale,
  initialization, batch composition, loss normalization, the threshold itself), read
  together with loss behavior and update sizes. It is **not** an automatic signal to lower
  the LR. The LR shapes the training trajectory but does not scale the raw gradient being clipped.

#### 10e. Joint LR × dropout grid
- Grid: `peak_lr ∈ {3e-4, 1e-3, 2e-3}` × `dropout ∈ {0.0, 0.1, 0.2}` = **9 runs**, on the
  baseline configuration at the intended baseline budget. It **replaces** D9's separate
  dropout sweep, and D9's rules apply.
- **Before the grid:** benchmark representative training **and** evaluation throughput
  (D3's sliding-window evaluation, D6's float32 attention) to set the budget. Whether 9 runs
  fit in one Colab session is an **empirical question**. If they don't, the grid runs
  across **resumable sessions**; runs are not shortened to the point where the dropout
  comparison becomes uninformative.
- Finalists are confirmed with **paired seeds**, and the selected pair is **frozen** for
  all controlled ablations. **Made explicit in D24i:** the 2 best seed-0 pairs get seeds 1
  and 2, and the lower 3-seed mean wins, giving 13 runs.
- The selected pair is described as the **best tested configuration within this grid and
  budget**, not a universal optimum.
- **Ablation caveat:** one frozen LR across depths and context lengths means the ablations
  compare architectures under **shared optimization settings**. A deeper model might
  prefer a different LR.

#### Alternatives rejected
- **Sequential LR-then-dropout tuning:** fewer runs, but the result depends on the order.
- **Decaying embeddings (nanoGPT default):** adds direct pressure that shrinks the token
  term relative to fixed sinusoidal PE (D5).
- **β₂ = 0.999 (PyTorch default):** a longer averaging window, less suited to small-batch
  noise. Not tested, so stated as a heuristic choice.
- **Lowering the LR automatically when clipping is frequent:** not justified by
  clipping frequency alone.

**Depends on:** D3 (valid-target normalization, exposure accounting), D4 (warmup,
LayerNorm exclusion), D5 (embedding/PE balance), D6 (AMP precision policy), D9 (dropout
procedure). **Pending:** grid results. (Warmup fraction and budget were set in **D12**:
5%, 185 updates at the provisional 3,690-update budget.)

### D11. Baseline model configuration

**Status:** Accepted · 2026-10-10

**Decision:** Baseline = `d_model=256`, `n_layers=4`, `n_heads=4`, `context_length=128`,
`ffn_dim=1024`, giving **3,192,832 parameters** with the character vocabulary (V = 65). It is
the shared centre of all three ablations and the configuration for D10's LR × dropout grid.
**The shipped model is not decided here.**

#### Why this centre
- Depth 4, heads 4 and context 128 are the middle values of the handout's ablation grids
  (2/4/6, 2/4/8, 64/128/256), so the baseline is one point of every ablation and all three
  can be read against one reference.
- **Width 256** gives a broad per-head representation across the whole heads grid
  (`d_head` = 128 / 64 / 32 at 2 / 4 / 8 heads). That is sufficient justification. It is
  **not** claimed that width 128's 16-dimensional heads at h = 8 would be "starved"; that
  would need evidence.
- That the total is close to the handout's example output ("3.2M") is **incidental**, not a target.

#### Parameter counts (from D6–D8's formulas; V = 65)
Per block: `12d² + 13d` (attention `4d² + 4d`, FFN `8d² + 5d`, two LayerNorms `4d`).

| Configuration | Total parameters |
|---|---:|
| Width 128, depth 4 | 809,984 |
| **Width 256, depth 4 (baseline)** | **3,192,832** |
| Width 384, depth 4 | 7,148,544 |
| Width 256, depth 2 | 1,613,312 |
| Width 256, depth 6 | 4,772,352 |

Baseline breakdown:
- **Backbone parameters, excluding the token embedding and LM head:** 3,159,552
  (4 blocks × 789,760 + final LayerNorm 512). This label is used instead of the ambiguous
  "non-embedding".
- Token embedding: 16,640. Untied LM head: 16,640 (D8).
- The heads ablation does not change the parameter count (fixed `d_model`). The depth
  ablation does, and the count is reported next to each result.

#### Baseline vs shipped model
- The baseline is a **reference point for experiments**, not a commitment to ship it.
- A baseline run may be **reused** as a point in an ablation **only if** its configuration,
  seed, training budget and evaluation protocol match that ablation exactly.
- Combinations suggested by separate ablations (e.g. 6 layers **and** context 256) are
  **new candidates** that need their own validation. Improvements from separate ablations
  need not combine additively.
- The final model is chosen on **validation** results before any test result is examined
  (D1). Whether a final configuration needs its own LR/dropout check is decided in the
  final-model decision.

#### Deployment note (unmeasured)
- Without a KV cache, generating one token at context 128 costs roughly **0.87 GFLOP**
  (approximate, excluding several overheads).
- This does **not** establish serving latency. Autoregressive execution, CPU allocation,
  memory access and threading all matter. The model is small in weight storage (about 12.8 MB
  in fp32), but acceptable latency on the deployment platform requires a **benchmark**.
  KV caching is decided with generation.

#### Alternatives rejected
- **Width 128:** a valid, cheaper option, but narrower per-head representation across the heads grid.
- **Width 384:** more capacity and much more dense-layer compute per run, across 15+ runs.
  Whether it improves validation loss on this corpus is **unknown**, so it is not assumed better.

**Depends on:** D6–D8 (parameter formulas), D10 (grid runs on this configuration).
**Resolved later:** final shipped-model decision (D17, D24e); latency benchmark (D18, D24h).

### D12. Batch size, training budget, precision, evaluation cadence and seeds

**Status:** Accepted · 2026-10-10 · **Provisional budget plus a pilot protocol. Not a
promise about runtime or convergence.**

**Decision:** 8,192 valid targets per optimizer update in every run; a common budget of
30 exact epochs (extendable once, before the grid, after a pilot); fp32 training unless a
controlled AMP benchmark justifies switching; 5% warmup; evaluation at the start and once
per epoch; seeds 0/1/2 with dedicated generators for initialization, data sampling and
generation sampling.

#### 12a. Targets per update
- Each optimizer update uses **8,192 valid prediction targets**, counting non-pad targets
  rather than padded tensor positions. D3's short boundary blocks make `batch_size × T`
  differ from the real target count.
- Nominal shapes: baseline 64 × 128; context ablation T = 64 → 128 × 64, T = 256 → 32 × 256.
- Gradient accumulation is allowed when needed for **memory or target accounting**. The
  **last update of each epoch may be smaller** (the remainder).
- Accumulated loss is normalized by the **total valid-target count** of the update (D10d).
- **Packing contract: D24a.** Exactly 8,192 targets per update applies to causal runs with
  T dividing 8,192. The final update of an epoch takes the remainder plus the short
  boundary blocks. Non-dividing T and the MLM run use nominal budgets (D24a, D24c).
- Wording for the report: equal targets per update controls **batch size measured in
  targets**. It does **not** exactly equalize gradient noise, because longer windows contain
  more strongly correlated predictions and fewer independent sequences.

#### 12b. Training budget
- **Common experimental budget: 30 exact epochs** (D3: each eligible target once per epoch).
- The schedule is computed from actual counts, not hard-coded:
  `updates_per_epoch = ceil(N_eligible_targets / 8192)`.
  - Character baseline: 1,003,853 eligible targets → **123 updates/epoch** (including the
    remainder update) → **3,690 updates** over 30 epochs.
- **BPE runs:** the same epoch count gives approximately the **same raw-text exposure**,
  but fewer token targets and fewer optimizer updates. It is a **data-exposure-matched**
  comparison, not equal-compute or equal-update. The report gives raw-text exposure, token
  targets and optimizer updates for every run (D1).
- **Pilot (before the grid):** one baseline run (LR 1e-3, dropout 0.1) measures feasibility
  (wall-clock time for training and evaluation, peak memory) and whether the baseline is
  still clearly improving at 30 epochs. Any extension (e.g. to 45 epochs) is decided
  **before** the grid starts. The budget is then frozen for the grid and all controlled
  ablations.
- The pilot covers **one** LR/dropout pair, so it cannot show that every grid candidate has
  converged. 30 epochs is described as a **common experimental budget**, not a
  convergence guarantee.
- Runtime is **unmeasured**. The training FLOP estimate (about 6 × 3.2M × 30M) excludes
  evaluation and does not predict achieved hardware utilization.

#### 12c. Training precision
- **fp32 is the reference.** The T4 (Turing) has no TF32, so fp32 is full fp32.
- **AMP comparison (during the pilot phase)**, run under identical conditions:
  - same starting weights, batches and effective target counts;
  - warmed-up, synchronized timings that include the same evaluation and sampling work;
  - peak memory recorded; losses and gradients finite;
  - enough training to catch obvious instability (one timed epoch alone is not enough).
- **AMP is adopted only if** it is at least **1.5× faster end-to-end** **and** the
  numerical checks pass, using D6e's precision policy. D6's float32 attention scores limit
  how much of the T4's mixed-precision throughput can be used.
- The chosen training precision is **frozen** for the grid and controlled ablations.
- **Evaluation precision is fixed at fp32** whatever the training precision, so reported
  losses are comparable.

#### 12d. Warmup
- **5% of planned optimizer updates, rounded up:** 3,690 updates → **185 warmup updates**.
  The count is recomputed from the actual schedule if the budget changes.
- 5% is the top of D10's range because the grid includes the most aggressive LR (2e-3).
- If AMP is adopted, D24f's skipped-update rule applies (schedule by attempted updates).

#### 12e. Evaluation cadence
- **Initial evaluation** before training (step 0), then **once per epoch**.
- Each evaluation logs: D3 validation loss and train-subset loss; current LR; gradient-norm
  statistics (D10d); valid targets and raw characters seen; optimizer updates; wall-clock time.
- **One fixed-prompt sample per evaluation**, with fixed decoding settings, a fixed output
  length and a fixed sampling seed, so the change in samples over training is visible
  (handout M4).
- Evaluation cost at stride 32, per roughly 56k-token split, excluding generation:

  | Context | Evaluated input token-positions |
  |---:|---:|
  | 64 | 0.11M |
  | 128 | 0.22M |
  | 256 | 0.45M |

  Both validation and the training subset are evaluated, so the overhead is measured
  for both in the pilot.

#### 12f. Seeds and random-number generators
- Grid and ablation runs use **seed 0**. Paired-seed confirmations (D1, D9, D10) use
  **seeds 1 and 2**.
- Derived from the run seed:
  - **Initialization:** deterministic, from the seeded default generator before training starts.
  - **Data sampler:** a dedicated `torch.Generator` (D3, D9.3).
  - **Generation sampling:** a dedicated `torch.Generator` passed to sampling calls.
  - **Dropout:** standard `nn.Dropout` has **no generator argument**, so it uses the
    default CPU/CUDA RNG, which is seeded at start-up.
- Evaluation (`eval()` mode, no dropout) and sample generation (dedicated generator) must
  **not consume or change** the default RNG state that training dropout uses.
- Checkpoints save **all** relevant RNG states (default CPU, CUDA, sampler, generation)
  so resumed runs continue identically. The details are part of the checkpointing decision.

#### Alternatives rejected
- **Fixed batch size in sequences across the context ablation:** changes targets per
  update together with context.
- **Budget in steps:** cannot guarantee exact once-per-epoch coverage (D3).
- **AMP by default:** adds GradScaler, skipped updates and autocast handling, with no
  measured benefit yet.

**Depends on:** D3 (sampler, valid targets), D6 (precision policy), D9–D10 (grid, warmup
rule, clipping order), D11 (baseline). **Pending:** pilot results (budget extension, AMP
decision, runtime).

### D13. Checkpointing, resume, selection and distribution

**Status:** Accepted · 2026-10-10

**Decision:** Each run keeps a resumable `last.pt` and a selection `best.pt`. Checkpoints are
staged locally and copied to Drive with checksum verification. Loading uses restricted
`torch.load` plus checksum and schema validation. The released inference artifact is a
`model.pt` exported from the selected best checkpoint and hosted as a pinned GitHub Release
asset. An exact-resume test on CPU in a fresh subprocess backs all of this.

#### 13a. Checkpoint roles and contents
**`last.pt`: resume.** Saved **only at completed optimizer-update boundaries**, after
gradient accumulation has finished, so partially accumulated gradients never need saving.
Contents:
- model state; optimizer state; **GradScaler state** if D12 adopts AMP;
- **scheduler definition and the frozen budget** (D10c, D12b), not just a counter;
- counters: successful optimizer updates, attempted updates, epoch; actual exposure in
  valid targets and raw characters (D3, D12);
- **sampler state, saved explicitly:** the current epoch's window boundaries, its
  shuffled order, the consumed cursor (next unconsumed batch), and the sampler generator's
  state for **future** epochs. Saving the order rather than regenerating it avoids the
  trap that a generator state captured after shuffling cannot reproduce the same shuffle;
- all RNG states from D12f (default CPU, CUDA, sampler, generation sampling), stored as
  tensors or plain types;
- the best checkpoint's epoch, update number and validation metric;
- the resolved config, the tokenizer (character list or BPE merges), the data-manifest
  hash, and the git commit with a dirty flag.

**`best.pt`: selection** (D9.4). The model state at the best per-epoch validation loss.
It is updated only on **strict improvement** of the **unrounded, finite** validation loss,
which keeps the earliest of exactly tied epochs.

**Save cadence vs capability:**
- The code supports **explicit mid-epoch saves** at any completed update.
- By default saves happen at epoch ends, so recovery goes back to the last epoch boundary.
  **Periodic mid-epoch autosaves** are added if the pilot (D12b) shows that losing an
  epoch would be costly.

#### 13b. Storage: local staging plus verified copy to Drive
`os.replace` is atomic only on a filesystem that supports it. It does not guarantee that a
mounted Google Drive has durably saved the replacement, and Colab documents I/O and
interruption risks for mounted Drive. Procedure:
1. Write the checkpoint to **local disk** with temporary-file-then-replace, and validate it
   by reloading.
2. Copy it to Drive under a **unique filename containing the update number**.
3. **Verify the checksum** of the Drive copy before recording it as the latest valid
   checkpoint, in a small run-state file.
4. **Keep the previous verified checkpoint** until the new one is confirmed.

- `last.pt` remains the **logical** "latest" role. On recovery, if the newest upload is
  incomplete or fails its checksum, the run falls back to the previous verified checkpoint.
- Logs are **buffered locally** and copied to Drive together with checkpoints, not written to
  Drive one entry at a time.
- **Retention:** a finished run's resume checkpoints are deleted only **after** its results,
  `best.pt` and logs have been verified on Drive, not just when the training loop exits.
  The final model keeps everything.

#### 13c. Format and safe loading
- `torch.save` of a dict containing only tensors and plain Python types.
- Loaded explicitly with `torch.load(path, map_location="cpu", weights_only=True)`.
- `weights_only=True` is a **restricted unpickler, not a complete security guarantee**.
  PyTorch documents remaining denial-of-service and possible memory-corruption risks. The
  real trust boundary is the **SHA-256 checksum against the repository manifest**. The
  loader **never** falls back to `weights_only=False`.
- Validation on load: `schema_version`; tokenizer contents (character list or ordered BPE
  merges and IDs); vocabulary size; configuration keys and values; tensor shapes. Model
  weights load with `strict=True`. Any mismatch fails with a clear message.
- **Formal experiments** (grid, ablations, final model) run from a **clean committed
  revision**. A dirty flag records the problem but does not preserve the code that produced
  a result.

#### 13d. Inference artifact and hosting
- `model.pt` is exported from the **selected `best.pt`**, not from the final in-memory
  model. It contains weights, resolved config, tokenizer, and provenance: selected
  epoch/update, metrics with the **evaluation-protocol identity** (D3: stride, split,
  scoring rule), git commit and training-config hash. No optimizer state; about 12.8 MB
  for the baseline.
- Hosted as a **GitHub Release asset** with a **pinned release tag and asset** and a
  SHA-256 recorded in the repo.
- One shared downloader, **`src/artifacts.py`** (`python -m src.artifacts`; it replaces the
  earlier `scripts/download_checkpoint.py`, see D24g), used by the Colab notebook, CLI, API
  and Docker build, checks the checksum.
- An optional Google Drive mirror must serve **identical bytes** and pass the same checksum.

#### 13e. Exact-resume test (automated, CPU)
- Tiny config, a few updates, run in a **fresh subprocess**. It uses **nonzero dropout**,
  shuffling, partial boundary windows, and a **mid-epoch interruption** whose continuation
  **crosses an epoch boundary**.
- Fixed software, fixed thread settings, deterministic operations. RNG states are
  restored **after** the model and optimizer are constructed and loaded.
- Compared **exactly** against an uninterrupted run: parameters, optimizer state, loss
  values, learning rate and counters. Wall-clock fields are excluded.
- Claim, stated narrowly: the test verifies **this tested execution path**. It does **not**
  verify bitwise reproducibility across different GPUs or software versions.

#### Alternatives rejected
- **Saving directly to Drive with `os.replace`:** no guarantee the file was durably saved.
- **Regenerating the shuffle from a generator state:** ambiguous and can break mid-epoch resume.
- **safetensors plus a JSON sidecar:** avoids pickle, but adds a dependency and a second file.
  The checksum against the manifest remains the trust boundary either way.
- **Committing checkpoints to git history:** goes against handout Section 8.

**Depends on:** D3 (sampler, exposure), D9 (best-validation selection), D10 (scheduler,
clipping), D12 (budget, AMP, RNG streams).

### D14. Generation (decoding) design

**Status:** Accepted · 2026-10-10

**Decision:** One shared, seeded `generate()` used by the CLI, API and training-time samples.
Greedy decoding at temperature 0; otherwise temperature sampling with optional top-k and
top-p, both off by default (temperature 0.8). The model input is a left-cropped window with
local positions, recomputed in full at every step (no KV cache). `max_new_tokens` plus
optional stop strings with exact boundary rules. Full reproducibility metadata on every output.

#### 14a. Shared function and state handling
- `generate()` sets `model.eval()` **and** runs under `torch.inference_mode()`. Inference
  mode alone does not turn off dropout. The model's previous training mode is **restored
  afterwards, including when an exception is raised**, so training-time samples (D12e)
  cannot leave training in evaluation mode.
- **Each API request gets its own generator**, seeded from the request's seed. Each
  fixed-seed training sample **re-seeds** its generator. A generator whose state has already
  advanced does not reproduce a sample just because its original seed matches.
- `generate()` is for **text samples only**. D3's loss/BPC evaluator scores the model's
  unmodified probabilities, with no temperature and no truncation.

#### 14b. Sampling pipeline (each step)
1. Take the logits at the last position.
2. **Temperature = 0:** greedy argmax. Sampling and both filters are skipped.
3. Otherwise: logits ÷ temperature → optional **top-k** → optional **top-p** → softmax →
   `torch.multinomial(probs, 1, generator=g)`.
- Filtering probabilities and the final softmax are computed in **float32**.
- **Top-p** keeps the smallest prefix of tokens (sorted by probability) whose cumulative
  probability reaches `top_p`, **including the token that crosses the threshold**.
- With both filters on, top-p works on the distribution **renormalized after top-k**.
- At least one candidate is always kept. Non-finite parameters or logits are rejected.
- **Validated ranges:** temperature ∈ [0, 2]; top_k ∈ [1, V] or off; top_p ∈ (0, 1] or off.
- **Defaults:** temperature **0.8** (the handout's baseline demo value); top-k and top-p
  **off**. How much top-k changes sampling depends on k and on the predicted
  probabilities (k = 1 makes it greedy), so no claim is made about its size. Whether
  truncation helps belongs to the optional decoding-strategy experiment.

#### 14c. Context and prompt handling
- The prompt is **encoded once**. Sampled token IDs are appended to that ID sequence, and
  only the **model's input** is cropped to the last `context_length` tokens, with positions
  restarting at 0 (D5). The growing BPE sequence is never decoded and re-tokenized.
- The original prompt text is kept separately for display.
- A prompt longer than the context is truncated from the left for the model input.
- An **empty prompt** becomes `"\n"`, because the model needs at least one input token and
  speeches and scenes in this corpus start after newlines. No BOS token is added (D1).
- Both the **prompt-truncation flag** and the **newline-fallback flag** are recorded.
- Characters outside the vocabulary are rejected before generation (D1).

#### 14d. No KV cache: full-window recomputation
- **Reason:** we recompute the full window every step to **keep exactly the generation
  semantics this model uses**. Each step's input is freshly recomputed from the current
  token-ID window, locally positioned (D5), and a conventional KV cache cannot reproduce that once the window
  starts sliding.
- How long a cache would help before the window fills **depends on the prompt length**. A
  prompt that already fills the context leaves essentially no such phase.
- **Changing the position scheme (e.g. RoPE) would not make cache eviction equivalent to
  recomputing a cropped window.** Cached states in upper layers still carry information from
  tokens that have since been evicted. Other caching schemes are possible, but each would
  have its own semantics and need its own equivalence tests. They are out of scope.
- Latency is measured by D11's benchmark.

#### 14e. Stopping
- `max_new_tokens` is required. The API cap is set in the API decision.
- Optional **stop strings**:
  - empty stop strings are rejected;
  - matching covers **only the generated continuation**, including matches that span
    several generated tokens;
  - stop strings already present in the prompt never trigger a stop;
  - if a BPE token completes a stop string and also contains trailing text, the visible
    output is **trimmed right after the matched stop string**, which stays in the output;
  - with several matches, the **earliest-ending** one wins, and ties go to the order of the
    stop-string list.
- `finish_reason` is `"stop_string"` (with the matched string) or `"max_new_tokens"`.
  D19 adds `"time_limit"` and `"cancelled"` for the API.

#### 14f. Output metadata (CLI print and API response)
- prompt; generated continuation; prompt-truncation flag; newline-fallback flag;
- requested `max_new_tokens`; stop strings; `finish_reason` and matched stop string;
- **sampled token count** and **visible continuation character count**, which differ
  after stop-string trimming and for BPE;
- model parameters, context length, tokenizer identity, temperature, top-k, top-p, seed;
- checkpoint ID (release tag and SHA-256 prefix);
- wall-clock generation time and characters per second, measured with the model **already
  loaded**, GPU work synchronized where relevant, and **prompt characters excluded**.
- Environment and precision details are kept in the run manifest. A seed alone is not a
  complete record for reproducing an output.

#### 14g. Byte-level decoding (BPE experiment)
- ASCII training data does **not** stop the BPE model from sampling invalid byte
  sequences: all 256 byte tokens remain possible outputs.
- Token bytes are **concatenated before decoding**, or decoded with an **incremental UTF-8
  decoder** that buffers incomplete sequences and is **flushed** when generation ends.
  Decoding each token on its own could corrupt characters split across tokens.
- `errors="replace"` is the display fallback, and **whether decoding errors occurred is
  recorded**, so invalid-byte generation shows up as a quality issue instead of being hidden.

#### Alternatives rejected
- **KV cache:** cannot reproduce the cropped-window semantics once the window slides.
- **Truncating sampling (top-k/top-p) by default:** its benefit is unmeasured here.
- **BOS token for empty prompts:** D1 has no special tokens; the newline fallback matches
  how text starts in this corpus.
- **Decoding BPE token by token:** corrupts characters split across tokens.

**Depends on:** D1 (vocabulary, unknown characters, BPE), D3 (evaluator kept separate),
D5 (local positions), D11 (latency benchmark), D12 (generation RNG, training samples).

### D15. Evaluation metrics, baselines, fixed prompts and sample assessment

**Status:** Accepted · 2026-10-10 · 15e ship rule tightened and locked with D17
(single pass, 6 seeds per prompt, consistent −0.5 tolerance, t-interval).

**Decision:** BPC is the primary metric, with character-normalized perplexity reported
alongside it. Uniform and add-k n-gram reference baselines are scored on identical targets.
A frozen set of 8 prompts produces 32 samples per model. Automatic diversity, overlap,
coverage and format diagnostics are reported with honest names and limits. A blind,
anchored human rubric has an exact ship rule. The test set is evaluated once, after
everything is frozen.

#### 15a. Loss metrics (D3 protocol, fp32)
- Loss in nats per token; **BPC** (primary); **character-normalized perplexity = 2^BPC**.
- For BPE, character-normalized perplexity comes from the likelihood of the **canonical
  token sequence**, normalized by scored characters. It is not a separately measured
  character-level prediction distribution.
- Per-token perplexity is reported only **within** one tokenizer, never across tokenizers (D1).
- Identical raw-text scoring boundaries and D3's actual scored-character denominator.

#### 15b. Reference baselines
- **Uniform:** `log₂(V_char)` bits per character, using the actual character vocabulary
  size (65 → about 6.02).
- **Add-k character n-gram baselines, n = 1…5:** counts are fitted on the **training split
  only**. The smoothing grid for k is **frozen before** any candidate is evaluated, and k is
  chosen on validation. They are scored on exactly the same targets and boundaries as the
  Transformer (D3).
- They are labelled **add-k reference baselines**, not optimized n-gram systems.
- **Interpretation limit:** beating an add-k 5-gram does **not** prove that the Transformer
  uses long-range context, since differences in smoothing, generalization and representation
  could also explain it. The evidence for context use is D3's **matched-target prefix
  diagnostic**.

#### 15c. Fixed prompt set
Stored in `configs/prompts.yaml` and **frozen before any candidate output is inspected for
model selection**.

| Group | Prompts |
|---|---|
| Speaker cue (in-domain) | `"ROMEO:\n"`, `"First Citizen:\n"`, `"KING RICHARD III:\n"` |
| Mid-line continuation | `"O, what a"`, `"My lord, I"` |
| Held-out continuation | first 100 characters of a fixed validation passage, frozen by **character offsets and a SHA-256 hash** |
| Empty | `""` (D14c newline fallback) |
| Out-of-domain | `"The future of artificial intelligence"` |

- **8 prompts → 32 samples per model:** for each prompt, 3 sampled at temperature 0.8
  (fixed seeds) plus 1 greedy, giving **24 sampled + 8 greedy**. The BPE ship comparison
  (15e) adds 3 more predetermined seeds per in-domain prompt.
- **Length:** generate until at least **300 decoded continuation characters** are
  available, under a documented safety token cap, then score and display the first 300.
  **Stop strings are disabled** for this comparison. Any shortfall or decoding error is recorded.
- **Sampled and greedy results are kept separate.** The temperature-0.8 samples drive the
  human ship comparison; greedy output is a diagnostic.
- Raters see the prompt, but the prompt is **excluded from all continuation metrics**.
- The three acceptance-test prompts (handout Section 16) are the three speaker cues.
- The out-of-domain prompt is **reported separately**, so it does not hide in-domain quality.

#### 15d. Automatic sample diagnostics
| Metric | What it measures (and does not) |
|---|---|
| Distinct-n (word 1/2-grams); repeated character 10-gram rate | **Diversity and repetition diagnostics.** Nonsense can also score as highly diverse. |
| **Training-word coverage** | Fraction of generated words found in the training vocabulary. Unseen words may be valid names, inflections or English words, so this does **not** measure correctness directly. |
| **Training-text overlap** | Longest substring matching training text, and fraction of generated character 20-grams found in training. **Flags passages for inspection**; it cannot cleanly separate memorization from ordinary shared language. |
| Speaker formatting | Fraction of speaker-like lines matching a **corpus-derived pattern** that allows labels such as `First Citizen:`, not only all-caps names. If a sample has no speaker-like lines, the result is **not applicable**, not 0 or 1. |
| BPE decoding-error rate | From D14g. Counted as a quality failure. |

- Documented processing rules: word segmentation, apostrophe handling, case normalization,
  and the denominator of every metric.
- Only **generated text** is scored. An **artificially cut final word** is excluded from
  word-based metrics.

#### 15e. Human assessment (D1 ship rule)
**Anchored rubric**, fixed before any rating:

| Criterion | 1 | 3 | 5 |
|---|---|---|---|
| Readability / coherence | mostly unintelligible | understandable fragments with frequent breakdowns | mostly coherent continuation |
| Freedom from repetition | dominated by loops | noticeable unnecessary repetition | little distracting repetition |
| Shakespeare-like style and formatting | little resemblance | recognizable but inconsistent | consistently plausible for this corpus |

**Procedure:**
- **Blind:** samples are shuffled and given random IDs. Model identity is hidden until all
  ratings are finished.
- Each rater rates independently, and the mean is reported. With a single rater, the report
  says so.
- Models are compared on **matched prompts and sampling seeds**, with **prompts weighted
  equally**. Per-criterion differences are reported with uncertainty. Several samples from one
  prompt are not independent prompts, and several raters do not multiply the number of
  independent texts.
- The out-of-domain prompt is excluded from the ship comparison and reported separately.

**Ship-comparison samples (fixed in advance, single pass):** for each of the **7 in-domain
prompts**, **6 predetermined temperature-0.8 sampling seeds** per model. The first 3 are
15c's samples. These are generation seeds, not extra training runs. Both models are the
**actual proposed shipping checkpoints** (D17). In total that is 7 × 6 × 2 = 84 blind
samples per rater.

**Aggregation:** for each prompt, average over sampling seeds and raters first. This gives
**7 paired per-prompt differences** (BPE − character) per criterion, with prompts weighted
equally.

**Ship rule (human part).** BPE ships only if **both** hold:
1. D1's validation BPC rule passes against the **selected character finalist** (D17a),
   with D1's seed check if the gap is small; **and**
2. on **every** criterion:
   - the mean per-prompt difference is **≥ −0.5**, **and**
   - the lower bound of the **two-sided 90% paired t-interval** over the 7 per-prompt
     differences is **≥ −0.5**.

The rule is applied **once**. If it fails, **character-level ships**. Rater disagreement is
reported, but it does not trigger another round.

**Why a t-interval, not a percentile bootstrap:** with only 7 prompts, a percentile bootstrap
is too narrow. In a simulation (normal data, n = 7, 4,000 trials), a nominal 90% percentile
interval covered the true mean only **83.1%** of the time, while the t-interval covered
**90.6%**. A too-narrow interval would make BPE's gate easier to pass. A seeded percentile
bootstrap (10,000 resamples) is **reported alongside** as a robustness check. It does not
decide anything. **Caveat (D24m):** the coverage simulation assumed normal data, so the
t-interval is a **benchmark acceptance convention**. It does not establish coverage for 7
hand-picked prompts with bounded author ratings. The script is kept at
`docs/analysis/ci_coverage_sim.py`.

**Interpretation:** −0.5 on a 1–5 scale is a **chosen practical tolerance**, and the rule is a
predetermined acceptance rule for this benchmark. It is deliberately conservative: BPE must
**show** it is not worse before it replaces the default. Seven chosen prompts give limited
coverage, and passing is **not** proof of general equivalence.

#### 15f. Test-set policy
- Before any test result is revealed, these are frozen: model choice, decoding settings,
  baseline smoothing, evaluation code and the prompt set.
- The test set is evaluated **once** for the final model and the baselines. Reruns to verify
  correctness are allowed; using test results to retune anything is not.

#### Alternatives rejected
- **Per-token perplexity across tokenizers:** not comparable (D1).
- **n-gram beat as evidence of long-range context:** an overclaim. D3's prefix diagnostic is used instead.
- **"Lexical validity" / "memorization" labels:** claim more than the metrics measure.
- **Unanchored 1–5 ratings or an undefined "not worse":** leave room to decide after seeing results.

**Depends on:** D1 (ship rule), D2 (splits), D3 (scoring protocol, prefix diagnostic),
D14 (generation, stop strings, decoding errors).

### D16. Experiment plan, seeds and priorities

**Status:** Accepted · 2026-10-10 · **Seed tier is fixed after the pilot, before any
ablation result is seen.**

**Decision:** Seven ablation configurations under one pre-registered seed tier (3 seeds
preferred); two short Lesson-06 demonstrations (target leakage and MLM reconstruction); an
LSTM-vs-attention micro-benchmark; a greedy-vs-sampling comparison; the D1 BPE comparison
with seed confirmation for a winner; and optional frozen-vs-trainable positions and Post-LN,
in that priority order. RoPE is skipped.

#### 16a. Required ablations
- Depth {2, **4**, 6}, heads {2, **4**, 8}, context {64, **128**, 256}. Bold = D11 baseline,
  shared, so there are **7 distinct configurations**.
- **Seed tier, fixed after the pilot (D12b) and before any ablation result is seen.**
  The same tier applies to **all seven configurations**:
  - **Preferred:** 3 seeds (0, 1, 2).
  - **Reduced:** 2 seeds (0, 1).
  - **Minimum:** 1 seed (0), presented as **exploratory**, with no claims about seed variance.
- How many seeds a configuration gets **never** depends on how its first result looked.
- Reporting: each seed's result, mean ± **sample** standard deviation, and each variant's
  **difference from the baseline at the same seed**. Three seeds give useful evidence with
  limited precision, and the report says so.
- Baseline runs are reused only under D11's exact-match rule.

#### 16b. Lesson-06 demonstrations (short, 5 epochs, single seed, illustrative)
**Target-leakage run:**
- The baseline trained with **no causal mask** on next-token prediction, compared against a
  **causal baseline trained fresh with the identical 5-epoch schedule**, optimizer,
  batches, seed and exposure. Only the mask differs (**D24c**). An epoch-5 checkpoint from
  a 30-epoch run is not a matched control. Both runs use a key-padding mask where D24c
  requires it.
- Report the leaky model's held-out loss both **without and with** the causal mask, show its
  generation, and **measure** how much it degrades.
- Note for the report: the **last target in each shifted window** (`s[i+T]`) is not in the
  input, so it is not directly visible even without the mask. Leakage affects positions
  0…T−2.
- Loss collapse and generation failure are **hypotheses to test**, not promised outcomes.

**MLM run:**
- Bidirectional attention (no causal mask). **15% of positions** are selected, with the
  corruption policy documented (e.g. BERT's 80/10/10 replace-with-`[MASK]`/random/keep), using
  a `[MASK]` token added as a documented vocabulary extension (D1). The model predicts the
  **original tokens at the selected positions**, with **no next-token shift**.
- Loss and accuracy are computed **only at the selected positions**. Evaluation uses
  **fixed held-out corruption masks**.
- Reported: reconstruction examples, and masked-token accuracy against a suitable baseline
  (the training-set unigram predictor).
- MLM masked-token accuracy is **not ranked against** CLM next-token accuracy. They are
  different tasks with different conditioning (BERT uses both left and right context).
- Wording: the MLM model is **"not trained for GPT-style next-token generation"**, not
  "cannot generate". Masked models can generate through other procedures (Wang & Cho, 2019).
- Together, the two runs separate **target leakage** from **legitimate bidirectional
  reconstruction**.

#### 16c. Lesson-01 micro-benchmark (no training)
- **Our manual attention layer vs PyTorch's optimized `nn.LSTM`**, labelled that way, at
  sequence lengths 64…1024 on the T4.
- Fixed batch size, width, precision, device and train/eval mode. Warmup, synchronized GPU
  timing and repeated measurements (`torch.utils.benchmark`). Parameter counts and a
  clearly defined peak-memory measurement are reported.
- **Interpretation limits:** equal width does not mean equal parameters or compute. Timing
  supports the hand analysis but does not by itself prove asymptotic complexity. The
  quadratic memory claim refers to **our explicitly stored score matrices** (D6).

#### 16d. Decoding comparison (no training)
- Fixed comparison: **greedy vs temperature 0.8** on the final model, using D15's prompts and
  metrics. Other settings (temperature 1.0, top-p 0.9) are optional.
- Any decoding setting used for the shipped model is chosen **before** test results are
  revealed (D15f).

#### 16e. BPE comparison (D1)
- Seed 0 **screens** the BPE baseline and the matched-context character run.
- **A BPE model that would ship must be confirmed with seeds 1 and 2**, however large its
  first-seed gap is. A large first gap is not evidence of low variance. **The full
  promotion rule is D24d** (the character finalist gets the same 3 seeds; gap ≥ 0.02 BPC).
- The matched-context character run stays **exploratory** unless a stronger conclusion
  depends on it.
- Caveat: BPE runs use the LR and dropout tuned for the character model (D10), so this is a
  comparison at shared settings, not at BPE's own optimum.

#### 16f. Optional extras (only after everything above, in this order)
1. **Frozen vs trainable positions.** The trainable position table is **initialized from the
   same sinusoidal values**, so initialization scale is controlled (D5). Compare against the
   frozen sinusoidal baseline. It is labelled as "frozen vs trainable sinusoid-initialized
   positions", with the added parameters reported (`context_length × d_model`).
2. **Pre-LN vs Post-LN at depth 6** (D4), single seed, with per-layer gradient norms logged.
   **Illustrative:** with one shared LR schedule it shows behavior **under that schedule**, not
   each normalization scheme's best achievable performance.
3. **RoPE: skipped.** It would change the attention code (D6c, D14d) for an extra-credit item
   that option 1 already covers.

#### Run inventory (full 30-epoch runs; assumes the grid and finalists already include all
three baseline seeds)
| Block | Additional full runs |
|---|---:|
| Grid and finalist confirmations (D10e) | 13 |
| Required ablations after baseline reuse | 18 |
| Initial BPE and matched-context comparison | 2 |
| Final-model training (D17e: 4 per new combined candidate, up to 2 candidates) | 0–8 |
| Full pilots, if needed | 1–2 |
| **Preferred-tier subtotal (not a ceiling, D24i)** | **34–43** |
| Optional frozen-vs-trainable positions and Post-LN | 4 |

**Zero-cost note:** 34–43 is the **preferred-tier subtotal**. Including optional runs and
BPE promotion, the upper end is 51 (D24i). None of it is guaranteed capacity. Free Colab GPU
access is variable and limited. If capacity runs short, the cut order below is applied and
the seed tier is chosen after the pilot. Buying compute is not an option.

Not included above: the two short Lesson-06 runs (plus a 5-epoch causal reference), and
BPE seed confirmations if needed. The AMP timing comparison is not automatically a full run.
Time is estimated **per configuration**, because depth and context variants cost
different amounts.

#### Cut order (if the pilot shows the plan doesn't fit)
1. Optional extras (16f).
2. Grid finalist confirmations beyond the planned minimum.
3. A lower **uniform** seed tier for the ablations (3 → 2 → 1), chosen before any ablation
   result is seen.

**Protected scope:** the grid, the required ablations at the chosen tier, 16b–16e. Free
compute is reserved in D24i's order. If even this scope cannot fit, it is **amended
explicitly in this log**, never cut silently.

**Depends on:** D1, D4, D5, D10, D11, D12, D15. **Resolves:** the experiment-budget items left
open in D4 (Post-LN) and D5 (learned positions, RoPE).

### D17. Final-model selection

**Status:** Accepted · 2026-10-10 · **The candidate set and the 0.02 BPC band are fixed
before any final-candidate run.** The latency limit comes from the deployment decision (D18).

**Decision:** The candidates are C0 (the grid-selected baseline) plus at most two combined
character-level candidates, all declared in advance. Each new candidate gets a local
two-point LR check. The winner is the lowest mean validation BPC, with a fixed 0.02 BPC band
in which the faster model wins. The best-validation seed of the winning configuration
ships. BPE can replace it only through the D1/D15e rule, applied against that character
finalist.

#### 17a. Candidate set
- **C0:** D11's architecture with D10's grid-selected LR and dropout. It **guarantees that a
  trained checkpoint exists**. It does **not** guarantee that the checkpoint meets D18's
  latency limit.
- **Amended by D24e:** already-trained sinusoidal character ablation configurations are also
  candidates, at no retraining cost. Zero new combinations is allowed. C0 is **not**
  automatically eligible.
- **At most 2 combined candidates**, e.g. the best depth with the best context length,
  chosen from the D16a ablation results. Each one's **full architecture, tokenizer and
  hyperparameters are written down before its final training starts**.
- **BPE's opponent is the selected character finalist** (the winner of 17c), not C0. Without
  this, BPE could beat C0 and still be worse than the character model the ablations found.
  The BPE candidate is D16e's baseline-architecture BPE model. Comparing it with a
  differently shaped character finalist is a **ship decision**, not a clean attribution to
  the tokenizer. D16e's matched comparison remains the tokenizer evidence.
- The human comparison (D15e) uses the **actual proposed shipping checkpoints**. The test set
  stays untouched throughout.

#### 17b. Local LR check for new candidates
- LR ∈ {grid LR, ½ × grid LR} at **seed 0**. Dropout, budget and evaluation protocol stay
  frozen. The LR with the lower validation BPC is chosen, then trained with seeds 1 and 2.
- This is called a **local LR check**. It does not claim to fully fix the mismatch between a
  new shape and the baseline's optimum (the new shape might prefer a higher LR or a
  different dropout).

#### 17c. Selection rule
1. **Eligibility gates:** all D13 and D14 checks pass, there are **zero decoding errors on
   the frozen D15 sample suite** (D24d), and the model meets **D18's latency and memory
   limits**. BPE promotion follows **D24d** only.
2. Among eligible candidates, find the lowest **mean validation BPC** over training seeds.
3. Candidates within **0.02 BPC** of that value enter the tie-break. 2^0.02 ≈ 1.014, so the
   band is about a **1.4% difference in character-normalized perplexity**. This is an
   **engineering tolerance**, fixed now. Candidates inside it are **not** described as
   statistically equivalent.
4. **Tie-break:** the fastest candidate on D18's predetermined CPU benchmark wins. The
   benchmark uses the same prompts, hardware, thread settings and **number of generated
   characters**, not tokens, so that tokenizers are compared fairly.
- Paired per-seed differences and seed variability are reported alongside, as information.
  The band does not depend on them.

#### 17d. Which checkpoint ships
- The `best.pt` of the **best-validation seed** within the winning configuration, exported
  as described in D13.
- **Reported:** the configuration's mean ± sample std over training seeds; the shipped
  checkpoint's validation BPC, seed and epoch; and its **single test result**, taken after
  all selections are frozen (D15f).
- **Wording:** the selected validation BPC is **optimistic** because it was used to choose
  the checkpoint. The test result estimates **that shipped model's** performance, not the
  configuration's average over seeds.

#### 17e. Budget and run count
- Every candidate uses the **D12 pilot-confirmed budget** (30 epochs, or the one extension
  chosen before the grid). `best.pt` is the best epoch **within that budget**, which does
  not show that more training could not help.
- At the preferred 3-seed tier, each new candidate costs **4 full runs** (2 LR trials at
  seed 0, then seeds 1 and 2), so **up to 8** for two candidates. Reduced tier: 3 per
  candidate. Minimum tier: 2. C0 reuses the matching existing runs (D11 exact-match rule).
  BPE seed confirmations (D16e) are counted separately. D16's run inventory has been updated.

#### Alternatives rejected
- **"Within 2× seed std" as the tie band:** it is unclear whose std that means, and noisier
  configurations would get a wider band. Seed std measures variability, not a negligible
  difference.
- **BPE judged against C0:** it could ship a model worse than the character finalist.
- **Retuning new candidates over the full LR × dropout grid:** costs too much for the
  likely gain.
- **Shipping a seed-averaged or "median" seed:** shipping the best-validation seed is valid,
  as long as the report is honest about its optimism.

**Depends on:** D1, D10, D11, D12, D13, D15, D16, D18 (latency limit). **Resolves:**
D11's pending "final shipped-model decision" (the latency benchmark itself moves to D18).

### D18. Deployment platform, container and latency limit

**Status:** Accepted · 2026-10-10 · **Render Free is the platform, provided the early
deployment pilot (18c) passes. The latency target stays provisional until that pilot is
measured, and is frozen before any final-candidate comparison (D17).**

**Decision:** Render Free hosts a portable, CPU-only Docker container with the checkpoint
baked in. Before any final-candidate comparison, an early deployment pilot measures
memory, warm and cold latency, and responsiveness. D17's latency gate is: **warm, 100
generated characters within 30 s**, provisional until the pilot confirms it. D14's no-cache
decision stays.

#### 18a. Platform
- **Render Free** (0.1 CPU, 512 MB RAM, sleeps after 15 idle minutes, about 1 minute to
  wake, 750 instance hours a month). It is used **without a payment method**. Render's docs
  say that exhausting instance hours **suspends** free services, and that bandwidth overage
  is billed only if a payment method exists.
- **Hugging Face Spaces is excluded:** its docs say Docker and Gradio Spaces "require a paid
  plan to create". The free exception, Gradio on ZeroGPU, does not fit a FastAPI container.
- **If the pilot fails:** first benchmark the smaller shapes that already exist in the
  ablations (e.g. depth 2). They are eligible candidates under D24e. Then consider another
  free host after checking its current terms. A local demo is kept as a backup in every
  case, but it **does not meet** the hosted-service requirement (D24h).

#### 18b. Container
- One **portable Dockerfile**. The base image is pinned **by exact version or digest**, not
  a moving tag like `python:3.13-slim`. Python and PyTorch versions are a **tested pair**,
  and PyTorch comes from the **CPU-only wheel index**.
- The checkpoint is supplied **during the build** through `MODEL_SOURCE` and `MODEL_SHA256`
  (**D24g**): the pinned Release URL in production, or a labelled test/random artifact for
  CI and the pilot. It is checked against its SHA-256, and a failure stops the build.
  Nothing is downloaded at runtime.
- A non-root user, `$PORT`, **one uvicorn worker** and **`torch.set_num_threads(1)`**.
  Matching the thread count to vCPUs is meaningless with a fractional CPU allocation.
- **Concurrency:** one active generation at a time, behind a short waiting queue. Requests
  beyond the queue get an immediate "busy" error (the exact status code and limits belong
  to the API decision), so several visitors at once cannot exhaust memory.
- A `/health` endpoint reports the model's hash and configuration. Health checks must keep
  responding while a generation runs, which the pilot checks.
- Builds are kept few, because free build minutes are limited (the PyTorch download is
  large).

#### 18c. Early deployment pilot (before final-candidate comparison)
- **Latency does not depend on the weights,** so the pilot uses a **randomly initialized
  model of each candidate shape** and can run **before any training finishes**. This allows
  D17's latency gate to be checked per shape early.
- **Measured on Render Free:** startup success; **peak memory during generation** (the
  512 MB limit covers PyTorch, its dependencies and intermediate tensors, not just a
  12.8 MB checkpoint); warm time for **100 and 300** generated characters; cold-start time;
  and `/health` responsiveness during generation.
- **Fixed benchmark:** D15's 7 in-domain prompts × 3 seeds, **plus full-window prompts**
  (prompts already ≥ T tokens). Temperature 0.8, stop strings disabled, a few warm-up
  requests, one request at a time; median and p95 are reported. *(Amended by D19.)* With a
  short prompt, many generation steps process fewer than T positions, so short prompts alone
  would understate the cost. The latency gate applies to the **worse of the two prompt
  sets**, because API users can paste long text.
- **BPE caveat:** characters per token depend on what the model generates, so random
  weights cannot measure a BPE model's final characters per second. That needs the trained
  checkpoint.
- **Local reference only, not the gate:** on the development laptop (one thread, D11 shape,
  no cache), 100 characters took **1.1–1.35 s** from short prompts and **1.94 s** from a
  full 128-token prompt, and 300 characters took **9.4 s**. Laptop timings vary from run to
  run. Scaling
  this by about 10× for a 0.1 CPU allocation is only a rough guess, since architecture and
  throttling differ. That is why the gate is measured on Render.

#### 18d. Latency gate (D17 eligibility) and workloads
- **Usability gate (provisional):** warm p95 for **100 generated characters ≤ 30 s** on
  Render Free. It is confirmed or adjusted **once**, from the pilot, before any
  final-candidate comparison, and never after seeing candidate quality.
- The **300-character** workload stays the quality-evaluation length (D15), and its speed
  is reported as **throughput**. Shorter output only shortens the wait; it does not make
  sustained generation faster.

#### 18e. Demo policy
- Wake the service before recording or presenting. Cold and warm timings are reported
  separately. A **local demo backup** is always kept ready.

#### Alternatives rejected
- **Hugging Face Spaces PRO ($9) and Render Starter ($7):** break the zero-cost constraint.
- **Adding a KV cache for Render's slow CPU:** a cache is exact only until the context
  reaches T tokens. After that, D14's cropped window re-indexes the absolute positions (D5),
  which makes cached keys and values invalid. Claude's earlier claim that a cache would
  cut sustained generation cost "several times over" was an **overclaim**.
- **"50 ms per character" as a limit:** had no basis on free hosting.
- **Threads = vCPU count:** meaningless for a 0.1 CPU allocation.

**Depends on:** D5, D11, D13, D14, D15, D17. **Resolves:** D11's pending latency benchmark.

### D19. API design

**Status:** Accepted · 2026-10-10

**Decision:** A FastAPI app with `GET /health`, `GET /model-info`, `POST /generate`,
`POST /generate/stream` (server-sent events, SSE) and a static same-origin UI at `/`. One
generation engine produces incremental output: `/generate` collects it and `/generate/stream`
forwards it. **Admission** (1 active + 2 waiting), queue timeouts and cancellation signals live on the
**HTTP side**. A **single dedicated model worker** runs accepted work, checks deadlines
and cancellation between steps, and frees its slot only when computation stops.

#### 19a. Request schema (Pydantic, `extra="forbid"`, strict types)
| Field | Rule |
|---|---|
| `prompt` | Up to 1,000 **Unicode characters**. The **whole** prompt is validated against the loaded tokenizer (D1: unknown character → 422) **before** it is cropped to the last T tokens (D14). |
| `max_new_tokens` | Strict integer, 1–300, default 100. For BPE, tokens ≠ characters: this field limits **work**, while D18's gate is measured in **characters**. |
| `temperature` | Finite float, 0–2, default 0.8. 0 means greedy (D14). |
| `top_k` | Optional strict integer, 1…V of the **loaded** tokenizer. Off by default. |
| `top_p` | Optional finite float in (0, 1]. Off by default. |
| `seed` | Optional strict integer, 0…2³²−1. If omitted, the server picks one and **returns it**. Each request uses its own `torch.Generator`; global random state is never reset. |
| `stop` | Up to **4** non-empty strings, each ≤ **64** characters. These are app limits, not assignment requirements. D14e's matching rules apply. |

Misspelled or unknown fields are **rejected**, so a typo cannot silently fall back to a
default.

#### 19b. Response (D14f metadata, plus serving fields)
- `prompt` (as sent); `text` is the **continuation only**; prompt-truncation flag;
  newline-fallback flag.
- Sampled token count and visible character count; `finish_reason` and the matched stop
  string.
- Resolved settings: `seed`, temperature, top-k, top-p, `max_new_tokens`, stop strings.
- Model identity: tokenizer, context length, parameter count, release tag and checkpoint
  SHA-256.
- `queue_ms`, `generation_ms` and characters per second (D14f definition).
- **`finish_reason` values** (D14e extended): `"max_new_tokens"`, `"stop_string"`,
  `"time_limit"`, `"cancelled"`.
- `/model-info` returns the stable model fields, plus vocabulary size, configuration, the
  `max_new_tokens` and prompt limits, and default settings.
- **Reproducibility wording:** a seed reproduces the sampling sequence only under matching
  conditions (same checkpoint, settings and environment, D14f). A `time_limit` stop can
  also end the same request at a different length on another run.

#### 19c. Concurrency, deadlines, cancellation
- **A single dedicated model worker thread** runs every generation. Requests are admitted
  through a counter shared by **both** generation endpoints: **1 active + 2 waiting**.
  - **Admission, queue timeouts and cancellation signals live on the HTTP side (the event
    loop)**, so they stay responsive while the model thread is busy. The model thread only
    takes work that has already been accepted. *(Clarified at D20.)*
  - FastAPI runs plain `def` endpoints in a thread pool, but that does **not** run requests
    one at a time. Admission is therefore explicit.
  - Handlers are `async def`, and model computation never runs on the event loop.
- **Queue wait limit 10 s.** A full queue, or a request that waits longer, gets **503 with
  `Retry-After`**.
- **Generation budget 60 s**, measured with a **monotonic clock** from the moment generation
  starts. The loop checks its deadline and a cancellation flag **between token steps**.
  When the deadline passes, the partial text is returned with `finish_reason="time_limit"`.
  - This deadline is **cooperative**: it cannot interrupt a PyTorch operation midway.
  - Wrapping a thread call in a timeout does not stop the thread (AnyIO documents that
    abandoned threads keep running). That is why the loop must check the deadline itself.
- **Client disconnect:** a waiting request is removed from the queue. An active one is
  signalled to stop.
- **The active slot is released only when computation has actually stopped**, including when
  an exception occurs (`try/finally` inside the worker).
- About 10 s of queueing plus 60 s of generation is an **application budget**, not a hard
  end-to-end guarantee. It excludes waking the service and the network, and it can
  overrun slightly while a tensor operation finishes.

#### 19d. Streaming (`POST /generate/stream`, SSE)
- **Events:** `start` (resolved settings and seed), `delta` (visible text) and `done` (the
  **same** metadata as `/generate`). An `error` event is used only for failures **after**
  the response headers have been sent.
- **Validation and worker acquisition happen before any streaming headers are sent.** A
  streaming request waits until it actually holds the model worker, not just a queue slot,
  or until its queue wait expires. So 422 and a 503 for queue overflow or timeout still
  arrive as real HTTP statuses (D24m). `start` is sent only after that.
- A client that disconnects is not promised a `done` event. `cancelled` is recorded
  internally.
- **Text is emitted only when it is final** *(corrected at D20)*:
  - Incremental UTF-8 decoding (D14g) runs before emitting.
  - D14 **keeps** the matched stop string in the output and trims only what comes **after**
    it. So text before a possible match never has to be taken back, and no hold-back buffer
    is needed. A rolling suffix (longest stop string − 1 characters) of text already
    emitted is kept only to **detect** matches that span chunks. Each newly decoded piece is
    checked before it is emitted, and only the text after the earliest-ending match is cut.
  - Claude's earlier hold-back buffer would have delayed streaming for no reason.
- The output buffer is bounded and writes respect cancellation.
- **Browser client:** `EventSource` cannot send a POST body, so the UI uses `fetch()` POST
  and reads the response stream, parsing SSE framing that may be **split across network
  chunks**.
- The Render pilot (D18c) must confirm that text **actually arrives incrementally**, since
  proxies can buffer responses.
- Streaming improves how responsive the demo **feels**, not how fast the model generates.
  Most of the work is in handling the stream's lifecycle; there is no line-count promise.

#### 19e. UI and access
- Static HTML at `/`, same origin (no cross-origin setup needed), plus FastAPI's `/docs`.
  No login.
- **Waiting** and **generating** states are shown, and there is a **Stop** button that aborts
  the fetch, which counts as a disconnect.
- Generated text is displayed **as text, never as HTML**, with whitespace preserved.
- **Wording:** the queue limits concurrent work; it is **not rate limiting**.
- **A best-effort, in-memory per-client limiter** for basic fairness (e.g. 10 generation
  requests a minute). The client is identified from Render's forwarded-for header, and how
  that header behaves is checked in the D18c pilot. It is not a security control, and it
  resets when the service restarts.

#### 19f. Required tests
- With no timeout, a fixed seed gives **identical** text and **deterministic** metadata
  from `/generate` and from the reassembled stream. Timing fields (`queue_ms`,
  `generation_ms`, characters/second) are excluded.
- The queue limits are **shared** across both endpoints: a 4th simultaneous request gets
  503 with `Retry-After`. The queue wait limit expires correctly.
- **Cancellation really stops the work:** after a disconnect, the worker becomes free and
  the next request is admitted. Errors release the slot.
- `time_limit` returns partial text (tested with an injected fake clock).
- A stop string **split across chunks or tokens** is detected. Only the text after it is
  trimmed, and that text is never streamed.
- 422 cases: unknown character, unknown field, non-integer `seed`, NaN temperature,
  out-of-range `top_k`.
- `/health` keeps responding while a generation is running.

#### Alternatives rejected
- **Plain `def` endpoints alone:** they do not run requests one at a time, and timeouts
  cannot stop thread work.
- **`EventSource`:** it can only send GET requests, with no request body.
- **Separate streaming and non-streaming engines:** they would drift apart.
- **Claude's original `finish_reason` names (`length`, `stop`):** inconsistent with D14.

**Depends on:** D1, D14, D18. **Amends:** D14e (`finish_reason` values), D18c (full-window
benchmark prompts, BPE throughput caveat).

### D20. Repository layout, dependencies and configuration system

**Status:** Accepted · 2026-10-10

**Decision:** Keep the handout's Section 10 layout, extended with a few modules. Code runs
from the repository root as `python -m src.…`, with no installed package. Dependencies
are pinned per environment with full transitive lock files: **CPU** for serving and local
work, **CUDA** for Colab training. YAML plus frozen dataclasses, resolved in one documented
order. **Inference takes architecture and tokenizer settings from the checkpoint.**

#### 20a. Layout
```
README.md  pyproject.toml  Dockerfile  .gitignore
requirements/   serve-cpu.txt  dev-cpu.txt  colab-cuda.txt   (+ *.in sources)
configs/        tiny_gpt.yaml (D11 baseline)  prompts.yaml  experiments/*.yaml (diff-only overlays)
data/           README.md  manifest.json (committed)  input.txt (downloaded, verified, ignored)
src/            __init__.py  config.py  artifacts.py (D24g)  tokenizer.py  dataset.py  attention.py
                positional_encoding.py  transformer_block.py  gpt.py  train.py
                generate.py  evaluate.py  checkpoint.py  utils.py
src/serve/      __init__.py  app.py  worker.py  static/index.html
tests/          test_attention.py  test_mask.py  test_shapes.py  test_overfit.py  … (D21)
experiments/    attention_visualization.py  ablation.py  evaluate.py  ngram_baselines.py
                lstm_benchmark.py  latency_pilot.py  human_eval.py  make_figures.py
notebooks/      minigpt_colab.ipynb
outputs/        figures/ logs/ samples/   (ignored, except the final report figures and tables)
checkpoints/    .gitkeep   (ignored)
report/         article source, figure index, exported PDF
docs/           decisions.md
```
- Every Python package directory has an `__init__.py`. Commands run from the repository
  root, with **no `sys.path` changes**.
- **Experiment scripts import the production** tokenizer, evaluator, checkpoint loader and
  `generate()`. They never reimplement any of them.
- **Final report figures are committed together with their small result tables** (CSV or
  JSON: metrics, run IDs, config and checkpoint references), so every chart can be checked.
  Large logs and checkpoints stay out of git.
- **Ignore rules:** `data/*` followed by `!data/manifest.json` and `!data/README.md`.
  Ignoring the whole `data/` directory would make git unable to re-include those two
  files. A test confirms that `git check-ignore` does **not** ignore the manifest.

#### 20b. Dependencies and environments
| Environment | Lock file | PyTorch build |
|---|---|---|
| Render and local CPU (serving) | `requirements/serve-cpu.txt` | Pinned CPU-only wheel |
| Local CPU development, tests, analysis | `requirements/dev-cpu.txt` | Same CPU-only wheel, plus test, plot and notebook dependencies |
| Colab training | `requirements/colab-cuda.txt` | Pinned, tested CUDA build that works with the Colab runtime |

- **Full transitive locks** are compiled from small `*.in` source files with a free lock
  tool (e.g. `uv pip compile`). Exact pins on direct dependencies alone still leave the
  rest of the dependency tree free to change.
- `pyproject.toml` holds metadata and tool settings (pytest, linter) **only**. There is no
  second dependency list in it to fall out of sync.
- **The serving image contains only `serve-cpu.txt`.** Plotting, notebook and experiment
  dependencies stay out of it.
- **A CPU install never reaches a Colab training run:** the notebook first picks its path
  (CPU: `serve-cpu.txt`; CUDA: `colab-cuda.txt`). A GPU is required, and checked with a
  clear error, **only when training is requested** (D24l).
- **Python 3.12 and 3.13 support is something we test, not just claim:** install plus the
  core tests on both (D21), and a separate CUDA smoke test in Colab. Each run also records
  its exact environment in the run manifest.
- Running from the root without installing is a valid interface. The README, notebook and
  Dockerfile all set the repository root as the working directory. The `src` package name
  matches the handout's `python -m src.generate` example, and packaging the project later
  is still possible.

#### 20c. Configuration
- **Sections:** `model`, `tokenizer`, `data`, `train`, `generate`, `eval`, `serve` (queue
  capacity, wait timeout, generation deadline, request limits). Every Section 11 parameter is
  configurable.
- **One order for resolving settings:** baseline YAML → each listed overlay **in order** →
  command-line `--set a.b=value` overrides → **fill in derived values** → **validate** →
  **freeze and save**.
  - Nested mappings merge recursively. Scalar values and lists **replace** the earlier value.
  - Defaults apply only to values that were not given. For example, a missing `ffn_dim`
    becomes 4 × `d_model`, but a width set explicitly stays as given.
  - Derived values (gradient accumulation, total updates, warmup) are computed after both
    the config and the dataset size are known.
- **Validation:** YAML is loaded with `yaml.safe_load`, and **duplicate keys are rejected**
  using a custom loader. **Unknown keys are rejected at every nesting level.** Types are
  checked and coerced explicitly (a dataclass's annotations and `frozen=True` do not validate
  anything on their own). Non-finite numbers are rejected. Cross-field rules are checked
  too, e.g. `d_model % n_heads == 0`, and dropout and ranges.
- **Paths** are relative to the **repository root**, never to the folder of whichever script
  loaded them.
- **What is authoritative:**
  - For **inference**, the architecture and tokenizer come from the **checkpoint** (D13).
    A current YAML file never overrides them silently, and a request to override them is
    an error.
  - **Resume** rejects changes that would make training incompatible (model, data, optimizer
    or schedule settings). It allows documented operational changes, such as the output
    directory or the logging interval.
- `python -m src.config --config … [--overlay …] [--set …]` **prints the fully resolved
  config without training**. Dataset-dependent counts are shown only when the tokenizer
  and data metadata exist, and are otherwise marked "unresolved" (D24b).
- The resolved config is saved in the run directory and stored in the checkpoint as a
  plain dict (D13).

#### Alternatives rejected
- **Hydra or OmegaConf:** too much implicit behaviour to defend line by line. A small loader
  can be explained fully.
- **One CPU-only install recipe everywhere:** it would turn off GPU training on Colab.
- **Exact pins on direct dependencies only:** the rest of the dependency tree could change
  between installs.
- **Inference rebuilding the model from the current YAML:** the model could silently stop
  matching its checkpoint.

**Depends on:** D2, D12, D13, D18, D19. **Amends:** D19 (stop-string streaming, where
admission runs).

### D21. Verification: tests, CI, acceptance script and CLI

**Status:** Accepted · 2026-10-10

**Decision:** Three test tiers with pytest markers. Free GitHub Actions CI on CPU for
Python 3.12 and 3.13. One acceptance command that runs the handout's Section 16 checks.
A CLI that matches Section 12.1 and returns D14f's metadata.

#### 21a. Test tiers
| Marker | Contents | Where it runs |
|---|---|---|
| `unit` | Tiny configs, CPU only; runtime target calibrated on the dev laptop (D24k) | Locally and in CI |
| `integration` | Overfit gate; checkpoint reload **in a fresh process**; D13 exact-resume subprocess test; API tests (D19f) with a tiny checkpoint **generated during the test**, never committed | Locally and in CI |
| `slow` | Colab CUDA smoke test (D20b); latency pilot (D18c) | Run manually |

#### 21b. Required tests (handout Section 10/16, plus earlier decisions)
- `test_attention`: the implementation reproduces **D23a's published hand-calculation
  numbers** to within 1e-4, both with and without the mask. Also checked against the SDPA
  math backend (D6).
- `test_mask`: changing tokens at positions > t leaves logits at positions ≤ t unchanged,
  and leaves the gradients of a **prefix-only loss** (prefix logits, fixed prefix targets,
  dropout off) unchanged; ∂(prefix scalar)/∂(future embeddings) = 0 (D24k). Checked in the model forward pass **and** during generation (Section 16, check 6).
  The attention probability for every future position is exactly 0.
- `test_shapes`: shape checks at every stage of split, attention, merge and projection,
  plus the parameter-count formula (D7, D11).
- `test_overfit`: on one repeated tiny batch, loss falls **below 10% of its starting value
  within N CPU steps**. N is calibrated once on the tiny config and then frozen. This shows
  that gradients and target shifting work.
- Plus every test already required in D1–D3, D6, D8, D13, D14, D19f and D20a (the
  manifest is not ignored by git).

#### 21c. CI (GitHub Actions, free)
- Matrix: Ubuntu × Python {3.12, 3.13}, CPU, using `requirements/dev-cpu.txt`.
- Jobs: lint (ruff); `unit` + `integration` tests; **Docker build plus a container smoke
  test** (`/health` and one generation from the test checkpoint); a free secret scanner
  (e.g. gitleaks).
- This uses only free minutes (free for public repositories, with an included monthly
  allowance for private ones). There is no paid runner.

#### 21d. Acceptance command
- `python -m experiments.acceptance` runs Section 16 checks 1–7 in order and prints
  **PASS/FAIL** for each:
  1. Dependencies install.
  2. Unit tests pass.
  3. Tiny overfit.
  4. The **release checkpoint** loads in a fresh process.
  5. Generation from the three speaker-cue prompts (D15c).
  6. Causal check during generation.
  7. README commands run, and the secret scan is clean.
- The README's commands live in code blocks tagged for testing, and the acceptance command
  **runs them**. This shows that those blocks work under the tested conditions.
- **D24k:** `--fresh` installs into a new environment from a clean clone. Inside an
  existing environment, check 1 is marked SKIPPED. Failures give a non-zero exit code. The
  direct shift test and the CLI's stdout/stderr rules are also defined there.

#### 21e. CLI (`python -m src.generate`)
- **Flags:** `--prompt --max-new-tokens --temperature --top-k --top-p --seed --stop
  (repeatable) --checkpoint --json`.
- With no `--checkpoint`, it downloads the D13 release `model.pt`, **verifies its
  SHA-256** and caches it. The architecture and tokenizer come from the checkpoint (D20c).
- **Output:** the Section 12.1 layout (Prompt / Generated / parameters / context length /
  temperature) plus all D14f metadata. `--json` gives machine-readable output with the same
  fields as the API response (D19b).
- **Exit codes:** 0 for success, 2 for invalid input (e.g. an unknown character, D1), 1 for
  any other error.

**Depends on:** D1, D6, D13, D14, D15, D18, D19, D20.

### D22. Experiment operations: logging, run layout, figures and Colab notebook

**Status:** Accepted · 2026-10-10

**Decision:** Local, file-based tracking only. Each run gets its own folder, and an index
CSV lists all runs. All figures are rebuilt from logs by one script. A tagged Colab notebook
that only calls repository commands, with training off by default.

#### 22a. Run folders
- **ID:** `{YYYYmmdd-HHMM}_{config-name}_s{seed}_{short-hash}`.
- **Contents:**
  - `config.resolved.yaml`.
  - `manifest.json`: git commit and uncommitted-changes flag, Python, PyTorch and CUDA
    versions, GPU model, data SHA-256 (D2), start and end time, wall-clock time.
  - `metrics.jsonl`: one row per log or eval event, with step, epoch, train loss, val loss
    and BPC, learning rate, gradient norm, clipping rate, tokens per second, peak memory.
  - `samples.jsonl`: a fixed-prompt sample at every evaluation (handout M4).
  - References to the checkpoints.
- Staged locally and copied to Drive with checksum verification (D13).
- **No external tracker:** W&B needs an account, and TensorBoard isn't needed to make the
  figures.

#### 22b. Run index and results tables
- `outputs/runs.csv`: one row per finished run (ID, config, seed, best val BPC and epoch,
  status). It is **rebuilt from the run manifests** by `python -m experiments.index`, and
  events marked `abandoned` are excluded (D24l). Every results table in the report is built
  from it, never typed in by hand.

#### 22c. Figures
- `python -m experiments.make_figures` rebuilds **all 10 handout figures** using only the
  run folders and the release checkpoint. It uses one fixed matplotlib style and writes PNG
  and SVG.
- Each figure gets a small data table next to it (D20a).

#### 22d. Colab notebook (`notebooks/minigpt_colab.ipynb`)
- **Opens a pinned git tag.**
- **Sections:**
  - §0 Setup: clone, then **choose the CPU or CUDA path** and install the matching lock
    file. The GPU check and Drive mount happen only when training is requested (D24l).
  - §1 Quick inference: download the release model and generate. Works on a CPU runtime
    too.
  - §2 Tests and architecture checks.
  - §3 Training: calls `python -m src.train`. Resuming requires an **explicit run ID or
    checkpoint** and is recorded as a new attempt (D24l). **No model or
    training logic in the notebook.**
  - §4 Evaluation and figures.
- **`RUN_TRAINING = False` by default**, so "Run all" finishes in minutes.
- The committed notebook has **its outputs stripped**. Evidence from a full run goes into
  the report as screenshots.

**Depends on:** D2, D12, D13, D20, D21.

### D23. Deliverables: hand calculation, report, diagram, demo, presentation, attribution

**Status:** Accepted · 2026-10-10 · **The team has two members.**

#### 23a. Hand calculation (handout Section 6, graded on its own)
- Q = [[1,0],[0,1],[1,1]], K = [[1,0],[1,1],[0,1]], V = [[1,0],[0,1],[1,1]] (3 tokens,
  d_k = d_v = 2).
- QKᵀ = [[1,1,0],[0,1,1],[1,2,1]], then ÷ √2 → [[.7071,.7071,0],[0,.7071,.7071],
  [.7071,1.4142,.7071]].
- Softmax rows: [.4011,.4011,.1978], [.1978,.4011,.4011], [.2483,.5035,.2483]. Output
  rows: [.5989,.5989], [.5989,.8022], [.4965,.7517].
- **Causal mask:** row 1 → [1,0,0] (output = v₁ = [1,0]); row 2 → [.3302,.6698,0] (output
  [.3302,.6698]); row 3 **unchanged**.
- **Step 7 explanation:** token 1 can only use itself, token 2 loses access to token 3,
  and the last token was never affected.
- All of this is worked by hand with exact √2 values and written out in full. Every number
  was checked by Claude with NumPy (to 4 decimals), and `test_attention` checks them again
  (D21b).

#### 23b. Medium article (handout Section 15)
- Section 15's ten parts in order. Length is judged on the rendered article and PDF with
  its figures against the 8–12 page recommendation; about 3,500–4,500 words is only a
  guide. The appendix holds extra detail.
- **Medium has no native math or table support:** equations and tables are rendered as
  images from committed sources, and code is embedded as gists. A PDF export goes in
  `report/`.
- Every result is marked as an **observation** (a measurement) or an **interpretation**
  (with its supporting evidence), as Section 13 asks.
- **At least 2 failure modes, taken from real evidence**, e.g. the out-of-domain prompt,
  repetition under greedy decoding, made-up speaker names, the leakage run (D16b).
- All figures and tables come from D22, built from the evidence archive (D24j).

#### 23c. Architecture diagram
- Made in draw.io (free), with **tensor shapes on every edge** (B, T, d, h, d_h, V). The
  source `.drawio` file and a PNG export are both committed.

#### 23d. Demo video and live demo (handout Section 17)
- Section 17's nine steps, in order. **Both the live demo and the recording aim for 5–7
  minutes**, which satisfies Section 17 and Section 4's 5–10 (D24m). Recorded with OBS
  (free).
- **Clean environment:** a fresh clone and a new virtual environment on the laptop:
  `pip install`, tests, CLI generation, then the **live Render service, woken up
  beforehand** (D18e). A local backup is kept ready.
- **Two-person split, roughly equal time:**
  - **Member A:** repository tour, attention and causal mask, test suite, configuration
    and parameter count (steps 1–4).
  - **Member B:** loss curves, CLI generation, attention visualisation, deployed live
    inference, the main ablation result and a limitation (steps 5–9).
- A written script with timings.

#### 23e. Presentation and shared understanding
- About 10 slides that follow the demo, plus a slide prepared for oral-defence questions.
- **Both members must be able to explain and change every core component** (handout
  Section 18 allows individual questions). Practise by having each member field questions
  on the **other's** sections. `docs/decisions.md` is the shared reference.

#### 23f. Human evaluation with two raters (amends D15e)
- Both members **score the same blinded samples independently, before discussing them**.
- Ratings are averaged within each prompt as D15e requires. Each rater's means and
  **disagreement** (per criterion: mean absolute difference and how often they agree on
  the direction) are reported.
- Samples and ratings record the **SHA-256 of the exact artifact** that was assessed (D24j).
- Two raters do **not** double the number of independent prompts. The D15e rule still uses
  7 per-prompt differences.
- **Caveat:** the raters are also the authors, so blinding hides which model produced each
  sample but cannot remove what they already know about the project.

#### 23g. Citations, contributions and AI-use disclosure
- **References** in both the article and the README: Vaswani et al. 2017; Radford et al.
  2018/2019; Devlin et al. 2019; Xiong et al. 2020 (Pre-LN); Hendrycks & Gimpel (GELU);
  Loshchilov & Hutter (AdamW); Sennrich et al. 2016 (BPE); Wang & Cho 2019; Karpathy's
  char-rnn (dataset source, D2) and nanoGPT (consulted), saying **exactly what was
  adapted**, if anything.
- **Contribution statement:** brief and accurate, saying who did what.
- **AI-use disclosure:** AI was used for design review, drafting the decision log and help
  with code. The disclosure says what the team checked themselves. `docs/decisions.md`
  (including the record of AI overclaims that were caught and corrected) is the evidence
  of the team's reasoning.

**Depends on:** D6, D15, D16, D18, D20, D21, D22. **Amends:** D15e (two raters).

### D24. Consistency-audit amendments (cross-decision contracts)

**Status:** Accepted and **locked** · 2026-10-10 · Follows an external audit of D1–D23,
plus a follow-up review (offset guard, AMP details, shift test, notebook test
dependencies). Claude checked
every claim against the log and the handout, and recomputed every number the audit cites:
the D11 parameter counts; 1,003,853 eligible targets → 123 updates; and 1,003,853 mod 128 =
77, so whole blocks cannot always add up to exactly 8,192. **Where this entry and an
earlier decision disagree, this entry wins.** Each affected decision carries a pointer
here.

#### 24a. Exact-target update packing (amends D3a, D12a, D13a)
*Problem:* D12a promises exactly 8,192 valid targets per update, but D3a's two short
boundary blocks (prefix of r targets, suffix of s targets, with r + s ≡ 77 mod 128 for the
character baseline) cannot always be combined with whole blocks to make exactly 8,192.
- **Rule chosen:** each epoch, only the **full T-target blocks** are shuffled. Updates
  1 … ⌊N/8192⌋ each take exactly 8,192 / T full blocks (8,192 targets). The **final update of
  the epoch** takes the leftover full blocks **plus both short boundary blocks**. For the
  baseline that is 1,003,853 − 122 × 8,192 = **4,429 targets**. This matches D12's
  "remainder update" and keeps 123 updates per epoch.
- **Offset guard (fixed at lock):** with update budget B (8,192, or the nominal B from
  the next bullet), an epoch's random offset r is **accepted only if its two short blocks
  together fit in the remainder**: r + s(r) ≤ N mod B, where s(r) = (N − r) mod T. Offsets
  are drawn uniformly from the valid set with the sampler generator. **Offset 0 is always
  valid** when T divides B, because N mod T ≤ N mod B. If N mod B = 0, there is **no
  final update** (offset 0, so no short blocks).
  - *Why it is needed:* N = 8,193, T = 128, r = 64 gives short blocks of 64 and 65, so only
    63 full blocks remain while the first update needs 64.
  - For the character dataset **every** offset is valid at T ∈ {64, 128, 256} (at most 333
    short-block targets against a remainder of 4,429, checked by Claude). The allowed range
    can only narrow for some BPE token counts or other N; the report states the actual
    allowed range for each configuration.
- **Why not the audit's split-window option:** it is also exact, but it needs within-window
  cursors, the same window recomputed across an optimizer step, and more resume state. The
  chosen rule keeps whole windows and exact once-per-epoch coverage. Its only cost is that
  ≤ 2 short blocks are always trained in the epoch's last update: at most **0.0077% /
  0.0204% / 0.0332%** of an epoch's character targets at T = 64 / 128 / 256. The **actual
  fraction is reported per configuration**. The random offset still moves **which text**
  those blocks contain every epoch.
- **Context lengths T that do not divide 8,192** (e.g. D1's matched-context character run):
  each update takes ⌊8,192 / T⌋ full blocks, a **nominal** budget of ⌊8,192/T⌋·T targets.
  Its actual targets per update and number of updates are reported. The exact-8,192 claim
  is limited to T ∈ {64, 128, 256} and to BPE runs at T = 128.
- **Scope:** D12a's exact-target rule applies to **causal-LM runs only**. D24c covers the
  auxiliary objectives.
- **Sampler state (D13a):** the epoch's block boundaries, the shuffled order of full blocks,
  and the cursor (next update index) are saved. Since windows are never split, no
  within-window cursor is needed.
- **Tests:** every eligible target is selected exactly once per epoch; updates 1…n−1
  contain exactly B targets; the last update contains the remainder; target counts **just
  below, equal to and just above** an update boundary (e.g. N = 8,191 / 8,192 / 8,193 at
  T = 128), including the case with no final update; offsets violating the guard are never
  chosen; resuming in the middle of an epoch reproduces the same path (D13e).

#### 24b. One stable scoring population (amends D1, D3b, D13, D15a–b, D17c)
- **Fit the experimental BPE tokenizer early** (milestone M1, training text only). Then,
  **before the grid**, freeze one common raw-text start offset *b* for each split (the end of
  the first BPE token in that split), recorded in `data/manifest.json` together with an
  **evaluation-protocol ID** (stride, *b*, split hashes, scoring rule).
- **Every** reported held-out number uses targets from *b* onward: character models, BPE,
  n-gram baselines, `best.pt` selection, D17 ranking and the final test report. This way all
  selection and comparison metrics refer to the same set of targets. An "all character
  targets" figure may be reported only under a separate label.
- **Windows are defined by absolute target indices.** Each window scores only eligible
  targets **not scored before**. "Last 32 targets" is shorthand for full strides. The final,
  partial stride scores only its actual new targets.
- **Tests:** sequences shorter than T; lengths that are exact multiples of the stride and
  ones with remainders; T ∈ {64, 128, 256}; and every eligible target from *b* scored
  exactly once, with the **same character denominator** for character and BPE models.
- **What the BPE tokenizer file stores (amends D1):** the regex pattern string **and its
  flags**, the byte↔token mapping, the merges ordered by rank, the tie-break rule and a
  format version. A checkpoint must be able to rebuild its tokenizer with no code defaults.
- **Prerequisite for the config printer (amends D20c):** update counts that depend on the
  dataset are printed only when the tokenizer and data metadata already exist. Otherwise
  they are shown as "unresolved", never guessed.

#### 24c. Auxiliary objectives: leakage and MLM runs (amends D6e, D12a, D16b)
- **Objective modes** in the training code: `causal` (default and the only mode used for
  shipping), `leak` (no causal mask, next-token targets) and `mlm`.
- **Padding:** without the causal mask, real queries can attend to right-padding, so
  `leak` and `mlm` pass a **key-padding mask**. Padded queries and targets are excluded from
  every loss. Pads are never chosen for MLM corruption. D6e's "no padding mask needed"
  argument applies **only to causal mode**. A test checks right-padding invariance in
  bidirectional mode **with** the key mask.
- **MLM accounting:** the same windows and updates are used, with an input budget of
  **8,192 input positions per update** (not 8,192 targets). Loss is normalized by the number
  of **selected reconstruction targets** in that update, and those counts are reported.
- **Matched leakage control:** the `leak` run and its causal reference are **both trained
  fresh** for 5 epochs with the **same** 5-epoch schedule (warmup is 5% of their updates,
  cosine decay to 0.1×), the same optimizer, batches, seed and exposure. Only the mask
  differs. A checkpoint taken at epoch 5 of a 30-epoch run is **not** a matched control.

#### 24d. The single BPE promotion rule (replaces D1's seed clause; amends D15e, D16e, D17)
*Problem:* D1 still used "about 2× the seed std, then more seeds", which D17 had rejected,
and D16e's 3 BPE seeds could be compared against a character finalist with fewer seeds.
**This rule is the only one.** D1, D15e, D16e and D17 point here.
1. **Screen:** BPE seed 0 (D16e).
2. **Confirmation, only for a serious promotion attempt:** BPE and the **selected character
   finalist** (D17c) each have the **same three training seeds {0, 1, 2}**. Any of those
   runs that are missing are trained, **at most 4 extra full runs** (2 BPE + up to 2
   character), even if the general ablations used a smaller seed tier.
3. **BPC gate:** BPE's 3-seed mean validation BPC (protocol from 24b) must be **lower by at
   least 0.02 BPC** than the character finalist's. Paired per-seed differences are
   reported. This is an **engineering threshold, not a significance test**. It matches
   D17's 0.02 tie band: a difference inside the band is not a reason to switch tokenizer.
4. **Human gate:** D15e, run once, on the actual proposed checkpoints.
5. **Deployment gates:** D17c eligibility (latency and memory), measured with the **trained**
   BPE checkpoint (D18c BPE caveat).
6. **"No decoding errors"** means **zero errors observed on the frozen D15 sample suite**.
   Finite testing cannot guarantee that every future byte sequence will be valid.
- If the confirmation budget is not available, or any gate fails, the **character finalist
  ships**.
- *Note:* this is **stricter** than D1's original "any lower BPC". It was the audit's
  suggestion, adopted for consistency with D17, and **approved by the team on
  2026-10-10**, before any BPE result.

#### 24e. Final-candidate eligibility (amends D17a, D17c, D18a)
- **Candidate set:** C0; **plus every already-trained sinusoidal character ablation
  configuration** (D16a) at no retraining cost, ranked using its runs at the selected
  uniform tier and reused only under D11's exact-match rule; **plus at most 2 new combined
  configurations** (D17b). **Zero new combinations is allowed.**
- Every candidate is ranked on the **same seed tier**.
- **C0 is not eligible automatically:** it must pass the same gates. If **no** candidate
  passes the deployment gates, the report says so plainly and D18's contingency applies
  (smaller existing shapes, then another free host). The project never quietly ships a
  model that fails the gates.
- D18's fallback ("a smaller ablation shape, e.g. depth 2") is therefore a valid candidate,
  not an exception.

#### 24f. Mixed precision and skipped updates (amends D10c, D12c–d)
- **The LR schedule is indexed by attempted updates,** not successful ones. The
  **scheduled** LR always reaches `min_lr` at the end of the fixed number of epochs, and
  data exposure stays the main budget. If the final attempt is skipped, the last
  **applied** update used a slightly higher LR; the report states which. Data is never replayed and training is never extended to make up for a skipped
  update.
- Attempted and successful updates, skip count and final LR are logged and reported.
- **Numerical failure:** skip rate = skipped ÷ attempted updates, counted **after the first
  100 attempted updates** (GradScaler calibration, which is excluded). A rate above 1%
  stops the run.
- **Precision is consistent across formal comparisons.** An AMP failure leads to one
  documented precision decision that applies to the **whole comparison set** (grid,
  ablations, candidates), with affected runs rerun. One architecture is never quietly
  switched to fp32 on its own, and architectures never get different hidden extra
  exposure.
- AMP mechanics follow PyTorch's guidance: accumulate at one gradient scale, unscale
  **once** per effective update, then D10d's clipping order.
- All of this applies only if the pilot adopts AMP. With fp32, nothing changes.

#### 24g. Model artifacts for build, CI and pilot (amends D13d, D18b, D20a, D21c)
- **The shared downloader is `src/artifacts.py`** (`python -m src.artifacts …`), replacing
  `scripts/download_checkpoint.py`. Every artifact is verified by SHA-256 and checked by
  the same loader and schema checks (D13c).
- **The Docker build takes two required arguments:** `MODEL_SOURCE` (the pinned Release
  URL **or** a local file in the build context) and `MODEL_SHA256`. **CI and the latency
  pilot** use a generated, clearly labelled **test/random-weight artifact** given as a
  local file. **Production** uses the pinned Release URL.
- A failed or mismatched production download **fails the build**. There is never a
  fallback to random or test weights. `/health` and `/model-info` report the artifact's
  label and hash.

#### 24h. Serving limits vs scientific generation; pilot protocol (amends D15c, D18c–e, D19c)
- **D15's 300-character samples are generated offline** with the shared engine and D15c's
  documented safety cap. **They never go through the API's 60-second deadline.** If a
  deployed 300-character throughput run times out, it is reported as a timeout or partial
  result, never as a completed sample.
- **Recorded before the pilot runs:** the exact full-window prompts (stored in
  `configs/prompts.yaml`), seeds, warm-up count, p95 method (nearest-rank over all timed
  requests), and timing boundaries (from the worker starting the request to the last
  token; queueing and network excluded, and reported separately).
- The gate stays **provisional** (100 characters warm, p95 ≤ 30 s, the worse of the two
  prompt sets) until the pilot, and is then **re-measured on the actual proposed shipping
  checkpoints**.
- A local demo backup **supports** the demo but does **not** meet the handout's
  requirement for a hosted service.

#### 24i. Run ledger and compute reservation (amends D10e, D16 inventory and cut order, D17e)
- **D10e finalists, made explicit:** the **2** grid pairs with the lowest seed-0
  validation BPC get seeds 1 and 2 (4 runs). The pair with the lower 3-seed mean is
  selected. Grid + finalists = **13** runs. There are no open-ended extra confirmations.
- **D16's "34–43" is a *preferred-tier subtotal*, not a ceiling.** With the 4 optional
  runs and up to 4 BPE-promotion runs (24d), the upper end is **51 full runs**, before
  short runs, pilots of the auxiliary objectives, or retries.
- **Run ledger** (`outputs/run_ledger.csv`, committed): planned vs completed counts for grid
  finalists, uniform seed tier, number of candidates, BPE confirmations, pilots and short
  controls.
- **Reserve order for free compute:** (1) the grid, the required ablations at the chosen
  tier, and one working final checkpoint plus the submission checks; (2) 16b–16e; (3) new
  combined candidates; (4) optional extras.
- Any lower uniform seed tier is chosen **in advance**, never after looking at which
  variants seem promising.
- **"Never cut" is softened:** free Colab access is not guaranteed. If even the minimum
  scope does not fit, that scope is **amended explicitly in this log**, with the reason,
  never silently.

#### 24j. Evidence archive for figures (amends D22c, D23)
- These **inputs** are saved (with run, config, checkpoint and protocol IDs) under
  `outputs/evidence/`, committed if small: the hand-calculation matrices, mask matrices,
  shape-trace printouts, positional-encoding arrays, the model summary, metrics, the
  trained attention arrays shown in Figure 7 with their token labels, the generated
  samples, and the **actual CLI/API transcripts and screenshots**.
- `make_figures` rebuilds plots **from these inputs only**. If an input is missing it
  **fails clearly**, and it never makes a placeholder. Screenshots of real runs stay as
  captured, as the handout requires; they are never regenerated.
- `best.pt` and `model.pt` hold enough config, tokenizer and provenance information to load
  **on their own** (optimizer state is not needed).
- **Samples and human ratings record the SHA-256 of the exact artifact that was assessed.**

#### 24k. Verification amendments (amends D21)
- **Gradient causality test:** with dropout off, the loss is built **only from the
  unchanged prefix logits and fixed prefix targets**. Its gradients are identical before and
  after the future tokens change. A full-sequence loss could legitimately change gradients
  of shared parameters. Also, ∂(prefix scalar)/∂(future input embeddings) = 0.
- **Direct shift test:** for a known stream and every window starting at `start`,
  `x[t] == stream[start+t]` and **`y[t] == stream[start+t+1]`** for all t, including the
  last position (comparing with `x[t+1]` cannot check the last position). Every target
  index must also be covered. The overfit test alone does not prove correct
  shifting, because an identity task would also overfit.
- **Acceptance:** `python -m experiments.acceptance --fresh` builds a **new virtual
  environment from a clean clone** of the commit (or runs inside the CI container) before
  checks 1–7. When run inside an existing environment, it marks check 1 **SKIPPED**, not
  PASS. Any failure gives a **non-zero exit code**, and skipped checks are labelled.
- **README:** running the tagged blocks shows that **those blocks work under the tested
  conditions**. It does not prove the README can never drift.
- **CLI:** with `--json`, stdout contains **only JSON** and diagnostics go to stderr.
  Cached artifacts are re-checked against their SHA-256. A missing `--checkpoint` path
  gives a clear error (exit 1) and never falls back to the release download.
- **Unit test runtime:** "< 60 s" is a **target calibrated on the development laptop** and
  then frozen as a CI warning threshold. It is not a promise for every machine.

#### 24l. Operations amendments (amends D20b, D22)
- **Notebook order:** §0 first **chooses the runtime path**: CPU (install
  `serve-cpu.txt` for quick inference; §2's tests install **`dev-cpu.txt`**; no GPU check,
  no Drive) or CUDA (install `colab-cuda.txt`). A GPU is
  required **only when `RUN_TRAINING = True`**. Drive is mounted **only** for training or
  resuming. §1 inference works on either path.
- **Resume needs an explicit run ID or checkpoint path.** The checkpoint is checked for
  compatibility (D20c). Each resume becomes a numbered **attempt** in the manifest.
- **Recovery:** metric and sample events written **after the checkpoint being resumed
  from** are kept in the logs but marked `abandoned` (by attempt ID). They are left out of
  every aggregate and figure.
- **`outputs/runs.csv` is a derived index:** `python -m experiments.index` rebuilds it from
  the run manifests, and it is never edited by hand.

#### 24m. In-place cleanups made with this entry
- D5: tying is shown as resolved (D8). D6/D7/D9/D11: stale "open/pending" items point to
  the decisions that resolved them.
- D9: the rejection of a "convention" dropout rate is reworded. The handout asks for a
  **justified** configuration, not necessarily a sweep. The sweep is the team's chosen
  evidence.
- D10b: the optional trainable position table (D16f) goes in the **no-decay** group, like
  the token embedding.
- D14d: "freshly encoded" becomes "freshly recomputed from the current token-ID window"
  (D14c: encode once, never re-tokenize).
- D15e: the t-interval is a **benchmark acceptance convention**. The coverage simulation
  assumed normal data, so it does not establish coverage for 7 hand-picked prompts with
  bounded author ratings. The simulation script is kept at `docs/analysis/ci_coverage_sim.py`.
  The mean condition is implied by the interval condition when both use the same margin,
  but keeping both does no harm.
- D19: the summary is fixed (admission is on the HTTP side). For streaming, the request
  waits for **worker acquisition** (not just a queue slot) before the response headers are
  sent, so a queue timeout still returns HTTP 503. Equality tests leave out the timing
  fields. A disconnected client is not promised a `done` event. 10 s + 60 s is an
  **application budget**, not a hard end-to-end guarantee: it excludes waking the service
  and the network, and it can overrun slightly while a tensor operation finishes.
- D23: the recorded demo also aims for **5–7 minutes**, which satisfies both Section 4 and
  Section 17. Report length is judged on the rendered article and PDF, figures included;
  the word count is only a guide.

**Depends on:** D1–D23.

### D25. Deadline amendment: submission in ~50 hours

**Status:** Accepted · 2026-10-10 · **Wins over D1–D24 wherever they conflict.** The team set
completion as the main goal; studying the code in depth comes after submission. D1–D24
remain the full design. D25 is the subset built now, and the report names what was cut.

**Kept (everything the handout grades):** character tokenizer (D1 core), the pinned and
verified data split (D2), the D4–D8 architecture as designed (Pre-LN, scaled embeddings plus
sinusoidal PE, manual multi-head attention, GELU FFN, GPT-2 init, untied head), AdamW +
warmup/cosine + clipping (D10, at fixed LR 1e-3 and dropout 0.1), baseline D11, the required
ablations (depth, heads, context: 6 extra runs, **seed 0 only, labelled exploratory**), a short
leakage (no-causal-mask) demonstration for the MLM-vs-CLM evidence, an LSTM-vs-attention timing,
greedy vs temperature sampling, the hand calculation (D23a), handout tests, CLI, FastAPI on
Render Free, the Colab notebook, the figures, README, Medium article, demo video and slides.

**Simplified:**
- Training is **step-based with random contiguous windows** (nanoGPT style), with a
  **fixed step budget** shared by all runs and periodic evaluation. This replaces D3a/D24a's
  epoch packing.
- Validation uses **non-overlapping windows covering the whole split**. The report notes the
  low-context bias, and a stride evaluation can be added later.
- **fp32** only; `best.pt` is chosen by validation loss; there is a simple `last.pt`
  resume; the checkpoint is published as a GitHub Release asset with its SHA-256.
- API: `/health`, `/model-info` and `/generate` (no streaming), with a lock so only one
  generation runs at a time and a cap on `max_new_tokens`, plus a minimal HTML page.
- CI: pytest on CPU through GitHub Actions.

**Cut until after submission:** the LR × dropout grid, extra seeds, BPE and its promotion
rule, n-gram baselines, the blind human rating, the latency pilot, SSE streaming,
lock files beyond `requirements.txt`, the full MLM training run, and the optional extras.
The report presents these as limitations or future work and never implies they were done.
