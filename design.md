# CSE 251B NanoGPT — Strong Baseline Recipe (≤100M params)

This recipe targets **minimum perplexity on a hidden test set** under a single-4090 / ~$20 compute budget. Every decision is committed.

---

## 1. Architecture Design

Decoder-only transformer, Llama / modded-nanogpt-style, **weight-tied** embedding ↔ LM head.

| Component | Specification |
|-----------|----------------|
| Layers | **12** |
| d_model | **640** |
| Heads | **10** (head_dim = 64) |
| FFN | **SwiGLU**, d_ff = **1728** (≈ 8/3 · d_model, padded to multiple of 64) |
| Activation | **SwiGLU** (silu·gate) |
| Norm | **RMSNorm**, pre-norm, **QK-norm** before attention |
| Positional encoding | **RoPE**, base = 10000 |
| Biases | **None** (Llama-style) |
| Tying | **Tied** input embedding & output projection |
| Logit softcap | **30** (Gemma-style), tanh softcapped |

**Parameter budget (with weight tying):**

- Embed / LM head: 50257 × 640 ≈ 32.2 M
- Per block: attn (4·d²) + SwiGLU (3·d·d_ff) ≈ 1.64 M + 3.32 M = 4.96 M
- 12 blocks: 59.5 M
- **Total ≈ 91.7 M** (safe margin under 100M)

### Rationale

- **Embeddings dominate at this scale.** The GPT-2 vocab (50257) consumes ~32M parameters at d=640 (and ~38.6M at d=768). Weight tying is non-negotiable; it buys an entire transformer worth of parameters.
- **Depth > width below ~200M.** With ≤100M parameters, more layers consistently lowers perplexity more than wider matmuls in the nanoGPT / Pythia / Chinchilla low-N regime. Twelve layers is a strong sweet spot once embedding cost is paid.
- **head_dim = 64** aligns with FlashAttention / SDPA efficiency on RTX 4090. d_model = 640 with 10 heads is the largest hardware-friendly width that fits 12 layers with margin under the cap.
- **SwiGLU at ~8/3 expansion** is parameter-equivalent to GELU at 4× and typically beats GELU by ~0.05–0.1 nats on FineWeb at this scale.
- **RoPE @ base 10000** matches the competition eval context (1024 tokens). No NTK scaling required.
- **QK-norm + logit softcap** improve bf16 stability and allow higher learning rates without loss spikes.
- **No biases, no dropout** in the architecture removes regularization noise that hurts in the undertrained regime.

---

## 2. Optimizer

**Muon (matrix parameters) + AdamW (embeddings, tied LM head, scalars / norms)** — validated in modded-nanogpt-style workloads.

| | Muon | AdamW |
|---|------|---------|
| Applies to | All 2-D weights in transformer blocks (Q, K, V, O, FFN) | Embedding (= LM head when tied), RMSNorm gains, scalars |
| Learning rate | **0.02** | **3e-4** |
| Momentum / betas | momentum **0.95**, Newton–Schulz steps **5** | β₁ **0.9**, β₂ **0.95** |
| Weight decay | **0** | **0.1** |
| Gradient clipping | **1.0** (global L2) | **1.0** |

### Rationale

- Muon orthogonalizes 2-D update directions and is often **~1.5–2× more sample-efficient** than AdamW alone on nanoGPT-class pretraining.
- AdamW remains better for embedding-like parameters (sparse, row-imbalanced); Muon on embeddings tends to underperform.
- β₂ = 0.95 (not 0.999) suits short pretraining: faster adaptation, stable with QK-norm and softcapped logits.
- Weight decay only on the AdamW path. Muon already constrains scale; extra WD on Muon weights often hurts.
- Global grad clip 1.0 is cheap insurance against rare bf16 spikes.

---

## 3. Learning Rate Schedule

**Trapezoidal: Warmup–Stable–Decay (WSD)** — preferred over naive cosine for **fixed compute** and final perplexity.

- **Warmup:** linear, **1.5%** of total optimizer steps (~150 steps for ~10k steps)
- **Stable phase:** constant LR, **~78.5%** of steps
- **Cooldown:** linear decay to **0** over the **last 20%** of steps
- Apply the same fractional schedule to both Muon and AdamW learning rates (scaled by their respective base LRs).

### Rationale

- WSD often matches or beats cosine in reproducible nanoGPT comparisons; typical final-PPL gain vs cosine: ~0.05–0.10 nats for the same FLOPs.
- A long stable phase plus a dedicated cooldown supports **checkpoint branching** and optional **data annealing** during cooldown without redoing the full pretrain.
- Linear cooldown to **exactly 0** is appropriate when the model is not continued after this run.

---

## 4. Data Strategy

**Single high-quality source for the stable phase; a quality + domain mixture during cooldown** to favor generalization on a hidden, multi-domain test set.

- **Stable phase (~80% of training tokens):** **FineWeb-Edu `sample-10BT`**, document-shuffled, token-packed to sequence length 1024 with `<|endoftext|>` document boundaries.
- **Cooldown phase (~20% of tokens):** switch loader to a **higher-quality mixture** (anneal trick from MiniCPM / modded-nanogpt-style finals):
  - **70%** FineWeb-Edu **top quality decile** (e.g. score ≥ 4)
  - **20%** **DCLM-Edu** (e.g. `sample-3.0T` filtered subset) for complementary distribution
  - **10%** **The Stack v2** small slice (Python + Markdown) to broaden domain coverage
- **Tokenizer:** **GPT-2 BPE** (`tiktoken` encoding `gpt2`, vocab **50257**) — required by competition `evaluate.py`.
- **Do not train on `val.bin`.** Exclude public val shards from training data by hash or split ID during preparation.

### Rationale

- FineWeb-Edu offers strong signal-to-noise for pretraining at this scale.
- Annealing with higher-quality and slightly broader-domain data during LR cooldown is high leverage for final PPL with no extra base-training cost.
- Avoid noisier general web mixes (e.g. raw C4 / RefinedWeb) as the main diet; they tend to hurt perplexity per token at this budget.

---

## 5. Regularization

| Technique | Value |
|-----------|--------|
| Dropout (attention, residual, embedding) | **0.0** |
| Weight decay (AdamW parameters only) | **0.1** |
| Weight decay (Muon parameters) | **0** |
| Label smoothing | **0** |
| Z-loss (auxiliary on logit log-sum-exp) | **1e-4** |
| Stochastic depth / DropPath | **0** |

### Rationale

- Training is **undertrained** relative to Chinchilla-optimal token counts for ~92M non-embedding parameters. In this regime dropout usually **hurts** validation perplexity.
- Weight decay 0.1 on AdamW is a well-tuned default; it matters for controlling embedding scale when tied to the LM head.
- Z-loss 1e-4 stabilizes bf16 logits at high LR with softcapping; negligible PPL cost.
- No label smoothing — it directly increases cross-entropy / perplexity.

---

## 6. Training Tricks — Decisions

| Trick | Use? | Reason |
|-------|------|--------|
| Token packing with EOS between documents | **Yes** | Maximizes useful FLOPs; standard for packed pretraining. |
| Sequence length curriculum | **No** | For final PPL at eval **1024**, train at **1024** throughout to match train and eval geometry. |
| Batch size scheduling | **No** | Small gains; complicates Muon bookkeeping. Use fixed effective batch. |
| Gradient accumulation to ~0.5M tokens / step | **Yes** | Stable optimization; standard for nanoGPT-scale runs. |
| bf16 mixed precision | **Yes** | Throughput; use fp32 master weights for AdamW; Muon in bf16 with fp32 accumulation where implemented. |
| `torch.compile` + FlashAttention / SDPA | **Yes** | Large throughput win; no PPL penalty. |
| Multi-token prediction (MTP) | **No** (baseline) | Real gain possible but adds complexity; first ablation if time allows. |
| Distillation from a larger model | **No** (baseline) | Allowed by rules but adds engineering risk; optional ablation. |
| Document-level attention masking inside packs | **No** | Usually not worth the complexity at 1024 context for this budget. |
| Global gradient clipping (L2 = 1.0) | **Yes** | Safety. |
| EMA of weights for submission | **No** | WSD cooldown already targets a sharp minimum; EMA redundant. |

---

## 7. Final Baseline Config

### Model

- **layers:** 12  
- **d_model:** 640  
- **heads:** 10  
- **FFN:** SwiGLU, d_ff 1728  
- **activation:** SwiGLU (SiLU gate)  
- **positional encoding:** RoPE, base 10000  
- **norm:** RMSNorm, pre-norm + QK-norm  
- **output:** tied embedding ↔ LM head; tanh logit softcap = 30  
- **biases:** none  
- **total parameters:** ~91.7M  

### Training

- **optimizer:** Muon (2-D block weights) + AdamW (embedding / tied LM head, RMSNorm)  
- **lr:** Muon 0.02; AdamW 3e-4  
- **betas / momentum:** Muon momentum 0.95, NS steps 5; AdamW (0.9, 0.95)  
- **schedule:** WSD — 1.5% linear warmup, ~78.5% constant, last 20% linear cooldown to 0  
- **batch size:** ~0.5M tokens per optimizer step (e.g. 512 × 1024 via accumulation)  
- **sequence length:** 1024 (fixed)  
- **precision:** bf16 mixed; fp32 AdamW master  
- **grad clip:** 1.0  
- **target tokens:** ~5B (~10,000 optimizer steps at 0.5M tokens/step)  
- **data:** FineWeb-Edu sample-10BT (stable) → quality + domain anneal mix (cooldown)  

### Regularization

- **dropout:** 0.0  
- **weight decay:** 0.1 on AdamW parameters only  
- **z-loss:** 1e-4  
- **label smoothing:** 0  

---

## Summary: Why This Should Be Competitive

1. **Parameters where they count:** depth (12 layers) + tied embeddings, not excess width.  
2. **Muon + AdamW hybrid:** multiplicative sample-efficiency vs pure AdamW under fixed compute.  
3. **WSD + data anneal in cooldown:** strong lever for final hidden-test PPL without extra base pretraining cost.  
4. **Train/eval alignment:** GPT-2 tokenizer, vocab 50257, context 1024 — matches competition evaluation exactly.  
5. **Minimal harmful regularization:** no dropout / no label smoothing in the undertrained regime.  
6. **Proven stack:** RoPE, SwiGLU, RMSNorm, QK-norm, softcapped logits, FineWeb-Edu — all with independent evidence in nanoGPT-class setups.

Use this document as the **locked baseline**; ablations (MTP, distillation, longer anneal, alternate depth/width at d=576 / n=16) build on top.
