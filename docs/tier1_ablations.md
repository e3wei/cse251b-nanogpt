# Tier-1 ablation studies

This document describes each **Tier-1** experiment in `config/tier1_ablation/`. The goal is to change **one major axis** at a time while keeping data, batching, default training length, and (except where noted) model family fixed, so you can attribute differences in validation perplexity to that axis.

**Control run:** `control_current.py` — this is the reference. Every other config is meant to be compared against it.

**How to run:** From the repo root, use `scripts/run_tier1_ablation.ps1` or the wrapper `ablation.ps1`, or call `python train.py config/tier1_ablation/<name>.py` directly. See [README.md](../README.md#tier-1-ablation-configs-control--o1o2a1r1) for CLI details.

---

## Summary table

| Config file | Short name | What you isolate |
|-------------|------------|------------------|
| `control_current.py` | Control | Baseline recipe (reference) |
| `o1_muon_adamw.py` | O1 | Optimizer: AdamW → Muon + AdamW |
| `o2_wsd.py` | O2 | Learning-rate schedule: cosine → WSD |
| `a1_depth_width_12x640.py` | A1a | Architecture: depth/width (12 layers × 640 dim) |
| `a1_depth_width_16x576.py` | A1b | Architecture: depth/width (16 layers × 576 dim) |
| `r1_fixed_1024.py` | R1 | Training: sequence-length curriculum → fixed 1024 |

---

## Control — `control_current.py`

**Purpose:** Establish the baseline you compare all ablations to.

**What it does:**

- **Model:** 22 transformer blocks, hidden size 512, 8 heads (64-dim heads), SwiGLU FFN (`ffn_mult=2.5`), RMSNorm, RoPE base 10 000, QK-norm on, no biases on linear layers.
- **Optimizer:** AdamW with peak LR `3e-4`, `min_lr=3e-5` after decay, weight decay 0.1, β₁=0.9, β₂=0.95, global grad clip 1.0.
- **Schedule:** Warmup 200 steps, then **cosine decay** from peak LR down to `min_lr` over `lr_decay_iters=12000`, for `max_iters=12000`.
- **Sequence-length curriculum:** Training batches use context length 256 → 512 → 768 → 1024 at fixed iteration boundaries (see config), capped by `block_size=1024`.
- **Data / batching:** `train.bin` / `val.bin` in `.`, batch size 8, gradient accumulation 16 (effective batch in tokens scales with the active sequence length).

**Default checkpoint directory:** `out_tier1_control_current` (or `out_tier1_quick_control_current` if you use the `quick` profile in `run_tier1_ablation.ps1`).

---

## O1 — Optimizer ablation — `o1_muon_adamw.py`

**Question:** Does a **Muon + AdamW hybrid** improve sample efficiency or final perplexity versus **AdamW alone**, holding architecture and LR schedule fixed?

**What changes vs control:**

- **`optimizer_type = "muon_adamw"`** in `train.py`: roughly, **2D weight matrices inside transformer blocks** are updated with **Muon** (orthogonalized, momentum-style updates); **everything else** (e.g. embeddings, norms, 1D parameters) stays on **AdamW** with the same hyperparameters as the control.
- Muon-specific settings: `muon_lr=0.02`, `muon_momentum=0.95`, `muon_ns_steps=5` (Newton–Schulz steps in the Muon implementation).

**What stays the same:** Model shape (22×512), cosine LR schedule, warmup/decay lengths, sequence-length curriculum, data and batching, seed.

**Default checkpoint directory:** `out_tier1_o1_muon_adamw` (or quick-profile equivalent).

---

## O2 — Learning-rate schedule ablation — `o2_wsd.py`

**Question:** Does a **warmup–stable–decay (WSD)** schedule beat **cosine decay** for the same step budget and optimizer?

**What changes vs control:**

- **`lr_schedule = "wsd"`** instead of `"cosine"`.
- **WSD shape:** After **200 warmup** steps, LR stays at **`learning_rate`** (stable phase), then **linearly decays** to **`wsd_final_lr = 0`** over the **last 20%** of training (`wsd_cooldown_frac = 0.2`), i.e. the final ~2400 steps of 12 000.
- **`min_lr` is not used** for the WSD path in the current implementation; the floor is `wsd_final_lr`.

**What stays the same:** AdamW, model, data, batching, sequence-length curriculum, `max_iters=12000`, seed.

**Default checkpoint directory:** `out_tier1_o2_wsd` (or quick-profile equivalent).

---

## A1 — Depth / width (two points) — `a1_depth_width_12x640.py` and `a1_depth_width_16x576.py`

**Question:** For a **similar parameter budget** under the 100M cap, is it better to be **deeper and narrower** (control: 22×512), **shallower and wider** (12×640), or **in between** (16×576)?

**What changes vs control:**

| Variant | Layers (`n_layer`) | Hidden size (`n_embd`) | Heads (`n_head`) |
|---------|-------------------|------------------------|------------------|
| Control | 22 | 512 | 8 |
| A1a (`a1_depth_width_12x640.py`) | 12 | 640 | 10 |
| A1b (`a1_depth_width_16x576.py`) | 16 | 576 | 9 |

Head count follows `n_head = n_embd // 64` so head dimension stays 64.

**What stays the same:** Optimizer (AdamW), cosine schedule, sequence-length curriculum, FFN and norm settings, data and batching, seed.

**Note:** Total parameter counts differ slightly between shapes; treat this as a **practical depth-vs-width tradeoff** study, not a perfectly iso-param sweep unless you add further width/layer tweaks.

**Default checkpoint directories:** `out_tier1_a1_12x640`, `out_tier1_a1_16x576` (or quick-profile equivalents).

---

## R1 — Sequence-length curriculum ablation — `r1_fixed_1024.py`

**Question:** Does **ramping context length** (256 → 1024) help compared with **training at full context from step zero**?

**What changes vs control:**

- **`seq_len_schedule = [(0, 1024)]`**: every training step uses sequence length **1024** (subject to `block_size`), so there is **no** short-context warm-up phase.

**What stays the same:** Model (22×512), AdamW, cosine LR schedule, data files, batch size and accumulation (note: **tokens per optimizer step** are higher from the start than in the control’s early phase, so wall-clock and compute per step differ early in training).

**Default checkpoint directory:** `out_tier1_r1_fixed1024` (or quick-profile equivalent).

---

## Interpreting results

- Compare **validation loss / perplexity** (from `evaluate.py` or in-training `val_loss`) **at the same training progress** (e.g. same `iter` or same tokens seen) when possible.
- **Quick profile** runs in `scripts/run_tier1_ablation.ps1` use fewer iterations and a separate `out_dir`; use them to **rank ideas**, then rerun winners with the **full** configs for report-quality numbers.
- Prefer evaluating on a **true held-out** bin (e.g. `provided_val.bin`) if your `val.bin` might overlap `train.bin`.

---

## File map

| Experiment | Config path |
|------------|-------------|
| Control | `config/tier1_ablation/control_current.py` |
| O1 | `config/tier1_ablation/o1_muon_adamw.py` |
| O2 | `config/tier1_ablation/o2_wsd.py` |
| A1 (12×640) | `config/tier1_ablation/a1_depth_width_12x640.py` |
| A1 (16×576) | `config/tier1_ablation/a1_depth_width_16x576.py` |
| R1 | `config/tier1_ablation/r1_fixed_1024.py` |
