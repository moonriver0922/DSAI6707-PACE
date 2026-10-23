# DICE-PO / PACE

DSAI6707 — **current direction: PACE** (`PACE_Research_Implementation_Plan.md`).  
DICE-PO results are **historical negatives**, frozen under `outputs/gate1/` and `reports/`.

## Status

| Track | Gate | Status |
|------|------|--------|
| DICE (legacy) | Gate 1 + fair ablation | **NO-GO** (negative result retained) |
| **PACE** | Gate 0 | **pass** (`pytest tests/pace`) |
| **PACE** | Gate 1a/1b (0 LLM) | **GO** — see `outputs/pace/gate1/PACE_GO_NOGO.md` |
| **PACE** | Gate 2 online | Conditional; not started |

See `PROJECT_DIRECTION.md`.

## Setup

```bash
cd /home3/river/DSAI6707/DICE_PO
# CPU / Gate 0 env (already created at .conda)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home3/river/DSAI6707/DICE_PO/.conda

# Keep HF downloads on /home3 (root disk is full):
export HF_HOME=/home3/river/DSAI6707/DICE_PO/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# GPU runtime used for smoke (torch+CUDA already available):
#   /home3/river/comfyui-minimax-h3/.venv/bin/python
```

Frozen assets already present:

- `data/prompt_pool.json` — 16 prompts
- `data/split_manifest.json` — 160 search + 200 test AG News IDs (seed 6707)

## Gate 0

```bash
.conda/bin/pytest
```

## GPU smoke

```bash
export HF_HOME=/home3/river/DSAI6707/DICE_PO/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
PYTHONPATH=src /home3/river/comfyui-minimax-h3/.venv/bin/python scripts/smoke_gpu.py --device cuda:0
```

## Gate 1 (after GPU smoke)

```bash
# One-shot precompute 16 × (160+200) evaluations (GPU python)
PYTHONPATH=src /home3/river/comfyui-minimax-h3/.venv/bin/python \
  scripts/precompute_matrix.py --device cuda:0

# Offline replay (CPU conda ok; no new LLM calls)
.conda/bin/python scripts/run_replay.py

# Pipeline dry-run without a matrix:
.conda/bin/python scripts/run_replay.py --synthetic --smoke
```

Primary metric: **test preference selection regret** at budget **640**.

## Gate 2

Do **not** run real online search until Gate 1 go/no-go. The stub requires `--go`:

```bash
.conda/bin/python scripts/run_online.py --go
```

## Layout

See plan §6. Policies implement `BudgetPolicy.select_queries(state, remaining_calls)`.
