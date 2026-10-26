# PACE (DSAI6707 course delivery)

**Primary write-up:** [`reports/PACE_Course_Report.md`](reports/PACE_Course_Report.md) · **6-min talk:** [`reports/PACE_Presentation_6slides.md`](reports/PACE_Presentation_6slides.md)  

Experiments are **frozen** (no new algorithms / GPU runs for delivery).  
DICE-PO negatives stay historical under `outputs/gate1/` and `reports/DICE_PO_Technical_Report.md`.

## Status

| Track | Gate | Status |
|------|------|--------|
| DICE (legacy) | Gate 1 | NO-GO (retained, not in main story) |
| **PACE** | Gate 0 / 1 | pass / GO |
| **PACE** | Gate 2 **15-seed primary** | `outputs/pace/gate2_cap_align_15seed/` |

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

## Gate 2 (PACE online)

Requires **PACE Gate 1 GO** (`outputs/pace/gate1/PACE_GO_NOGO.md`). Survivor-only contrast under shared Paired Racing: **B0** `point_estimate_truncate` vs **PACE** `pace_bounded_survive`. Winner from search observations only, then sealed test. Writes only under `outputs/pace/gate2/` (does not touch `outputs/gate1`, legacy `outputs/gate2`, or `reports/`).

```bash
export HF_HOME=/home3/river/DSAI6707/DICE_PO/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
PY=/home3/river/comfyui-minimax-h3/.venv/bin/python

# Smoke (small gen/budget)
PYTHONPATH=src $PY scripts/run_pace_gate2.py --smoke --device cuda:0

# Full (pop=8, gen=4, seeds=42/123/456, calls_cap=1280)
PYTHONPATH=src $PY scripts/run_pace_gate2.py --device cuda:0
```

Config: `configs/pace_gate2.yaml` (ε=0, n_min=8, B=200, μ=8, K=8).

Legacy EC stub (historical, not PACE): `scripts/run_online.py --go` → `outputs/gate2/`.

## Layout

See plan §6. Policies implement `BudgetPolicy.select_queries(state, remaining_calls)`.
