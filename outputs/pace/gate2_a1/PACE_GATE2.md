# PACE Gate 2 A1 Ablation (gate off, K=0)

Experiment: `gate2_a1` · Smoke: `False` · eval=`racing`

- μ=8, K=0, ε=0.0, n_min=8, B=200
- pop=8, gen=4, calls_cap=1280, seeds=[42, 123, 456]
- gate1_warm_cells=1280
- disable_confidence_gate methods: ['a1_no_gate']

## Method means

- **b0_point_nsga**: test_acc=0.7833, test_cost=114.7, calls=1269, conf_elim=0.0, forced=0.0 (n=3)
- **a1_no_gate**: test_acc=0.7867, test_cost=112.3, calls=1280, conf_elim=0.0, forced=24.0 (n=3)

## Within-run Δ vs B0 (this experiment)

- **a1_no_gate − B0**: Δacc=+0.0033, Δcost=-2.4, Δcalls=+11

## Δ vs original `outputs/pace/gate2`

- **b0_point_nsga**: Δacc=+0.0000 (this 0.7833 vs gate2 0.7833), Δcost=+0.0, Δcalls=+0
- **a1_no_gate vs gate2 B0**: Δacc=+0.0033, Δcost=-2.4, Δcalls=+11 (A1 should track B0)

## Runs (per seed)

| method | seed | calls | test_acc | test_cost | conf_elim | forced | Δacc_vs_gate2 | Δcalls_vs_gate2 |
|--------|------|-------|----------|-----------|-----------|--------|--------------|-----------------|
| b0_point_nsga | 42 | 1270 | 0.7800 | 112.6 | 0 | 0 | +0.0000 | +0 |
| a1_no_gate | 42 | 1280 | 0.7800 | 110.8 | 0 | 24 | +0.0000 | +10 |
| b0_point_nsga | 123 | 1270 | 0.8000 | 114.9 | 0 | 0 | +0.0000 | +0 |
| a1_no_gate | 123 | 1280 | 0.7950 | 109.5 | 0 | 24 | -0.0050 | +10 |
| b0_point_nsga | 456 | 1268 | 0.7700 | 116.5 | 0 | 0 | +0.0000 | +0 |
| a1_no_gate | 456 | 1280 | 0.7850 | 116.6 | 0 | 24 | +0.0150 | +12 |

Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. Original `outputs/pace/gate2/` not overwritten.
