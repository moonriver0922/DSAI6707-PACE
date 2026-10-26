# PACE Gate 2 Cold Start

Experiment: `gate2_cold` · Smoke: `False` · eval=`racing`

- μ=8, K=8, ε=0.0, n_min=8, B=200
- pop=8, gen=4, calls_cap=1280, seeds=[42, 123, 456]
- gate1_warm_cells=0
- disable_confidence_gate methods: none

## Method means

- **b0_point_nsga**: test_acc=0.7867, test_cost=111.9, calls=1280, conf_elim=0.0, forced=0.0 (n=3)
- **pace**: test_acc=0.8017, test_cost=112.1, calls=1271, conf_elim=1.3, forced=14.7 (n=3)

## Within-run Δ vs B0 (this experiment)

- **pace − B0**: Δacc=+0.0150, Δcost=+0.2, Δcalls=-9

## Δ vs original `outputs/pace/gate2`

- **b0_point_nsga**: Δacc=+0.0033 (this 0.7867 vs gate2 0.7833), Δcost=-2.7, Δcalls=+11
- **pace**: Δacc=-0.0083 (this 0.8017 vs gate2 0.8100), Δcost=-3.4, Δcalls=-9

## Runs (per seed)

| method | seed | calls | test_acc | test_cost | conf_elim | forced | Δacc_vs_gate2 | Δcalls_vs_gate2 |
|--------|------|-------|----------|-----------|-----------|--------|--------------|-----------------|
| b0_point_nsga | 42 | 1280 | 0.7700 | 105.3 | 0 | 0 | -0.0100 | +10 |
| pace | 42 | 1260 | 0.8300 | 118.7 | 1 | 15 | -0.0250 | -20 |
| b0_point_nsga | 123 | 1280 | 0.8000 | 113.9 | 0 | 0 | +0.0000 | +10 |
| pace | 123 | 1272 | 0.8000 | 113.9 | 1 | 15 | +0.0000 | -8 |
| b0_point_nsga | 456 | 1280 | 0.7900 | 116.6 | 0 | 0 | +0.0200 | +12 |
| pace | 456 | 1280 | 0.7750 | 103.8 | 2 | 14 | +0.0000 | +0 |

Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. Original `outputs/pace/gate2/` not overwritten.
