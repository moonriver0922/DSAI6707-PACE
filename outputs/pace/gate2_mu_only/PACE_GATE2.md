# PACE Gate 2 Mu-Only (K=0)

Experiment: `gate2_mu_only` · Smoke: `False` · eval=`racing`

- μ=8, K=0, ε=0.0, n_min=8, B=200
- pop=8, gen=4, calls_cap=1280, seeds=[42, 123, 456]
- gate1_warm_cells=1280
- disable_confidence_gate methods: none

## Method means

- **b0_point_nsga**: test_acc=0.7833, test_cost=114.7, calls=1269, conf_elim=0.0, forced=0.0 (n=3)
- **pace**: test_acc=0.7817, test_cost=117.6, calls=1277, conf_elim=1.0, forced=23.0 (n=3)

## Within-run Δ vs B0 (this experiment)

- **pace − B0**: Δacc=-0.0017, Δcost=+2.9, Δcalls=+8

## Δ vs original `outputs/pace/gate2`

- **b0_point_nsga**: Δacc=+0.0000 (this 0.7833 vs gate2 0.7833), Δcost=+0.0, Δcalls=+0
- **pace**: Δacc=-0.0283 (this 0.7817 vs gate2 0.8100), Δcost=+2.0, Δcalls=-3

## Runs (per seed)

| method | seed | calls | test_acc | test_cost | conf_elim | forced | Δacc_vs_gate2 | Δcalls_vs_gate2 |
|--------|------|-------|----------|-----------|-----------|--------|--------------|-----------------|
| b0_point_nsga | 42 | 1270 | 0.7800 | 112.6 | 0 | 0 | +0.0000 | +0 |
| pace | 42 | 1280 | 0.7800 | 110.8 | 1 | 23 | -0.0750 | +0 |
| b0_point_nsga | 123 | 1270 | 0.8000 | 114.9 | 0 | 0 | +0.0000 | +0 |
| pace | 123 | 1272 | 0.8000 | 129.6 | 0 | 24 | +0.0000 | -8 |
| b0_point_nsga | 456 | 1268 | 0.7700 | 116.5 | 0 | 0 | +0.0000 | +0 |
| pace | 456 | 1280 | 0.7650 | 112.3 | 2 | 22 | -0.0100 | +0 |

Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. Original `outputs/pace/gate2/` not overwritten.
