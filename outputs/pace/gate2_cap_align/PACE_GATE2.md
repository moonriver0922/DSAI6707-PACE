# PACE Gate 2 Cap-Align (B0+Archive vs PACE, μ+K=16)

Experiment: `gate2_cap_align` · Smoke: `False` · eval=`racing`

- μ=8, K=8, ε=0.0, n_min=8, B=200
- pop=8, gen=4, calls_cap=1280, seeds=[42, 123, 456]
- gate1_warm_cells=1280
- disable_confidence_gate methods: ['b0_archive']

## Method means

- **b0_archive**: test_acc=0.7617, test_cost=113.0, calls=1277, conf_elim=0.0, forced=16.0 (n=3)
- **pace**: test_acc=0.8100, test_cost=115.5, calls=1280, conf_elim=3.0, forced=13.0 (n=3)

## Within-run Δ vs b0_archive (this experiment)

- **pace − b0_archive**: Δacc=+0.0483, Δcost=+2.6, Δcalls=+3

## Δ vs original `outputs/pace/gate2`

- **pace**: Δacc=+0.0000 (this 0.8100 vs gate2 0.8100), Δcost=+0.0, Δcalls=+0

## Runs (per seed)

| method | seed | calls | test_acc | test_cost | conf_elim | forced | Δacc_vs_gate2 | Δcalls_vs_gate2 |
|--------|------|-------|----------|-----------|-----------|--------|--------------|-----------------|
| b0_archive | 42 | 1280 | 0.8550 | 116.1 | 0 | 16 | — | — |
| pace | 42 | 1280 | 0.8550 | 116.1 | 1 | 15 | +0.0000 | +0 |
| b0_archive | 123 | 1270 | 0.7750 | 107.5 | 0 | 16 | — | — |
| pace | 123 | 1280 | 0.8000 | 119.7 | 3 | 13 | +0.0000 | +0 |
| b0_archive | 456 | 1280 | 0.6550 | 115.3 | 0 | 16 | — | — |
| pace | 456 | 1280 | 0.7750 | 110.8 | 5 | 11 | +0.0000 | +0 |

Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. Original `outputs/pace/gate2/` not overwritten.
