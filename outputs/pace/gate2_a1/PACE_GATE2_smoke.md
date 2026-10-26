# PACE Gate 2 A1 Ablation

Experiment: `gate2_a1` · Smoke: `True` · eval=`racing`

- μ=4, K=0, ε=0.0, n_min=8, B=40
- pop=4, gen=2, calls_cap=256, seeds=[42]
- gate1_warm_cells=640
- disable_confidence_gate methods: ['a1_no_gate']

## Method means

- **b0_point_nsga**: test_acc=0.8250, test_cost=108.9, calls=256, conf_elim=0.0, forced=0.0 (n=1)
- **a1_no_gate**: test_acc=0.8250, test_cost=108.9, calls=256, conf_elim=0.0, forced=4.0 (n=1)

## Within-run Δ vs B0 (this experiment)

- **a1_no_gate − B0**: Δacc=+0.0000, Δcost=+0.0, Δcalls=+0

## Δ vs original `outputs/pace/gate2`

- **b0_point_nsga**: Δacc=+0.0417 (this 0.8250 vs gate2 0.7833), Δcost=-5.8, Δcalls=-1013
- **a1_no_gate vs gate2 B0**: Δacc=+0.0417, Δcost=-5.8, Δcalls=-1013 (A1 should track B0)

## Runs (per seed)

| method | seed | calls | test_acc | test_cost | conf_elim | forced | Δacc_vs_gate2 | Δcalls_vs_gate2 |
|--------|------|-------|----------|-----------|-----------|--------|--------------|-----------------|
| b0_point_nsga | 42 | 256 | 0.8250 | 108.9 | 0 | 0 | +0.0450 | -1014 |
| a1_no_gate | 42 | 256 | 0.8250 | 108.9 | 0 | 4 | +0.0450 | -1014 |

Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. Original `outputs/pace/gate2/` not overwritten.
