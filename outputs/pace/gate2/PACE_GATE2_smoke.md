# PACE Gate 2 Results

Smoke: `True` · eval=`racing` · treatment=`survivor only`

- μ=4, K=4, ε=0.0, n_min=8, B=40
- pop=4, gen=2, calls_cap=256, seeds=[42]

## Method means

- **b0_point_nsga**: test_acc=0.8250, test_cost=108.9, calls=256 (n=1)
- **pace**: test_acc=0.9000, test_cost=121.3, calls=256 (n=1)

## Runs

| method | seed | calls | test_acc | test_cost | winner | conf_elim | forced |
|--------|------|-------|----------|-----------|--------|-----------|--------|
| b0_point_nsga | 42 | 256 | 0.8250 | 108.9 | `b0_point_nsga_s42_g1_c02` | 0 | 0 |
| pace | 42 | 256 | 0.9000 | 121.3 | `pace_s42_g1_c01` | 0 | 0 |

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged.
