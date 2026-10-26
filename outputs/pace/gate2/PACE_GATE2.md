# PACE Gate 2 Results

Smoke: `False` · eval=`racing` · treatment=`survivor only`

- μ=8, K=8, ε=0.0, n_min=8, B=200
- pop=8, gen=4, calls_cap=1280, seeds=[42, 123, 456]

## Method means

- **b0_point_nsga**: test_acc=0.7833, test_cost=114.7, calls=1269 (n=3)
- **pace**: test_acc=0.8100, test_cost=115.5, calls=1280 (n=3)

## Runs

| method | seed | calls | test_acc | test_cost | winner | conf_elim | forced |
|--------|------|-------|----------|-----------|--------|-----------|--------|
| b0_point_nsga | 42 | 1270 | 0.7800 | 112.6 | `b0_point_nsga_s42_g2_c06` | 0 | 0 |
| pace | 42 | 1280 | 0.8550 | 116.1 | `pace_s42_g3_c07` | 1 | 15 |
| b0_point_nsga | 123 | 1270 | 0.8000 | 114.9 | `b0_point_nsga_s123_g3_c00` | 0 | 0 |
| pace | 123 | 1280 | 0.8000 | 119.7 | `pace_s123_g3_c05` | 3 | 13 |
| b0_point_nsga | 456 | 1268 | 0.7700 | 116.5 | `b0_point_nsga_s456_g3_c07` | 0 | 0 |
| pace | 456 | 1280 | 0.7750 | 110.8 | `pace_s456_g3_c00` | 5 | 11 |

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged.
