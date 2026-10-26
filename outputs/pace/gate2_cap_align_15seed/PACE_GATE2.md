# PACE Gate 2 Cap-Align 15-Seed Primary

Experiment: `gate2_cap_align_15seed` · Smoke: `False` · eval=`racing`

- μ=8, K=8, ε=0.0, n_min=8, B=200
- pop=8, gen=4, calls_cap=1280, seeds=[42, 123, 456, 6707, 2026, 31415, 27182, 16180, 90210, 13579, 86753, 24680, 97531, 11235, 42424]
- gate1_warm_cells=1280
- disable_confidence_gate methods: ['b0_archive']

## Method means

- **b0_archive**: test_acc=0.7840, test_cost=112.6, calls=1275, conf_elim=0.0, forced=16.0 (n=15)
- **pace**: test_acc=0.8010, test_cost=115.5, calls=1276, conf_elim=3.8, forced=12.3 (n=15)

## Within-run Δ vs b0_archive (this experiment)

- **pace − b0_archive**: Δacc=+0.0170, Δcost=+2.9, Δcalls=+1

## Δ vs original `outputs/pace/gate2`

- **b0_archive**: Δacc=+0.0223 (this 0.7840 vs gate2 0.7617), Δcost=-0.3, Δcalls=-1
- **pace**: Δacc=-0.0090 (this 0.8010 vs gate2 0.8100), Δcost=-0.1, Δcalls=-4

## Runs (per seed)

| method | seed | calls | test_acc | test_cost | conf_elim | forced | Δacc_vs_gate2 | Δcalls_vs_gate2 |
|--------|------|-------|----------|-----------|-----------|--------|--------------|-----------------|
| b0_archive | 42 | 1280 | 0.8550 | 116.1 | 0 | 16 | +0.0000 | +0 |
| pace | 42 | 1280 | 0.8550 | 116.1 | 1 | 15 | +0.0000 | +0 |
| b0_archive | 123 | 1270 | 0.7750 | 107.5 | 0 | 16 | +0.0000 | +0 |
| pace | 123 | 1280 | 0.8000 | 119.7 | 3 | 13 | +0.0000 | +0 |
| b0_archive | 456 | 1280 | 0.6550 | 115.3 | 0 | 16 | +0.0000 | +0 |
| pace | 456 | 1280 | 0.7750 | 110.8 | 5 | 11 | +0.0000 | +0 |
| b0_archive | 6707 | 1274 | 0.8200 | 117.1 | 0 | 16 | — | — |
| pace | 6707 | 1274 | 0.8200 | 117.1 | 3 | 13 | — | — |
| b0_archive | 2026 | 1272 | 0.8050 | 109.6 | 0 | 16 | — | — |
| pace | 2026 | 1256 | 0.8050 | 109.6 | 7 | 9 | — | — |
| b0_archive | 31415 | 1268 | 0.7600 | 105.6 | 0 | 16 | — | — |
| pace | 31415 | 1280 | 0.8100 | 124.7 | 4 | 12 | — | — |
| b0_archive | 27182 | 1266 | 0.8000 | 119.7 | 0 | 16 | — | — |
| pace | 27182 | 1280 | 0.8350 | 123.7 | 9 | 8 | — | — |
| b0_archive | 16180 | 1280 | 0.7750 | 103.8 | 0 | 16 | — | — |
| pace | 16180 | 1270 | 0.7850 | 116.9 | 2 | 14 | — | — |
| b0_archive | 90210 | 1280 | 0.7800 | 109.7 | 0 | 16 | — | — |
| pace | 90210 | 1276 | 0.8200 | 111.6 | 5 | 11 | — | — |
| b0_archive | 13579 | 1258 | 0.8200 | 122.6 | 0 | 16 | — | — |
| pace | 13579 | 1280 | 0.7650 | 103.4 | 4 | 12 | — | — |
| b0_archive | 86753 | 1280 | 0.7550 | 112.6 | 0 | 16 | — | — |
| pace | 86753 | 1278 | 0.7750 | 103.8 | 4 | 12 | — | — |
| b0_archive | 24680 | 1280 | 0.7950 | 109.6 | 0 | 16 | — | — |
| pace | 24680 | 1280 | 0.7900 | 126.9 | 4 | 12 | — | — |
| b0_archive | 97531 | 1280 | 0.7900 | 109.6 | 0 | 16 | — | — |
| pace | 97531 | 1280 | 0.7900 | 113.6 | 3 | 13 | — | — |
| b0_archive | 11235 | 1280 | 0.7750 | 113.9 | 0 | 16 | — | — |
| pace | 11235 | 1276 | 0.8350 | 121.6 | 2 | 14 | — | — |
| b0_archive | 42424 | 1280 | 0.8000 | 116.6 | 0 | 16 | — | — |
| pace | 42424 | 1268 | 0.7550 | 112.6 | 1 | 15 | — | — |

Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).

DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. Original `outputs/pace/gate2/` not overwritten.
