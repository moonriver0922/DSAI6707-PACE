# PACE Gate 2 — Cap-Align 15-Seed Primary Online Result

**Role:** subsequent extension / paper **primary** online experiment.
**Exploratory retained:** seeds `42, 123, 456` (3-seed exploratory; not discarded).
**Contrast:** PACE (μ=8, K=8, confidence gate on) vs B0+Archive (μ=8, K=8, gate off).
**Protocol:** frozen (AG News × Qwen2.5-7B, pop=8, gen=4, 1280 calls, Racing eval).

## Primary: paired test accuracy difference (PACE − B0+Archive)

- n = 15 fixed seeds
- mean Δacc = **+0.0170**
- 95% CI (Student-t) = **[-0.0065, +0.0405]**
- Win / Tie / Loss = **8 / 4 / 3**
- Paired sign-flip permutation p (two-sided, 100,000 perms) = **0.1496**
- Exploratory 3-seed mean Δacc (retained) = +0.0483

## Method means (all 15)

| Method | test_acc | test_cost (tok) | calls | budget util | search sel. regret | conf_elim | forced |
|--------|----------|-----------------|-------|-------------|--------------------|-----------|--------|
| b0_archive | 0.7840 | 112.6 | 1275 | 0.996 | 0.0000 | 0.0 | 16.0 |
| pace | 0.8010 | 115.5 | 1276 | 0.997 | 0.0000 | 3.8 | 12.3 |

## Per-seed paired table

| seed | cohort | B0 acc | PACE acc | Δacc | W/T/L | B0 cost | PACE cost | B0 calls | PACE calls |
|------|--------|--------|----------|------|-------|---------|-----------|----------|------------|
| 42 | exploratory | 0.8550 | 0.8550 | +0.0000 | Tie | 116.1 | 116.1 | 1280 | 1280 |
| 123 | exploratory | 0.7750 | 0.8000 | +0.0250 | Win | 107.5 | 119.7 | 1270 | 1280 |
| 456 | exploratory | 0.6550 | 0.7750 | +0.1200 | Win | 115.3 | 110.8 | 1280 | 1280 |
| 6707 | extension | 0.8200 | 0.8200 | +0.0000 | Tie | 117.1 | 117.1 | 1274 | 1274 |
| 2026 | extension | 0.8050 | 0.8050 | +0.0000 | Tie | 109.6 | 109.6 | 1272 | 1256 |
| 31415 | extension | 0.7600 | 0.8100 | +0.0500 | Win | 105.6 | 124.7 | 1268 | 1280 |
| 27182 | extension | 0.8000 | 0.8350 | +0.0350 | Win | 119.7 | 123.7 | 1266 | 1280 |
| 16180 | extension | 0.7750 | 0.7850 | +0.0100 | Win | 103.8 | 116.9 | 1280 | 1270 |
| 90210 | extension | 0.7800 | 0.8200 | +0.0400 | Win | 109.7 | 111.6 | 1280 | 1276 |
| 13579 | extension | 0.8200 | 0.7650 | -0.0550 | Loss | 122.6 | 103.4 | 1258 | 1280 |
| 86753 | extension | 0.7550 | 0.7750 | +0.0200 | Win | 112.6 | 103.8 | 1280 | 1278 |
| 24680 | extension | 0.7950 | 0.7900 | -0.0050 | Loss | 109.6 | 126.9 | 1280 | 1280 |
| 97531 | extension | 0.7900 | 0.7900 | +0.0000 | Tie | 109.6 | 113.6 | 1280 | 1280 |
| 11235 | extension | 0.7750 | 0.8350 | +0.0600 | Win | 113.9 | 121.6 | 1280 | 1276 |
| 42424 | extension | 0.8000 | 0.7550 | -0.0450 | Loss | 116.6 | 112.6 | 1280 | 1268 |

## Notes

- Acceptance is a credible mean/CI/cost record, **not** a required significant win.
- `search_sel_regret` is auxiliary on **search** estimates within the final retained set (sealed-test selection regret over all generated candidates was not purchased).
- Original 3-seed tree `outputs/pace/gate2_cap_align/` is untouched.
