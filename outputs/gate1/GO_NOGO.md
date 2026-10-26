# Gate 1 Go/No-Go (registered primary: budget 640)

**Date**: 2026-10-23  
**Matrix**: Qwen2.5-7B-Instruct × AG News, 16×(160+200)=5760 evals (once).  
**Replay**: 20 seeds; metric = macro test preference selection regret.

## Primary comparison (budget 640)

| policy     | mean regret | SD     |
|------------|-------------|--------|
| racing     | **0.0058**  | 0.0054 |
| uniform    | 0.0059      | 0.0031 |
| risk_only  | 0.0138      | 0.0194 |
| **dice**   | **0.0141**  | 0.0155 |

Paired seed mean(dice − racing) ≈ **+0.008** (DICE worse).

## Trend (320 / 640 / 1280)

Racing ≤ DICE at all three registered budgets; Uniform ≈ Racing at 640/1280.  
Impact ablation (risk_only vs dice): no clear gain for Full DICE.

## Decision (plan §3 Gate 1)

**NO-GO for Gate 2 expansion.**  
H1 not supported: Full DICE does not beat Paired Racing at 640, and 320/1280 show no consistent advantage.  
Do not scan hyper-parameters to force a win. Optional: one pre-registered diagnostic (e.g. impact definition / contender set) marked exploratory — not done yet.

## Notes

- Prompt `p06` (JSON-format) has search accuracy 0.0 under the frozen parser — left frozen per protocol.
- First replay bug (pair tags excluded from pop estimates) fixed before this table; results above use the corrected estimator.
