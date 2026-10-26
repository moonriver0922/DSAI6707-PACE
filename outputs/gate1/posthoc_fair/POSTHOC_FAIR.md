# Post-hoc Fair Ablation (0 new LLM calls)

**Status**: diagnostic / exploratory — does **not** replace pre-registered Gate-1 NO-GO.

## Design

- Shared: `racing_stop_r`, Uniform column fallback when no pair query, same matrix/seeds/budgets.
- Sole difference: `use_impact` True (`fair_dice`) vs False (`fair_racing`).
- `fair_risk_only` should match `fair_racing` (sanity).
- Original `dice` / `racing` included for side-by-side.

## Primary budget 640 — macro test preference regret

| policy         |   budget |   regret_mean |   regret_sd |   ci_lo |   ci_hi |   util_mean |   util_sd |   n |
|:---------------|---------:|--------------:|------------:|--------:|--------:|------------:|----------:|----:|
| fair_risk_only |      640 |       0.00584 |     0.0054  | 0.0035  | 0.00828 |     0.99531 |   0.00669 |  20 |
| fair_racing    |      640 |       0.00584 |     0.0054  | 0.00366 | 0.00829 |     0.99531 |   0.00669 |  20 |
| racing         |      640 |       0.00584 |     0.0054  | 0.0037  | 0.00825 |     0.99531 |   0.00669 |  20 |
| fair_dice      |      640 |       0.0121  |     0.01059 | 0.00818 | 0.01681 |     0.99422 |   0.00781 |  20 |
| dice           |      640 |       0.01407 |     0.01549 | 0.00831 | 0.02121 |     0.84328 |   0.12394 |  20 |

## Budget utilization (primary)

| policy         |   mean |    std |
|:---------------|-------:|-------:|
| dice           | 0.8433 | 0.1239 |
| fair_dice      | 0.9942 | 0.0078 |
| fair_racing    | 0.9953 | 0.0067 |
| fair_risk_only | 0.9953 | 0.0067 |
| racing         | 0.9953 | 0.0067 |

## Paired seed: fair_dice − fair_racing (regret; lower better for method)

- mean Δ = **0.00626** (positive ⇒ fair_dice worse)
- P(fair_dice better) = 0.20, P(tie)=0.10, P(racing better)=0.70

## Trend (fair policies only)

|                       |    mean |     std |
|:----------------------|--------:|--------:|
| (320, 'fair_dice')    | 0.01354 | 0.01399 |
| (320, 'fair_racing')  | 0.01009 | 0.01247 |
| (640, 'fair_dice')    | 0.0121  | 0.01059 |
| (640, 'fair_racing')  | 0.00584 | 0.0054  |
| (1280, 'fair_dice')   | 0.00677 | 0.00328 |
| (1280, 'fair_racing') | 0.00627 | 0.00444 |

## Verdict

- util gap (fair_dice − fair_racing) = -0.0011
- regret gap (fair_dice − fair_racing) = 0.00626
- Under equal early-stop/fallback, fair_dice still has **higher** mean regret than fair_racing. Supports that Decision-Critical Scoring itself lacks benefit in this setting (not only the missing stop/fallback).
