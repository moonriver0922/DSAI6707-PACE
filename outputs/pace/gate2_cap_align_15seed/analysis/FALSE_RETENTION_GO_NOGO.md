# PACE False-Retention / Archive Opportunity-Cost Audit (0 GPU)

Source: `/home3/river/DSAI6707/DICE_PO/outputs/pace/gate2_cap_align_15seed` · n=15 seeds · PACE vs B0+Archive logs only.

## Limitations (read first)

- parent_ids not logged; breeding inferred from active set at offspring steps only.
- No mid-run per-candidate accuracy in history; soft counterfactual is final-pool only.
- PACE vs B0 trajectories diverge; cannot attribute the same child to alternate parents.
- Zero new LLM calls; not a causal proof of SPaCE online gains.

## Headline means (PACE)

| Signal | PACE | B0+Archive |
|--------|------|------------|
| Forced share of eliminations | 0.765 | 1.000 |
| Mean conf_elim / run | 3.80 | 0.00 |
| Mean forced / run | 12.27 | 16.00 |
| Pure passenger share of archive-slot-steps | 0.465 | 0.436 |
| Archive-only → later breeding rate | 0.520 | 0.555 |
| Mean end archive search_acc | 0.787 | 0.755 |
| Mean end active search_acc | 0.769 | 0.762 |
| Final low-acc (<0.5) sitting in archive | 0.27 | 0.67 |
| Frac winners ever archive-only | 0.200 | — |
| Frac winners final archive-only (never needed active to win) | 0.000 | — |
| Corr(passenger_share, Δacc) | +0.267 | — |

## Interpretation checklist

1. **Protected → better offspring?** Exact parent_ids absent; proxy = archive-only candidates that later appear in `active` before an offspring step (`archive_then_bred_rate`).
2. **False retention occupying archive?** `passenger_archive_share` + final low-acc in archive.
3. **Soft vs hard?** Final-pool soft mass outside top-μ is observational only (mean soft mass outside top-μ ≈ 0.251).

## Go / No-Go (SPaCE extension)

**Decision: `CONDITIONAL`**

Mixed signals: document false-retention pressure and forced-eviction dominance, but do not claim soft-selection will fix online Δacc without a new frozen protocol. Default course delivery = freeze PACE v1; SPaCE only as labeled extension if pursued.

Signals:

- `passenger_heavy`: **True**
- `archive_weak_vs_active`: **False**
- `rarely_promoted_to_breeding`: **False**
- `forced_dominates_elims`: **True**
- `winner_often_touched_archive`: **False**
- `protection_valuable`: **False**
- `bottleneck`: **False**

PACE v1 (15-seed cap-align) remains the course primary deliverable regardless. SPaCE is a conditional research extension, not a replacement of frozen results.

## Per-seed paired snapshot

| seed | Δacc | PACE pass.share | arch→bred | PACE low@arch | winner touched archive |
|------|------|-----------------|-----------|---------------|------------------------|
| 42 | +0.000 | 0.21 | 0.79 | 1 | False |
| 123 | +0.025 | 0.35 | 0.50 | 0 | False |
| 456 | +0.120 | 0.66 | 0.35 | 0 | False |
| 6707 | +0.000 | 0.54 | 0.53 | 0 | False |
| 2026 | +0.000 | 0.56 | 0.41 | 0 | False |
| 31415 | +0.050 | 0.33 | 0.67 | 0 | False |
| 27182 | +0.035 | 0.62 | 0.41 | 0 | False |
| 16180 | +0.010 | 0.62 | 0.41 | 1 | False |
| 90210 | +0.040 | 0.49 | 0.47 | 0 | True |
| 13579 | -0.055 | 0.38 | 0.58 | 1 | False |
| 86753 | +0.020 | 0.50 | 0.47 | 0 | False |
| 24680 | -0.005 | 0.44 | 0.52 | 0 | True |
| 97531 | +0.000 | 0.45 | 0.56 | 1 | False |
| 11235 | +0.060 | 0.35 | 0.60 | 0 | False |
| 42424 | -0.045 | 0.47 | 0.53 | 0 | True |

## Insight (for report discussion, not a new claim of SOTA)

> Under a fixed evolutionary search budget, minimizing false elimination alone does not maximize the long-term value of the surviving population.

DICE asked whether each *evaluation* was worth buying; PACE asked whether each *elimination* was justified; the open question is whether each *survivor* deserves continued breeding / archive mass.
