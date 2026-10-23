# PACE Method Spec (v1.0 — implementation contract)

Aligned with `PACE_Research_Implementation_Plan.md`. Heuristic confidence gate unless anytime-valid control is added later.

## Objectives

Maximize accuracy \(a(p)\), minimize deployment cost \(c(p)\).

## Paired evidence

On common observed sample IDs \(S_{pq}\):

\[
d_i = z[p,i]-z[q,i]\in\{-1,0,1\},\quad
\hat\Delta_{pq}=|S_{pq}|^{-1}\sum_i d_i
\]

Bootstrap (B=200, seeded) the mean of \(d_i\) to get percentile CI \([L_{pq},U_{pq}]\) (nominal, not simultaneous).

## Evidence-gated elimination

Candidate \(q\) is **confidence-eliminated** if ∃ \(p\) such that:

1. Cost gate (v1): \(\hat c(p) \le \hat c(q)\) on the **shared uniform/anchor estimator** (point cost; cost uncertainty deferred).
2. Accuracy evidence: \(L_{pq} > \varepsilon\) with precommitted \(\varepsilon=0\) (strict positive lower bound on paired accuracy advantage of \(p\) over \(q\)).

If \(|S_{pq}| < n_{\min}\) (default 8): pair is **undecided** (never eliminate via that pair).

## Bounded survival

- Active capacity \(\mu\) (default 8).
- Soft archive capacity \(K\) (default 8) for temporarily undecided / non-eliminated extras.
- If after confidence eliminations the remaining set exceeds \(\mu+K\), apply **forced eviction** via NSGA-II rank + crowding on anchor estimates, then prompt-id tie-break. Log forced evictions separately from confidence eliminations.

## Matched evaluation (Gate 1)

All compared methods share:

- Same candidate pool, seeds, init common anchors, Paired Racing query schedule (or shared purchased mask), call budget.
- **Treatment:** survivor selection only — point-estimate NSGA truncate (B0) vs evidence-gated + bounded archive (PACE).

## Audit-only full matrix

Full 160-column search matrix may be used **only** by the offline auditor to define reference Pareto / false-elimination labels. Policies never see unaquired cells.
