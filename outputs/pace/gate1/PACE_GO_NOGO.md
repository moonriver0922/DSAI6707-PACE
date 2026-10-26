# PACE Gate 1 — GO/NO-GO

**Date:** 2026-10-23  
**Matrix:** frozen AG News × Qwen2.5-7B (`outputs/gate1/oracle_matrices.npz`)  
**Design:** Shared Paired-Racing purchase trajectory per (budget, seed); only survivor rule differs.  
**0 new LLM calls.**

## Reference Pareto (audit-only)

`p00, p01, p02, p03, p05, p15`

## Primary budget 640 (20 replay seeds)

| Method | False-elim ↓ | Ref-Pareto recall ↑ | Retained-set regret ↓ | Mean #retained |
|--------|--------------|---------------------|------------------------|----------------|
| B0 point NSGA (μ=8) | 0.183 | 0.817 | 0.0007 | 8.0 |
| PACE (μ=8, K=8) | **0.025** | **0.975** | **0.000** | 11.7 |

- H0 (nontrivial erroneous elimination under B0): **supported** (18.3% > 5%).  
- H1 (PACE reduces false elim): **supported** (Δ = −0.158).  
- H2 (retained utility not hurt): **supported** on retained-set regret (Δ = −0.0007).

**Capacity note:** PACE may keep up to μ+K; B0 keeps exactly μ. This is by design (bounded archive). Forced evictions were 0 at 640 in this matrix. Documented as a possible confound for “mere retention”; still, false-elim/recall improve and regret does not worsen.

## Decision

# **GO** for conditional Gate 2 (online PACE + Racing vs matched NSGA-II + Racing)

Precommitted rule: H0_rate≥0.05 ∧ Δ false-elim &lt; 0 ∧ Δ regret ≤ 0.002 → **GO**.

## Non-claims

- Not a claim that PACE is SOTA online yet (Gate 2 not run).  
- DICE-PO historical negatives remain negatives; not relabeled as PACE wins.  
- Replay seeds ≠ independent datasets; no significance theatre.
