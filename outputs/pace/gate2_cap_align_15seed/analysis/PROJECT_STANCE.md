# Stance after false-retention audit (2026-10-26)

## Course delivery (frozen)

**PACE v1** remains the DSAI6707 primary artifact:
- Gate 1 false-elim reduction
- Gate 2 capacity-aligned 15-seed primary (`Δacc=+0.017`, CI includes 0)
- Ablations (μ-only, A1, cold, cap-align)

Do **not** retune or re-run the 15-seed protocol to chase significance.

## SPaCE

**Not started.** Zero-GPU audit decision = `CONDITIONAL`  
(see `FALSE_RETENTION_GO_NOGO.md`).

Evidence does **not** cleanly show “gate keeps junk that never breeds.”
Archive-end search accuracy is **not** worse than active; ~52% of archive-only
candidates later enter the breeding `active` set. Dominant logged pressure is
**forced capacity eviction** (≈76% of PACE eliminations), not confidence kills.

If a research extension is pursued later: soft selection / breeding weights under
fixed μ+K — as a **labeled extension**, not a rewrite of PACE negatives or wins.

## Logging gap for any future online run

`parent_ids` exist in `evolution.Individual` but were **not** persisted in Gate2 JSON.
Exact parent→child value attribution requires a logging patch before new GPU work.
