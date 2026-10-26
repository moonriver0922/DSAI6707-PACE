# PACE — 6-Minute Presentation (6 slides)

Keep one message per slide. Numbers only from frozen results.

---

## Slide 1 — Why this problem?

**Title:** Noisy scores kill good prompts too early  

- Evolutionary prompt search under a **fixed LLM-call budget**  
- Bi-objective: **accuracy ↑** and **token cost ↓**  
- With few samples, point estimates fluctuate → NSGA truncate can drop a prompt that is still competitive  

**One line:** Premature elimination is a real failure mode in low-budget EC+LLM search.

---

## Slide 2 — What’s missing in standard EC?

**Title:** Matched evaluations, unmatched survival  

- Racing / shared samples already help **which data to buy**  
- Survival often still uses **noisy point ranks**  
- Gap: decide **retain vs eliminate** only when evidence is enough  

**One line:** We change the survivor rule, not the whole optimizer.

---

## Slide 3 — What is PACE?

**Title:** Evidence-gated survivor selection  

1. Paired bootstrap on shared samples → CI \([L,U]\)  
2. Eliminate \(q\) only if a cheaper-or-equal \(p\) has \(L_{pq}>0\) and enough overlap  
3. Otherwise keep in a **bounded archive** (\(\mu+K\)); forced eviction logged if over capacity  

**Visual:** Racing (shared) → Survivor (PACE vs B0) → Breed from active  

**One line:** Protect the undecided; don’t invent scores.

---

## Slide 4 — Experimental setup

**Title:** Fair, frozen protocol  

- Qwen2.5-7B · AG News · A100  
- Offline Gate 1: same purchase trajectory; survivor only differs  
- Online Gate 2: **PACE vs B0+Archive**, \(\mu=K=8\), 4 gens, **1280** calls  
- **15 fixed seeds** (primary); ablations for mechanism  

**One line:** Capacity-matched baseline; no hyperparameter fishing after seeing test.

---

## Slide 5 — Results

**Title:** Clear offline effect; cautious online trend  

| | Result |
|--|--------|
| Gate 1 false-elim | **18.3% → 2.5%** |
| Gate 2 mean test acc | **78.4% → 80.1%** (Δ **+1.7** pp) |
| 15-seed paired | W/T/L **8/4/3**; 95% CI **includes 0** |

**One line:** Mechanism works offline; online lift is positive but not locked.

---

## Slide 6 — Takeaway & limits

**Title:** Course-ready conclusion  

- **Done:** EC method + LLM task + baselines + ablations + multi-seed evidence  
- **Claim:** Confidence-aware survival reduces premature elimination; online accuracy trends up  
- **Limits:** Variance, forced capacity pressure, slight token cost ↑  
- **Not claiming:** SOTA / significant superiority / new soft-selection algorithm  

**Closing:** Simple story, complete delivery, honest numbers.
