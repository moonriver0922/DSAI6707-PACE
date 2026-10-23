# PACE — Research and Implementation Plan (v1.0)

**Course:** DSAI6707 Advanced Computational Intelligence  
**Working title:** *PACE: Paired-Assessment with Confidence-Preserving Evolution*  
**Status:** Proposed method; not experimentally validated  
**Hardware:** NVIDIA A100 80 GB (available)  
**Priority:** First test the premise with existing data, then decide whether online evolution is justified.

## 1. Research objective and positioning

**Research question.** Under limited LLM evaluation budgets, can evidence-gated survivor selection reduce erroneous elimination of promising prompts and improve final multi-objective search quality relative to point-estimate evolutionary selection?

**Key mechanism:** Evidence-Gated Survivor Selection. Evaluation uncertainty governs **whether the evidence is strong enough to eliminate a candidate**, rather than which pair receives the highest DICE-style decision-impact score.

PACE is not claimed to be the first confidence-based evolutionary method. Review noisy evolutionary optimization, racing, confidence-aware selection, and prompt evolution before making novelty claims. Behavioral diversity, reflection, complex surrogates, and adaptive operator scheduling are **out of scope** in v1.

## 2. What prior work taught us

Existing DICE-PO experiments are retained as historical evidence, not presented as PACE results.

- DICE Gate 1, at 640 evaluation calls: Paired Racing regret 0.00584 versus original DICE 0.01407.
- Post-hoc matched-stop/fallback comparison: fair Racing 0.00584 versus fair DICE 0.01210; no demonstrated advantage from decision-impact weighting.
- Previous NSGA-II online test ran successfully, but Random Search had more generated candidates (40 vs 32); observed mean test accuracy is **not** a fair causal comparison of survivor selection.
- Some winners showed search accuracy near 1.0 with materially lower test accuracy: consistent with, but not proof of, noisy selection/overfitting.

These motivate a falsifiable test of *erroneous elimination* before investing in a replacement algorithm.

## 3. Method definition

Let prompt candidate p have unknown generalization accuracy a(p) and deployment cost c(p). Optimization objectives are maximize a(p), minimize c(p). For candidate pair p,q on the *same observed sample IDs* S_pq, define d_i = 1[M(p,x_i)=y_i] - 1[M(q,x_i)=y_i] in {-1,0,1}. Compute paired difference mean delta_hat_pq and a confidence interval [L_pq,U_pq] using a specified bootstrap or paired concentration bound.

**Evidence-gated elimination (initial version):** candidate q is confidently dominated if there exists p with c(p) <= c(q) (under a predetermined cost uncertainty policy) and L_pq > epsilon. Here epsilon >= 0 is a precommitted practical accuracy margin. This is a **sufficient rule**, not proof that every non-eliminated candidate is good. Missing/common-sample insufficiency must mean **undecided**, not **safe forever**.

**Bounded survival:** maintain at most mu active candidates and K temporary/archive candidates. When confidence evidence is insufficient to fit capacity, apply a fully specified deterministic tie-break (e.g., standard NSGA-II rank/crowding on common anchor estimates plus seeded tie ordering). Record every forced capacity eviction separately from confidence-based elimination. A confidence gate cannot magically preserve unlimited candidates.

**Shared paired evaluation:** each candidate starts from common randomized anchor samples; further pairs receive the same fresh sample IDs, with a fixed evaluation-call budget. Use the existing Paired Racing machinery for additional evaluation and stopping. Do not use hidden/oracle labels for query selection. Cache hits count as zero new inference calls, but all inference used in candidate generation is separately logged.

**Important statistical safeguards:** clarify nominal vs simultaneous confidence coverage, repeated sequential looks, bootstrap calibration, and the distinction between confidence evidence and guaranteed correctness. Treat the first implementation as heuristic unless anytime-valid error control is implemented.

## 4. Hypotheses and falsification criteria

- **H0 (premise):** Point-estimate survivor selection makes a nontrivial number of erroneous eliminations at realistic low budgets.
- **H1 (mechanism):** PACE reduces erroneous elimination of high-quality/Pareto-promising candidates at identical candidate pools, observed information, and call budgets.
- **H2 (utility):** Gains in retention translate into equal or better final regret/hypervolume under a fixed archive capacity; merely retaining more is not success.
- **H3 (online, conditional):** Integrating PACE into genuinely evolutionary search improves final selection under equal generation opportunity and inference budgets.

**Stop if:** H0 fails materially; PACE does not outperform a genuinely matched non-gated baseline in H1/H2; or any advantage depends on seeing oracle/test data or unlimited archive size. Negative findings are retained and reported.

## 5. Lean experiment plan

### Gate 0 — Correctness (CPU only)

Implement unit tests for paired sample ID equality; no oracle leakage; cache/budget accounting; deterministic seeded replay; domination direction and token-cost consistency; forced archive evictions; capacity cap; tied scores; no common observations; remaining budget < pair batch; confidence gate with synthetic known outcomes.

**Pass criterion:** all tests pass and a reproducible audit artifact is saved.

### Gate 1a — Does erroneous elimination actually happen? (0 LLM calls)

Use existing AG News × Qwen2.5-7B, **16 prompts × 160 search samples** from the frozen Gate 1 oracle matrix. The algorithm receives only purchased observations. The full 160-sample matrix is used **only by the audit evaluator** to label reference-quality prompts, *not* for selecting queries or tuning.

Replay shared seeds and a prespecified low budget; compare point-estimate NSGA-II survival with the full-matrix reference. Measure:
- **False-elimination rate:** reference-Pareto/promising candidate eliminated under partial observations (state reference definition in advance).
- **Reference-Pareto recall among retained candidates** and **forced-eviction rate**.
- Final preference regret or full-matrix hypervolume of retained set.

Critical caution: all methods must use the **same candidate pool and archive capacity**. Since this is a fixed-pool replay, it establishes only selection behavior, not online evolutionary improvement.

### Gate 1b — Does the gate help? (0 LLM calls)

**Primary comparison:** matched point-estimate selection + Paired Racing vs PACE + *the same* Paired Racing. Same anchor observations, common-sample schedule, budget, archive capacity, random seed, and tie-breaking rules. The **only intended treatment** is evidence-gated elimination. Log effective budget, number of forced evictions, confidence eliminations, and final retained candidates.

Use a small precommitted budget set (e.g. 320 and 640 calls; 640 primary), rather than a grid. Use 20 replay seeds on the same frozen matrix but **do not treat them as 20 independent datasets**. Do not adjust thresholds after seeing test-reference outcomes. If capacity handling confounds the comparison, document it explicitly.

**Main output:** one table (error elimination/recall/regret/calls), one trajectory plot, one qualitative elimination trace. Do not claim statistical significance from repeated replay seeds alone.

### Gate 2 — Conditional online evolutionary proof (A100)

Only run if Gate 1 provides meaningful evidence. Use AG News + Qwen2.5-7B-Instruct, with true prompt offspring via the same frozen mutation and crossover operators in all comparable methods.

- Population mu=8; 4 generations; seeds 42,123,456.
- Maximum **1,280 new prompt–sample evaluations per run**.
- At minimum compare **NSGA-II + Racing** against **PACE + Racing**. Add Random Search only if the same generated-candidate count, operators, and selection opportunity can be matched.
- Match 8 initial + 24 generated candidates in every primary comparison; match total evaluation budgets and deployment cost definitions.
- Winner-only evaluation on a sealed 200-item test split, with final selection based exclusively on search/development observations.

For 2 methods × 3 seeds × 1,280, search cap is **7,680 evaluations**, plus at most 1,200 winner-test evaluations, total **8,880**; generation/model mutation calls and token costs are logged separately. If an additional matched Random baseline is included, update the budget prospectively. Avoid secondary datasets unless the original hypothesis is supported and time remains.

## 6. Baselines and ablations

| Label | Evaluation | Survivor selection | Purpose |
|---|---|---|---|
| B0 | Paired Racing | Standard NSGA-II point estimates | Primary controlled baseline |
| PACE | Identical Paired Racing | Evidence-gated + bounded archive | Proposed treatment |
| A1 (optional) | Same | Confidence gate disabled | Code-path ablation, should reproduce B0 |
| Random (conditional) | Equal evaluation budget | Matched random candidate proposal/selection | Broader search sanity check |

Avoid comparing a baseline with different fallback, stopping, candidate generation, or budget utilization and attributing outcomes to gating.

## 7. Environment and reproducibility

- GPU: 1× NVIDIA A100 80GB; model Qwen/Qwen2.5-7B-Instruct; BF16 inference; deterministic greedy decoding (temperature 0), max_new_tokens=8 for AG News labels.
- Python 3.11; compatible CUDA PyTorch; `transformers`, `accelerate`, `datasets`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pytest`, `pyyaml`, and optionally `pymoo`, `matplotlib`.
- Freeze versions **after smoke testing** (do not fabricate versions). Record model revision/hash, CUDA/driver details, seeds, sample IDs, prompt hashes, tokenizer config, parse failures, inference/cache logs and exact per-run token totals.
- Inference policy must handle invalid outputs explicitly without changing the frozen evaluation protocol post hoc.
- Work from current `DICE_PO` repository; preserve earlier DICE results untouched, place PACE work in a separate configuration/output namespace.

## 8. Outputs and acceptance criteria

**Required files:** `PACE_METHOD_SPEC.md`, frozen `pace_gate1.yaml`, code changes, unit tests, replay per-seed CSV, policy trace JSONL, precommitted stop decision, audit tables, plot, online results only if GO, README and full experiment manifests.

**Acceptance:** no oracle access; exact candidate and budget matching; logged reason for each elimination; all policies produce comparable final candidate sets; all claims trace to a table/trace; zero fabricated positive results; Gate 1 negative results remain visible if the method is not pursued.

**Course deliverables (per project brief):** PDF research report covering problem, 2024–2026 literature review **at least 20 relevant papers**, method, experiments, discussion, references; full source code + README; experiment logs and results; supplementary materials as appropriate; six-minute presentation. The assessment documents disagree on submission deadline (**2 Nov** vs **15 Nov 2026**); confirm with teaching staff and plan against earlier date meanwhile.

## 9. Implementation order for AI Coding

1. Freeze current DICE-PO v1.3 outputs and create `pace` branch.
2. Read `oracle_matrices.npz` format and audit boundaries between policy-view and evaluator-view.
3. Write Gate 0 tests before new policy implementation.
4. Build point-estimate baseline and evidence-gated bounded survivor selector; ensure only one mechanism differs.
5. Run Gate 1a premise check, then Gate 1b matched intervention, **no LLM calls**.
6. Produce GO/NO-GO decision; no threshold scan or post-hoc cherry-picking.
7. Only on GO: integrate into online NSGA-II search and execute Gate 2 within cap.
8. Prepare report sections without making unsupported superiority claims.

## 10. Research interpretation boundaries

- Reduction in premature elimination is a **mechanistic outcome**, not by itself evidence of better final prompt utility.
- Fixed-pool oracle is finite-sample reference, not true expected generalization; sealed test remains inaccessible to policy.
- Common observations and matched evaluation decisions matter; differences in survivor capacity, sample choices or stopping rules are confounders.
- Confidence-preserving selection has a strong existing intellectual lineage in noisy evolutionary algorithms: novelty requires precise literature comparison and demonstrated technical distinction.
- DICE negative results should be discussed honestly as motivation, never relabeled as PACE wins.
