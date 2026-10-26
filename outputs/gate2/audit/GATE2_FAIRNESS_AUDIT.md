# Gate 2 Fairness Audit (0 GPU)

## 1. Test-set protocol

- `evaluate_on_test(predictor, [winner], ...)` call sites: **2**
- Full-pop test call sites remaining: **0**
- Conclusion: Test set is only used in evaluate_on_test([winner]); selection uses search estimates. No full-pop test in current code.

## 2. Designed candidate budgets (from code)

- NSGA-II new candidates: **24** (total IDs 32)
- Random Search new candidates: **32** (total IDs 40)

### Unfairness flags

- Random Search generates 32 new candidates vs NSGA-II 24 (8 more).
- Random Search screens the entire pool of init+new under one Uniform allocation; NSGA truncates to μ each generation so many children get little budget.
- Operators differ: Random=mutate only; NSGA=crossover+mutate.
- Random finalist set is top-μ by utility among ALL generated; NSGA winner is from evolutionarily truncated pop of μ.

## 3. Per-run observed IDs (from logs)

- **nsga2_racing s123**: unique_seen=23 (init 8, child 15), winner=`racing_s123_g3_c07`, test_acc=0.82
- **nsga2_racing s42**: unique_seen=16 (init 8, child 8), winner=`racing_s42_g3_c02`, test_acc=0.78
- **nsga2_racing s456**: unique_seen=16 (init 8, child 8), winner=`racing_s456_g3_c06`, test_acc=0.775
- **nsga2_uniform s123**: unique_seen=19 (init 8, child 11), winner=`uniform_s123_g3_c06`, test_acc=0.775
- **nsga2_uniform s42**: unique_seen=19 (init 8, child 11), winner=`uniform_s42_g3_c03`, test_acc=0.765
- **nsga2_uniform s456**: unique_seen=18 (init 8, child 10), winner=`uniform_s456_g3_c00`, test_acc=0.775
- **random_search s123**: designed_pool=40, winner=`rand_s123_c25`, test_acc=0.795
- **random_search s42**: designed_pool=40, winner=`p15`, test_acc=0.8
- **random_search s456**: designed_pool=40, winner=`rand_s456_c25`, test_acc=0.88

## 4. Random Search text dedup (RNG replay, approximate)

- seed 42: n_texts=40, unique=34, dedup_rate=0.150
- seed 123: n_texts=40, unique=32, dedup_rate=0.200
- seed 456: n_texts=40, unique=36, dedup_rate=0.100

## 5. Search vs Test accuracy gap (NSGA logs only)

- nsga2_racing s123: search=1.000, test=0.820, gap=0.180
- nsga2_racing s42: search=1.000, test=0.780, gap=0.220
- nsga2_racing s456: search=1.000, test=0.775, gap=0.225
- nsga2_uniform s123: search=1.000, test=0.775, gap=0.225
- nsga2_uniform s42: search=0.889, test=0.765, gap=0.124
- nsga2_uniform s456: search=1.000, test=0.775, gap=0.225

## 6. Discussion bullets

- UNFAIR COMPARISON: Random Search evaluates a designed pool of 40 IDs (8+32); NSGA-II only creates 24 offspring (8×3) and truncates to 8 each generation.
- Random Search therefore has strictly more distinct candidates and a single late Uniform pass over the full pool — closer to 'screen many, pick best' than generational EA.
- Operator mismatch: Random uses mutate_prompt only; NSGA uses crossover+mutate.
- Test protocol OK: winner_only; selection on search estimates; Gate1 preload is search split only.
- NSGA winners are often late-generation children, so evolution did produce used candidates — failure is not 'mutation never runs'.
- Large search_acc−test_acc gaps on some NSGA winners (e.g. search_acc=1.0) suggest noisy/overfit search estimates under concentrated eval — aligns with exploitation-under-noise hypothesis.
- Do NOT claim NSGA inferior to Random on equal footing until candidate-count and operator are equalized (future controlled re-run, optional).
