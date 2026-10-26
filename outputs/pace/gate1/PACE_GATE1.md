# PACE Gate 1 Results (0 new LLM)

Reference Pareto (audit-only full search matrix): `['p00', 'p01', 'p02', 'p03', 'p05', 'p15']`

## Primary budget 640 — mean±std over 20 replay seeds

| method        |   ('false_elim_rate', 'mean') |   ('false_elim_rate', 'std') |   ('ref_pareto_recall', 'mean') |   ('ref_pareto_recall', 'std') |   ('retained_regret', 'mean') |   ('retained_regret', 'std') |   ('n_retained', 'mean') |   ('n_retained', 'std') |   ('n_conf_elim', 'mean') |   ('n_conf_elim', 'std') |   ('n_forced', 'mean') |   ('n_forced', 'std') |
|:--------------|------------------------------:|-----------------------------:|--------------------------------:|-------------------------------:|------------------------------:|-----------------------------:|-------------------------:|------------------------:|--------------------------:|-------------------------:|-----------------------:|----------------------:|
| b0_point_nsga |                        0.1833 |                       0.1197 |                          0.8167 |                         0.1197 |                        0.0007 |                       0.0018 |                      8   |                    0    |                       0   |                     0    |                      0 |                     0 |
| pace          |                        0.025  |                       0.0611 |                          0.975  |                         0.0611 |                        0      |                       0      |                     11.7 |                    2.08 |                       4.3 |                     2.08 |                      0 |                     0 |

## GO/NO-GO

- B0 mean false-elim rate (H0): **0.1833** (nontrivial≥0.05? True)
- Δ false-elim (PACE−B0): **-0.1583** (want <0)
- Δ recall (PACE−B0): **0.1583**
- Δ retained regret (PACE−B0): **-0.0007** (want ≤0.002)
- **Decision: GO**

DICE historical negatives remain unchanged and are not PACE wins.
