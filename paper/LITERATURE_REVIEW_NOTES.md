# Literature review source notes

The paper compares 20 core studies listed in its bibliography.
Three surveys and NSGA-II are additional background, outside that count.

Publication landing pages and their abstracts were checked on 2026-10-28 using
ICLR proceedings, PMLR, ACL Anthology, arXiv, AAAI, Springer and the publisher/
PubMed record for the robotics paper. The review is a focused conceptual
synthesis based on those records, not a systematic review or a full-text
replication of the papers. It does not claim their reported improvements are
comparable with PACE. Taxonomy assignments and the PACE-connection column are
editorial analysis, not claims made by the cited authors.

Important distinctions retained: BudgetAPO already uses paired comparisons;
IAPO's error guarantees do not apply to PACE; genetic drift in the selection-noise
paper differs from noisy estimated fitness; MO-CAPO and BudgetAPO are cited as
preprints.

## Core-study sources

- `evoprompt`: https://proceedings.iclr.cc/paper_files/paper/2024/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html
- `promptbreeder`: https://proceedings.mlr.press/v235/fernando24a.html
- `capo`: https://proceedings.mlr.press/v293/zehle25a.html
- `mocapo`: https://arxiv.org/abs/2605.18869
- `gepa`: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html
- `opro`: https://arxiv.org/abs/2309.03409
- `opro_revisit`: https://aclanthology.org/2024.findings-acl.100/
- `instructzero`: https://proceedings.mlr.press/v235/chen24e.html
- `promptwizard`: https://aclanthology.org/2025.findings-acl.1025/
- `riot`: https://aclanthology.org/2025.acl-long.1086/
- `budgetapo`: https://arxiv.org/abs/2610.05671
- `ipomp`: https://aclanthology.org/2025.findings-acl.147/
- `probsel`: https://aclanthology.org/2024.tacl-1.37/
- `iapo`: https://ojs.aaai.org/index.php/AAAI/article/view/40521
- `autopdl`: https://arxiv.org/abs/2504.04365
- `toolbox`: https://aclanthology.org/2025.luhme-1.6/
- `depth`: https://proceedings.mlr.press/v306/wang26ff.html
- `selectionnoise`: https://link.springer.com/article/10.1007/s00500-025-10652-6
- `robotnoise`: https://pubmed.ncbi.nlm.nih.gov/41604289/
- `evalbefore`: https://aclanthology.org/2025.acl-industry.44/

For the robotics article, the publisher record was also checked at
https://doi.org/10.1162/EVCO.a.381 .
