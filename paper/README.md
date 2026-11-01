# PACE paper

**Guosheng Wang · 24144419R**

[Read the PDF](main.pdf) · [Main TeX source](main.tex) ·
[Literature source notes](LITERATURE_REVIEW_NOTES.md)

The paper uses the unmodified official NeurIPS 2026 style in preprint mode.
It includes a classified review of 20 studies, method details, experiments,
archive analysis and full per-seed results. Every section has at most five
subsections. The page count includes references and appendices.

## Compile

From this directory:

```bash
tectonic main.tex
# Alternative: latexmk -pdf main.tex
```

For Overleaf, upload `main.tex`, `neurips_2026.sty`, `sections/` and `figures/`.
Set `main.tex` as the main document. The bibliography is embedded in the source.
The official style controls margins and fonts; ragged-bottom page filling avoids
stretching gaps between paragraphs and headings.

Official template: [NeurIPS 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip),
linked from the [conference call for papers](https://neurips.cc/Conferences/2026/CallForPapers).

## Source files

| File | Contents |
| --- | --- |
| `main.tex` | Abstract, main method and results, discussion, bibliography |
| `sections/literature.tex` | Review and comparison table for 20 core studies |
| `sections/background.tex` | Sampling rationale, dominance and archive design |
| `sections/paired_evidence_details.tex` | Bootstrap construction and boundary cases |
| `sections/algorithm.tex` | Survivor procedure, variation and evaluation scope |
| `sections/dataset_details.tex` | Data split and output parsing |
| `sections/protocol.tex` | Racing, resource accounting and seed-level inference |
| `sections/extended_results.tex` | Replay, trajectories, archive audit and sensitivity |
| `sections/appendix_results.tex` | Complete paired results and winner examples |

The review groups seven generation/refinement studies, five cost/allocation
studies, two structured-search studies, three evaluation studies and three noisy
selection studies. Three surveys and NSGA-II provide additional background.

## Figures

From the repository root:

```bash
python paper/make_figures.py
```

The script reads `outputs/pace/`, checks the 30 primary online records, and writes
seven vector PDFs plus PNG previews. Compilation only requires the generated
figures; regenerating them requires the stored experiment records.

| Figure | Display |
| --- | --- |
| `method_overview` | Search and survivor-selection flow |
| `offline_replay` | Mean false elimination and retention at budget 640 |
| `online_paired` | Per-seed bars and the primary mean confidence interval |
| `accuracy_cost` | Accuracy versus deployment tokens for each winner |
| `replay_budgets` | Replay means and sample standard deviations at both budgets |
| `search_dynamics` | Mean observation and retained-set trajectories |
| `seed_sensitivity` | Cohort and leave-one-seed-out means |

Figures use a blue/gray palette, common chart forms, embedded fonts and explicit
units. Recorded results are unchanged. Sensitivity plots are descriptive analyses
of those records, not new experiments.

## Checks

Tectonic is used to check compilation, citations and layout. Section counts,
figure paths, dataset split counts and primary statistics are checked against
the stored files. The source package is also compiled after extraction into a
separate directory. A fresh GPU run is not part of the manuscript build.
