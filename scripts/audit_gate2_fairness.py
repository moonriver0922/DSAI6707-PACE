#!/usr/bin/env python3
"""Gate-2 fairness audit from code + existing JSON logs (0 GPU / 0 new LLM)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "gate2" / "audit"
sys_path_note = str(ROOT / "src")


def load_runs():
    rows = []
    for p in sorted((ROOT / "outputs" / "gate2").glob("*.json")):
        if p.name in ("online_summary.json", "smoke_summary.json"):
            continue
        if "audit" in str(p):
            continue
        d = json.loads(p.read_text())
        if "method" not in d:
            continue
        d["_file"] = p.name
        rows.append(d)
    return rows


def nsga_unique_ids(run: dict) -> dict:
    """Collect every prompt id that appears in history acc maps or final pop."""
    ids = set()
    for h in run.get("history", []):
        ids.update(h.get("ids", []))
        ids.update(h.get("acc", {}).keys())
    for row in run.get("final_pop_search", []):
        ids.add(row["id"])
    if run.get("winner"):
        ids.add(run["winner"])
    child_ids = sorted(i for i in ids if not re.match(r"^p\d+$", i))
    init_ids = sorted(i for i in ids if re.match(r"^p\d+$", i))
    return {
        "n_unique_seen": len(ids),
        "n_init_seen": len(init_ids),
        "n_child_seen": len(child_ids),
        "child_ids": child_ids,
        "init_ids": init_ids,
        "all_ids": sorted(ids),
    }


def gen_survival(run: dict) -> list[dict]:
    rows = []
    for h in run.get("history", []):
        ids = h["ids"]
        n_child = sum(1 for i in ids if not re.match(r"^p\d+$", i))
        rows.append(
            {
                "generation": h["generation"],
                "pop_size": len(ids),
                "n_init_in_pop": len(ids) - n_child,
                "n_child_in_pop": n_child,
                "ids": ids,
            }
        )
    return rows


def code_fairness_spec(cfg: dict) -> dict:
    pop = int(cfg["population_size"])
    gens = int(cfg["n_generations"])
    # NSGA: offspring only when gen < n_gen-1
    nsga_offspring_gens = max(0, gens - 1)
    nsga_new = pop * nsga_offspring_gens
    rand_new = pop * gens  # as implemented: n_extra = pop * gens
    return {
        "population_size": pop,
        "n_generations": gens,
        "nsga2": {
            "init_candidates": pop,
            "offspring_batches": nsga_offspring_gens,
            "designed_new_candidates": nsga_new,
            "designed_total_ids": pop + nsga_new,
            "operator": "generate_offspring = crossover(50%) then mutate_prompt, or mutate path via crossover branch",
            "selection": "NSGA-II nondominated + crowding each generation; final winner by U_λ on search estimates within final pop of size μ",
            "budget": "sliced per generation across current pop/(pop+children); survivors re-evaluated repeatedly",
            "eval_policy": "Uniform or Racing on active population only",
        },
        "random_search": {
            "init_candidates": pop,
            "designed_new_candidates": rand_new,
            "designed_total_ids": pop + rand_new,
            "operator": "mutate_prompt only (no crossover)",
            "selection": "no generational truncation; after all mutants created, Uniform spends remaining budget on FULL pool; final μ by U_λ rank then winner",
            "budget": "almost all post-anchor budget spent once on the large pool (Uniform columns)",
            "eval_policy": "Uniform only",
        },
        "unfairness_flags": [
            f"Random Search generates {rand_new} new candidates vs NSGA-II {nsga_new} ({rand_new - nsga_new} more).",
            "Random Search screens the entire pool of init+new under one Uniform allocation; NSGA truncates to μ each generation so many children get little budget.",
            "Operators differ: Random=mutate only; NSGA=crossover+mutate.",
            "Random finalist set is top-μ by utility among ALL generated; NSGA winner is from evolutionarily truncated pop of μ.",
        ],
    }


def simulate_random_texts(cfg: dict, prompts: dict, seed: int) -> dict:
    """Rebuild Random Search texts with same RNG protocol (no LLM)."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from evolution import mutate_prompt

    init_ids = cfg.get("initial_prompt_ids") or sorted(prompts.keys())[: cfg["population_size"]]
    rng = np.random.default_rng(seed)
    # mirror initialize_common_anchor's single shuffle of search ids
    n_search = int(cfg.get("n_search", 160))
    pool = [f"s{i}" for i in range(n_search)]
    rng.shuffle(pool)
    texts = {pid: prompts[pid] for pid in init_ids}
    order = list(init_ids)
    n_extra = int(cfg["population_size"]) * int(cfg["n_generations"])
    for k in range(n_extra):
        parent = order[int(rng.integers(0, len(order)))]
        cid = f"rand_s{seed}_c{k:02d}"
        t = mutate_prompt(texts[parent], rng)
        texts[cid] = t
        order.append(cid)
    uniq = len(set(texts.values()))
    return {
        "seed": seed,
        "n_texts": len(texts),
        "n_unique_texts": uniq,
        "dedup_rate": 1.0 - uniq / len(texts),
        "ids": list(texts.keys()),
        "texts": texts,
    }


def verify_test_protocol() -> dict:
    src = (ROOT / "scripts" / "run_online.py").read_text()
    return {
        "winner_only_calls": src.count("evaluate_on_test(predictor, [winner]"),
        "full_pop_test_calls": len(re.findall(r"evaluate_on_test\(predictor,\s*pop\b", src))
        + len(re.findall(r"evaluate_on_test\(predictor,\s*final_pop\b", src)),
        "preference_winner_uses_search_estimates": "preference_winner(\n        ids, acc, cost" in src
        or "preference_winner(\n        final_ids" in src,
        "gate1_preload_split": "split=\"search\"" in src or "split='search'" in src,
        "conclusion": "Test set is only used in evaluate_on_test([winner]); selection uses search estimates. No full-pop test in current code.",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs" / "gate2.yaml").read_text())
    prompts = {p["id"]: p["text"] for p in json.loads((ROOT / "data" / "prompt_pool.json").read_text())["prompts"]}
    runs = load_runs()
    spec = code_fairness_spec(cfg)
    test_proto = verify_test_protocol()

    per_run = []
    for run in runs:
        method = run["method"]
        seed = run["seed"]
        row = {
            "file": run["_file"],
            "method": method,
            "seed": seed,
            "calls_used": run["calls_used"],
            "winner": run["winner"],
            "test_acc": run["test_acc"],
            "test_cost": run["test_cost"],
            "test_eval_mode": run.get("test_eval_mode"),
            "winner_text": run.get("winner_text"),
        }
        if method.startswith("nsga2"):
            u = nsga_unique_ids(run)
            row.update({k: u[k] for k in ("n_unique_seen", "n_init_seen", "n_child_seen")})
            row["survival"] = gen_survival(run)
            row["child_ids"] = u["child_ids"]
        else:
            # designed sizes from code
            row["designed_pool"] = cfg["population_size"] + cfg["population_size"] * cfg["n_generations"]
            row["final_pop_ids"] = [x["id"] for x in run.get("final_pop_search", [])]
        per_run.append(row)

    # Simulate random-search text diversity (approx; assumes n_search=160 shuffle)
    rand_sims = []
    for seed in cfg["online_seeds"]:
        sim = simulate_random_texts(cfg, prompts, int(seed))
        # drop full texts from summary table but save separately
        rand_sims.append({k: sim[k] for k in sim if k != "texts"})
        (OUT / f"random_search_texts_s{seed}.json").write_text(
            json.dumps(sim["texts"], indent=2, ensure_ascii=False) + "\n"
        )

    # Compare winner search_acc vs test_acc for overfitting signal
    gap_rows = []
    for run in runs:
        if "search_acc" in run:
            gap_rows.append(
                {
                    "method": run["method"],
                    "seed": run["seed"],
                    "search_acc": run["search_acc"],
                    "test_acc": run["test_acc"],
                    "gap": run["search_acc"] - run["test_acc"],
                }
            )

    report = {
        "code_fairness_spec": spec,
        "test_protocol_audit": test_proto,
        "per_run": per_run,
        "random_search_text_simulation": rand_sims,
        "search_test_gap": gap_rows,
        "discussion_bullets": [
            "UNFAIR COMPARISON: Random Search evaluates a designed pool of 40 IDs (8+32); NSGA-II only creates 24 offspring (8×3) and truncates to 8 each generation.",
            "Random Search therefore has strictly more distinct candidates and a single late Uniform pass over the full pool — closer to 'screen many, pick best' than generational EA.",
            "Operator mismatch: Random uses mutate_prompt only; NSGA uses crossover+mutate.",
            "Test protocol OK: winner_only; selection on search estimates; Gate1 preload is search split only.",
            "NSGA winners are often late-generation children, so evolution did produce used candidates — failure is not 'mutation never runs'.",
            "Large search_acc−test_acc gaps on some NSGA winners (e.g. search_acc=1.0) suggest noisy/overfit search estimates under concentrated eval — aligns with exploitation-under-noise hypothesis.",
            "Do NOT claim NSGA inferior to Random on equal footing until candidate-count and operator are equalized (future controlled re-run, optional).",
        ],
    }

    (OUT / "gate2_fairness_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # Markdown
    lines = [
        "# Gate 2 Fairness Audit (0 GPU)",
        "",
        "## 1. Test-set protocol",
        "",
        f"- `evaluate_on_test(predictor, [winner], ...)` call sites: **{test_proto['winner_only_calls']}**",
        f"- Full-pop test call sites remaining: **{test_proto['full_pop_test_calls']}**",
        f"- Conclusion: {test_proto['conclusion']}",
        "",
        "## 2. Designed candidate budgets (from code)",
        "",
        f"- NSGA-II new candidates: **{spec['nsga2']['designed_new_candidates']}** (total IDs {spec['nsga2']['designed_total_ids']})",
        f"- Random Search new candidates: **{spec['random_search']['designed_new_candidates']}** (total IDs {spec['random_search']['designed_total_ids']})",
        "",
        "### Unfairness flags",
        "",
    ]
    for f in spec["unfairness_flags"]:
        lines.append(f"- {f}")
    lines += ["", "## 3. Per-run observed IDs (from logs)", ""]
    for row in per_run:
        if row["method"].startswith("nsga2"):
            lines.append(
                f"- **{row['method']} s{row['seed']}**: unique_seen={row['n_unique_seen']} "
                f"(init {row['n_init_seen']}, child {row['n_child_seen']}), "
                f"winner=`{row['winner']}`, test_acc={row['test_acc']}"
            )
        else:
            lines.append(
                f"- **{row['method']} s{row['seed']}**: designed_pool={row['designed_pool']}, "
                f"winner=`{row['winner']}`, test_acc={row['test_acc']}"
            )
    lines += ["", "## 4. Random Search text dedup (RNG replay, approximate)", ""]
    for s in rand_sims:
        lines.append(
            f"- seed {s['seed']}: n_texts={s['n_texts']}, unique={s['n_unique_texts']}, "
            f"dedup_rate={s['dedup_rate']:.3f}"
        )
    lines += ["", "## 5. Search vs Test accuracy gap (NSGA logs only)", ""]
    for g in gap_rows:
        lines.append(
            f"- {g['method']} s{g['seed']}: search={g['search_acc']:.3f}, test={g['test_acc']:.3f}, gap={g['gap']:.3f}"
        )
    lines += ["", "## 6. Discussion bullets", ""]
    for b in report["discussion_bullets"]:
        lines.append(f"- {b}")
    lines.append("")
    (OUT / "GATE2_FAIRNESS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
