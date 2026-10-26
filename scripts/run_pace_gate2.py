#!/usr/bin/env python3
"""PACE Gate 2 — online NSGA-II with matched Racing evaluation.

Survivor-only contrast (same seeds / operators / calls_cap / Racing purchases):
  B0   = point_estimate_truncate (NSGA rank+crowding on point estimates)
  PACE = pace_bounded_survive (evidence-gated + bounded archive)

Each generation: one Racing purchase pass on the current candidate pool, then
survivor selection on that shared observation state. Confidence eliminations and
forced capacity evictions are logged separately. Winner is chosen from search
observations only; sealed test is evaluated for the winner afterward.

Does not modify DICE weights or rewrite historical DICE negatives.
Outputs → outputs/pace/gate2/ (Gate-1 / reports untouched).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache import DecodeConfig, EvaluationCache  # noqa: E402
from dataset import load_json, load_prompt_pool  # noqa: E402
from evolution import (  # noqa: E402
    EvolutionConfig,
    Individual,
    generate_offspring,
)
from inference import QwenPredictor, preload_gate1_matrix_into_cache  # noqa: E402
from metrics import preference_winner  # noqa: E402
from pace.survivor import pace_bounded_survive, point_estimate_truncate  # noqa: E402
from policies import get_policy  # noqa: E402
from replay import apply_queries, initialize_common_anchor  # noqa: E402
from state import TAG_ANCHOR, TAG_PAIR, TAG_UNIFORM, SearchState, empty_state  # noqa: E402


class LiveOracle:
    """Oracle backed by live LLM calls + cache (search or test)."""

    def __init__(self, predictor: QwenPredictor, samples: list[dict], prompt_texts: dict[str, str]):
        self.predictor = predictor
        self.samples = {r["sample_id"]: r for r in samples}
        self.sample_ids = tuple(r["sample_id"] for r in samples)
        self.prompt_texts = dict(prompt_texts)
        self.prompt_ids = tuple(prompt_texts.keys())

    def prefetch(self, queries: list[tuple[str, str]]) -> None:
        items = []
        for pid, sid in queries:
            row = self.samples[sid]
            items.append(
                (pid, self.prompt_texts[pid], sid, row["text"], int(row["label_id"]))
            )
        if items:
            self.predictor.predict_batch(items)

    def lookup(self, prompt_id: str, sample_id: str) -> tuple[int, int, int]:
        row = self.samples[sample_id]
        pred = self.predictor.predict(
            prompt_id,
            self.prompt_texts[prompt_id],
            sample_id,
            row["text"],
            int(row["label_id"]),
        )
        return int(pred.correct), int(pred.input_tokens), int(pred.output_tokens)

    def add_prompt(self, prompt_id: str, text: str) -> None:
        self.prompt_texts[prompt_id] = text
        if prompt_id not in self.prompt_ids:
            self.prompt_ids = tuple(list(self.prompt_ids) + [prompt_id])


def ensure_prompts_in_state(state: SearchState, individuals: list[Individual]) -> SearchState:
    texts = dict(state.prompt_texts)
    ids = list(state.prompt_ids)
    for ind in individuals:
        if ind.prompt_id not in texts:
            texts[ind.prompt_id] = ind.text
            ids.append(ind.prompt_id)
        else:
            texts[ind.prompt_id] = ind.text
    state.prompt_texts = texts
    state.prompt_ids = tuple(ids)
    return state


def allocate_racing(
    state: SearchState,
    oracle: LiveOracle,
    max_new_calls: int,
) -> int:
    """Spend up to max_new_calls with Paired Racing (one purchase trajectory)."""
    if max_new_calls <= 0:
        return 0
    hard_cap = state.calls_used + max_new_calls
    old_budget = state.calls_budget
    state.calls_budget = min(state.calls_budget, hard_cap)
    policy = get_policy("racing")
    start = state.calls_used
    idle = 0
    while state.remaining_calls() > 0 and state.calls_used < hard_cap:
        rem = min(state.remaining_calls(), hard_cap - state.calls_used)
        queries = policy.select_queries(state, rem)
        if not queries:
            idle += 1
            if idle >= 3:
                break
            continue
        idle = 0
        affordable = []
        cost = 0
        for q in queries:
            if q in state.observed:
                continue
            if cost + 1 > rem:
                break
            affordable.append(q)
            cost += 1
        if not affordable:
            break
        by_s: dict[str, set[str]] = {}
        for pid, sid in affordable:
            by_s.setdefault(sid, set()).add(pid)
        tag = TAG_UNIFORM if any(len(v) >= 3 for v in by_s.values()) else TAG_PAIR
        apply_queries(state, oracle, affordable, tag)
    state.calls_budget = old_budget
    return state.calls_used - start


def estimates_for(state: SearchState, ids: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    acc, cost = {}, {}
    for pid in ids:
        a = state.accuracy_pop(pid)
        c = state.cost_pop(pid)
        if np.isnan(a):
            a = state.accuracy_all(pid)
        if np.isnan(c):
            c = state.cost_all(pid)
        acc[pid] = 0.0 if np.isnan(a) else float(a)
        cost[pid] = 1e9 if np.isnan(c) else float(c)
    return acc, cost


def evaluate_on_test(
    predictor: QwenPredictor,
    individuals: list[Individual],
    test_rows: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    items = []
    index = []
    for ind in individuals:
        for row in test_rows:
            items.append(
                (ind.prompt_id, ind.text, row["sample_id"], row["text"], int(row["label_id"]))
            )
            index.append(ind.prompt_id)
    preds = predictor.predict_batch(items)
    buckets: dict[str, list[tuple[int, int]]] = {ind.prompt_id: [] for ind in individuals}
    for pid, pred in zip(index, preds):
        buckets[pid].append((pred.correct, pred.input_tokens + pred.output_tokens))
    acc, cost = {}, {}
    for pid, rows in buckets.items():
        acc[pid] = float(np.mean([r[0] for r in rows])) if rows else float("nan")
        cost[pid] = float(np.mean([r[1] for r in rows])) if rows else float("nan")
    return acc, cost


def survive(
    method: str,
    state: SearchState,
    candidates: list[str],
    cfg: dict,
    rng: np.random.Generator,
):
    if method == "b0_point_nsga":
        return point_estimate_truncate(state, candidates, mu=int(cfg["mu"]))
    # b0_archive / a1_no_gate: same μ+K capacity as PACE, confidence gate off
    # (cap-align fair ablation: only selection rule differs).
    if method in ("pace", "a1_no_gate", "b0_archive"):
        return pace_bounded_survive(
            state,
            candidates,
            mu=int(cfg["mu"]),
            archive_k=int(cfg["archive_k"]),
            epsilon=float(cfg["epsilon"]),
            n_min_common=int(cfg["n_min_common"]),
            n_bootstrap=int(cfg["n_bootstrap"]),
            rng=rng,
            disable_confidence_gate=(method in ("a1_no_gate", "b0_archive")),
        )
    raise ValueError(f"Unknown method: {method}")


def warm_start_children(
    state: SearchState,
    oracle: LiveOracle,
    children: list[Individual],
    rng: np.random.Generator,
    n_warm: int = 4,
) -> int:
    unused = list(state.sample_ids)
    rng.shuffle(unused)
    warm = unused[: min(n_warm, len(unused))]
    warm_q = [(ch.prompt_id, s) for s in warm for ch in children]
    afford = []
    left = state.remaining_calls()
    for q in warm_q:
        if q in state.observed:
            continue
        if left <= 0:
            break
        afford.append(q)
        left -= 1
    if afford:
        apply_queries(state, oracle, afford, TAG_ANCHOR)
    return len(afford)


def run_method(
    *,
    method: str,
    seed: int,
    cfg: dict,
    init_inds: list[Individual],
    search_rows: list[dict],
    test_rows: list[dict],
    predictor: QwenPredictor,
) -> dict:
    rng = np.random.default_rng(seed)
    # Retained pool may exceed μ under PACE (active ∪ archive); breed from active.
    retained = [Individual(i.prompt_id, i.text, 0) for i in init_inds]
    active = list(retained)
    texts = {i.prompt_id: i.text for i in retained}
    oracle = LiveOracle(predictor, search_rows, texts)
    state = empty_state(
        list(texts.keys()),
        [r["sample_id"] for r in search_rows],
        texts,
        calls_budget=int(cfg["calls_cap"]),
        rng_seed=seed,
        cost_norm_min=float(cfg["cost_norm_min"]),
        cost_norm_max=float(cfg["cost_norm_max"]),
        n_bootstrap=int(cfg["n_bootstrap"]),
        n_contenders=int(cfg["n_contenders"]),
        max_pairs=int(cfg["max_pairs"]),
        pair_batch_samples=int(cfg["pair_batch_samples"]),
        racing_stop_r=float(cfg["racing_stop_r"]),
        lambdas=tuple(cfg["lambdas"]),
    )
    initialize_common_anchor(state, oracle, int(cfg.get("init_common", 8)), rng)

    history = []
    n_gen = int(cfg["n_generations"])
    pop_size = int(cfg["population_size"])
    evo_cfg = EvolutionConfig(population_size=pop_size, n_generations=n_gen)
    assert evo_cfg.population_size == int(cfg["mu"])

    def _apply_survive(pool: list[Individual], gen: int, phase: str, spent: int, warm_n: int = 0):
        nonlocal retained, active
        ids = [ind.prompt_id for ind in pool]
        result = survive(method, state, ids, cfg, np.random.default_rng(10_000 + seed * 100 + gen))
        by_id = {ind.prompt_id: ind for ind in pool}
        retained = [by_id[pid] for pid in result.retained if pid in by_id]
        active = [by_id[pid] for pid in result.active if pid in by_id]
        order = {pid: i for i, pid in enumerate(result.active)}
        active.sort(key=lambda ind: order.get(ind.prompt_id, 10**9))
        hist = {
            "generation": gen,
            "phase": phase,
            "racing_calls": spent,
            "warm_calls": warm_n,
            "calls_used": state.calls_used,
            "n_candidates": len(ids),
            "candidate_ids": ids,
            "active": [s.prompt_id for s in active],
            "retained": [s.prompt_id for s in retained],
            "confidence_elims": [
                {
                    "eliminated": e.eliminated,
                    "by": e.by,
                    "L": e.L,
                    "U": e.U,
                    "delta": e.delta_hat,
                    "n": e.n_common,
                }
                for e in result.confidence_elims
            ],
            "forced_evictions": [
                {"prompt_id": e.prompt_id, "reason": e.reason} for e in result.forced_evictions
            ],
            "n_conf_elim": len(result.confidence_elims),
            "n_forced": len(result.forced_evictions),
        }
        history.append(hist)
        print(
            f"  [{method} s{seed} g{gen} {phase}] racing+{spent} "
            f"conf_elim={hist['n_conf_elim']} forced={hist['n_forced']} "
            f"active={len(active)} retained={len(retained)}",
            flush=True,
        )
        return hist

    # n_gen with (n_gen-1) offspring rounds → 8 init + 24 generated when pop=8, gen=4
    for gen in range(n_gen):
        rem = state.remaining_calls()
        gens_left = n_gen - gen
        slice_budget = rem // gens_left if gens_left else rem
        # One Racing purchase trajectory on current retained, then survivor
        spent = allocate_racing(state, oracle, slice_budget)
        _apply_survive(retained, gen, "racing_then_survive", spent)

        if gen == n_gen - 1:
            break

        children = generate_offspring(
            active,
            rng,
            pop_size,
            id_prefix=f"{method}_s{seed}",
            gen=gen + 1,
        )
        for ch in children:
            oracle.add_prompt(ch.prompt_id, ch.text)
        ensure_prompts_in_state(state, retained + children)
        warm_n = warm_start_children(
            state, oracle, children, rng, n_warm=min(4, int(cfg.get("init_common", 8)))
        )
        combined = retained + children
        rem = state.remaining_calls()
        slice_budget = rem // max(1, (n_gen - gen))
        spent = allocate_racing(state, oracle, slice_budget)
        _apply_survive(combined, gen, "offspring_racing_then_survive", spent, warm_n=warm_n)

    # Spend any leftover search budget on final retained (still Racing; no test leak)
    leftover = state.remaining_calls()
    if leftover > 0:
        spent = allocate_racing(state, oracle, leftover)
        history.append(
            {
                "generation": n_gen,
                "phase": "leftover_racing",
                "racing_calls": spent,
                "calls_used": state.calls_used,
            }
        )

    # Winner from search observations only (active ∪ retained estimates)
    pool = retained if retained else active
    ids = [ind.prompt_id for ind in pool]
    acc, cost = estimates_for(state, ids)
    lam = float(cfg["primary_lambda"])
    winner_id = preference_winner(
        ids, acc, cost, lam, float(cfg["cost_norm_min"]), float(cfg["cost_norm_max"])
    )
    winner = next(ind for ind in pool if ind.prompt_id == winner_id)

    test_acc, test_cost = evaluate_on_test(predictor, [winner], test_rows)

    return {
        "method": method,
        "eval_policy": "racing",
        "seed": seed,
        "calls_used": state.calls_used,
        "calls_cap": int(cfg["calls_cap"]),
        "winner": winner_id,
        "winner_text": winner.text,
        "search_acc": acc.get(winner_id),
        "search_cost": cost.get(winner_id),
        "test_acc": test_acc[winner_id],
        "test_cost": test_cost[winner_id],
        "test_eval_mode": "winner_only_sealed",
        "n_generated_approx": pop_size * max(0, n_gen - 1),
        "final_active": [ind.prompt_id for ind in active],
        "final_retained": [ind.prompt_id for ind in retained],
        "final_pop_search": [
            {"id": ind.prompt_id, "search_acc": acc[ind.prompt_id], "search_cost": cost[ind.prompt_id]}
            for ind in pool
        ],
        "history": history,
        "note": "DICE historical negatives unchanged; survivor-only contrast vs B0.",
    }


def apply_smoke_overrides(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["online_seeds"] = [42]
    cfg["n_generations"] = 2
    cfg["population_size"] = 4
    cfg["mu"] = 4
    # Preserve intentional archive_k=0 (μ-only / A1 capacity match); else shrink with smoke.
    if int(cfg.get("archive_k", 4)) > 0:
        cfg["archive_k"] = 4
    cfg["calls_cap"] = 256
    cfg["init_common"] = 4
    cfg["n_bootstrap"] = min(int(cfg.get("n_bootstrap", 200)), 40)
    cfg["initial_prompt_ids"] = list(cfg.get("initial_prompt_ids", []))[:4]
    return cfg


def _load_baseline_runs(path: Path | None) -> dict[tuple[str, int], dict]:
    """Index original gate2 runs by (method, seed) for Δ reporting."""
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {(r["method"], int(r["seed"])): r for r in data.get("runs", [])}


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def write_report(
    *,
    out_dir: Path,
    title: str,
    experiment: str,
    args,
    cfg: dict,
    methods: list[str],
    results: list[dict],
    method_summary: dict,
    summary: dict,
    baseline_runs: dict[tuple[str, int], dict],
    baseline_summary: dict | None,
) -> None:
    md = [
        f"# {title}",
        "",
        f"Experiment: `{experiment}` · Smoke: `{args.smoke}` · eval=`racing`",
        "",
        f"- μ={cfg['mu']}, K={cfg['archive_k']}, ε={cfg['epsilon']}, "
        f"n_min={cfg['n_min_common']}, B={cfg['n_bootstrap']}",
        f"- pop={cfg['population_size']}, gen={cfg['n_generations']}, "
        f"calls_cap={cfg['calls_cap']}, seeds={cfg['online_seeds']}",
        f"- gate1_warm_cells={summary['gate1_warm_cells']}",
        f"- disable_confidence_gate methods: "
        f"{[m for m in methods if m in ('a1_no_gate', 'b0_archive')] or 'none'}",
        "",
        "## Method means",
        "",
    ]
    for m, s in method_summary.items():
        md.append(
            f"- **{m}**: test_acc={s['mean_test_acc']:.4f}, "
            f"test_cost={s['mean_test_cost']:.1f}, calls={s['mean_calls']:.0f}, "
            f"conf_elim={s['mean_conf_elim']:.1f}, forced={s['mean_forced']:.1f} (n={s['n']})"
        )

    # Within-experiment Δ vs capacity-matched baseline (b0_archive) or classic B0
    baseline_key = next(
        (k for k in ("b0_archive", "b0_point_nsga") if k in method_summary),
        None,
    )
    if baseline_key is not None:
        b0 = method_summary[baseline_key]
        md.extend(["", f"## Within-run Δ vs {baseline_key} (this experiment)", ""])
        for m, s in method_summary.items():
            if m == baseline_key:
                continue
            md.append(
                f"- **{m} − {baseline_key}**: Δacc={s['mean_test_acc'] - b0['mean_test_acc']:+.4f}, "
                f"Δcost={s['mean_test_cost'] - b0['mean_test_cost']:+.1f}, "
                f"Δcalls={s['mean_calls'] - b0['mean_calls']:+.0f}"
            )

    # vs original gate2
    if baseline_summary and baseline_summary.get("method_summary"):
        md.extend(["", "## Δ vs original `outputs/pace/gate2`", ""])
        base_ms = baseline_summary["method_summary"]
        for m, s in method_summary.items():
            if m not in base_ms:
                # map a1 → compare to original pace or b0 as noted
                continue
            bm = base_ms[m]
            md.append(
                f"- **{m}**: Δacc={s['mean_test_acc'] - bm['mean_test_acc']:+.4f} "
                f"(this {s['mean_test_acc']:.4f} vs gate2 {bm['mean_test_acc']:.4f}), "
                f"Δcost={s['mean_test_cost'] - bm['mean_test_cost']:+.1f}, "
                f"Δcalls={s['mean_calls'] - bm['mean_calls']:+.0f}"
            )
        if "a1_no_gate" in method_summary and "b0_point_nsga" in base_ms:
            s = method_summary["a1_no_gate"]
            bm = base_ms["b0_point_nsga"]
            md.append(
                f"- **a1_no_gate vs gate2 B0**: Δacc={s['mean_test_acc'] - bm['mean_test_acc']:+.4f}, "
                f"Δcost={s['mean_test_cost'] - bm['mean_test_cost']:+.1f}, "
                f"Δcalls={s['mean_calls'] - bm['mean_calls']:+.0f} "
                f"(A1 should track B0)"
            )

    md.extend(
        [
            "",
            "## Runs (per seed)",
            "",
            "| method | seed | calls | test_acc | test_cost | conf_elim | forced | "
            "Δacc_vs_gate2 | Δcalls_vs_gate2 |",
            "|--------|------|-------|----------|-----------|-----------|--------|"
            "--------------|-----------------|",
        ]
    )
    for row in summary["runs"]:
        key = (row["method"], int(row["seed"]))
        # A1 alignment: compare to baseline B0 same seed when present
        base_key = key
        if row["method"] == "a1_no_gate":
            base_key = ("b0_point_nsga", int(row["seed"]))
        br = baseline_runs.get(base_key) or baseline_runs.get(key)
        d_acc = _delta(row["test_acc"], br["test_acc"]) if br else None
        d_calls = _delta(row["calls_used"], br["calls_used"]) if br else None
        d_acc_s = f"{d_acc:+.4f}" if d_acc is not None else "—"
        d_calls_s = f"{d_calls:+.0f}" if d_calls is not None else "—"
        md.append(
            f"| {row['method']} | {row['seed']} | {row['calls_used']} | "
            f"{row['test_acc']:.4f} | {row['test_cost']:.1f} | "
            f"{row['n_conf_elim_total']} | {row['n_forced_total']} | "
            f"{d_acc_s} | {d_calls_s} |"
        )

    md.extend(
        [
            "",
            "Capacity note: retained upper bound = μ+K; forced≫conf_elim implies capacity "
            "pressure. μ-only (K=0) removes μ+K confound. A1 = gate off (conf_elim=0).",
            "",
            "DICE historical negatives and `outputs/gate1` / `reports` remain unchanged. "
            "Original `outputs/pace/gate2/` not overwritten.",
            "",
        ]
    )
    (out_dir / "PACE_GATE2.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description="PACE Gate 2 online (Racing + B0 vs PACE survivor)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pace_gate2.yaml")
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompt_pool.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "split_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "pace" / "gate2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="Small gen/budget wiring check")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override methods (default from config). "
        "Known: b0_point_nsga, pace, a1_no_gate, b0_archive",
    )
    ap.add_argument(
        "--gate1-npz",
        type=Path,
        default=ROOT / "outputs" / "gate1" / "oracle_matrices.npz",
        help="Warm cache for initial prompts from Gate-1 matrix (read-only)",
    )
    ap.add_argument(
        "--no-gate1-warm",
        action="store_true",
        help="Cold start: do not preload Gate-1 matrix (gate1_warm_cells=0)",
    )
    ap.add_argument(
        "--archive-k",
        type=int,
        default=None,
        help="Override archive_k (K). Use 0 for μ-only retained cap (capacity confound control)",
    )
    ap.add_argument(
        "--experiment",
        type=str,
        default="gate2",
        help="Label for summary/MD (e.g. gate2_cold, gate2_a1, gate2_mu_only)",
    )
    ap.add_argument(
        "--baseline-summary",
        type=Path,
        default=ROOT / "outputs" / "pace" / "gate2" / "pace_gate2_summary.json",
        help="Original gate2 summary.json for Δ columns (optional)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="PACE Gate 2 Results",
        help="Markdown report title",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override online_seeds (e.g. fixed 15-seed final set). Protocol otherwise unchanged.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse {method}_s{seed}.json in --out-dir if present (for resume / merge prior seeds).",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.archive_k is not None:
        cfg["archive_k"] = int(args.archive_k)
    if args.seeds is not None:
        cfg["online_seeds"] = [int(s) for s in args.seeds]
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    methods = list(args.methods or cfg.get("methods", ["b0_point_nsga", "pace"]))

    prompts = load_prompt_pool(args.prompts)
    init_ids = cfg.get("initial_prompt_ids") or sorted(prompts.keys())[: int(cfg["population_size"])]
    init_inds = [Individual(pid, prompts[pid], 0) for pid in init_ids]
    manifest = load_json(args.manifest)
    search_rows = manifest["search"]
    test_rows = manifest["test"]
    if args.smoke:
        test_rows = test_rows[:40]

    decode = DecodeConfig(max_new_tokens=int(cfg["max_new_tokens"]), do_sample=False)
    cache = EvaluationCache()
    n_warm = 0
    if args.no_gate1_warm:
        print("Cold start: --no-gate1-warm set; gate1_warm_cells=0", flush=True)
    elif args.gate1_npz.is_file():
        n_warm = preload_gate1_matrix_into_cache(
            cache, cfg["model_id"], decode, args.gate1_npz, prompt_ids=init_ids, split="search"
        )
        print(f"Warmed {n_warm} Gate-1 search cells (read-only) for init {init_ids}", flush=True)
    else:
        print(f"No Gate-1 npz at {args.gate1_npz}; starting cold cache", flush=True)

    predictor = QwenPredictor(
        cfg["model_id"],
        decode,
        cache,
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=args.batch_size,
        device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse writing into legacy DICE/EC trees
    resolved = args.out_dir.resolve()
    for banned in (ROOT / "outputs" / "gate1", ROOT / "outputs" / "gate2", ROOT / "reports"):
        if resolved == banned.resolve() or banned.resolve() in resolved.parents:
            raise SystemExit(f"Refusing to write under frozen path {banned}")
    # Do not overwrite the canonical warm Gate2 tree unless explicitly targeting it
    canonical = (ROOT / "outputs" / "pace" / "gate2").resolve()
    if resolved == canonical and args.experiment not in ("gate2", "gate2_smoke"):
        raise SystemExit(
            f"Refusing to write experiment={args.experiment!r} into canonical {canonical}; "
            "use a separate --out-dir"
        )

    baseline_runs = _load_baseline_runs(args.baseline_summary)
    baseline_summary = None
    if args.baseline_summary and args.baseline_summary.is_file():
        baseline_summary = json.loads(args.baseline_summary.read_text())

    results = []
    t0 = time.time()
    for seed in cfg["online_seeds"]:
        for method in methods:
            out_path = args.out_dir / f"{method}_s{seed}.json"
            if args.skip_existing and out_path.is_file():
                r = json.loads(out_path.read_text())
                results.append(r)
                print(
                    f"=== PACE Gate2 [{args.experiment}] method={method} seed={seed} "
                    f"SKIP existing {out_path.name} ===",
                    flush=True,
                )
                continue
            print(
                f"=== PACE Gate2 [{args.experiment}] method={method} seed={seed} "
                f"K={cfg['archive_k']} warm={n_warm} ===",
                flush=True,
            )
            r = run_method(
                method=method,
                seed=int(seed),
                cfg=cfg,
                init_inds=init_inds,
                search_rows=search_rows,
                test_rows=test_rows,
                predictor=predictor,
            )
            results.append(r)
            out_path.write_text(json.dumps(r, indent=2) + "\n")
            slim = {k: r[k] for k in r if k not in ("history", "winner_text", "final_pop_search")}
            print(json.dumps(slim, indent=2), flush=True)

    # Per-method means
    by_m: dict[str, list[dict]] = {}
    for r in results:
        by_m.setdefault(r["method"], []).append(r)
    method_summary = {}
    run_rows = []
    for r in results:
        n_conf = sum(h.get("n_conf_elim", 0) for h in r["history"])
        n_forced = sum(h.get("n_forced", 0) for h in r["history"])
        run_rows.append(
            {
                "method": r["method"],
                "seed": r["seed"],
                "calls_used": r["calls_used"],
                "test_acc": r["test_acc"],
                "test_cost": r["test_cost"],
                "search_acc": r["search_acc"],
                "winner": r["winner"],
                "n_conf_elim_total": n_conf,
                "n_forced_total": n_forced,
            }
        )
    for m, rows in by_m.items():
        confs = [sum(h.get("n_conf_elim", 0) for h in x["history"]) for x in rows]
        forceds = [sum(h.get("n_forced", 0) for h in x["history"]) for x in rows]
        method_summary[m] = {
            "n": len(rows),
            "mean_test_acc": float(np.mean([x["test_acc"] for x in rows])),
            "mean_test_cost": float(np.mean([x["test_cost"] for x in rows])),
            "mean_search_acc": float(np.mean([x["search_acc"] for x in rows])),
            "mean_calls": float(np.mean([x["calls_used"] for x in rows])),
            "mean_conf_elim": float(np.mean(confs)),
            "mean_forced": float(np.mean(forceds)),
            "seeds": [x["seed"] for x in rows],
        }

    treatment = {
        "gate2": "survivor_only (b0_point_nsga vs pace_bounded_survive)",
        "gate2_cold": "cold_start no Gate1 warm; survivor_only B0 vs PACE",
        "gate2_a1": "A1 ablation: confidence gate off (a1_no_gate); should track B0",
        "gate2_mu_only": "μ-only retained cap (K=0); capacity confound control B0 vs PACE",
        "gate2_cap_align": (
            "capacity-aligned fair ablation: b0_archive (μ+K, no gate) vs pace (μ+K, gate); "
            "same retained cap, only selection rule differs"
        ),
        "gate2_cap_align_15seed": (
            "PRIMARY online extension (15 fixed seeds): b0_archive vs pace, "
            "capacity-aligned μ+K; exploratory 3-seed retained; protocol frozen"
        ),
    }.get(args.experiment, f"experiment={args.experiment}")

    summary = {
        "track": "PACE Gate 2",
        "experiment": args.experiment,
        "smoke": bool(args.smoke),
        "config": str(args.config),
        "eval_policy": "racing",
        "methods": methods,
        "treatment": treatment,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_cache": len(cache),
        "gate1_warm_cells": n_warm,
        "no_gate1_warm": bool(args.no_gate1_warm),
        "batch_size": args.batch_size,
        "calls_cap": int(cfg["calls_cap"]),
        "n_generations": int(cfg["n_generations"]),
        "population_size": int(cfg["population_size"]),
        "epsilon": float(cfg["epsilon"]),
        "n_min_common": int(cfg["n_min_common"]),
        "n_bootstrap": int(cfg["n_bootstrap"]),
        "mu": int(cfg["mu"]),
        "archive_k": int(cfg["archive_k"]),
        "method_summary": method_summary,
        "runs": run_rows,
        "baseline_summary": str(args.baseline_summary) if args.baseline_summary else None,
        "frozen_untouched": ["outputs/gate1", "outputs/gate2", "reports", "outputs/pace/gate2"],
        "note": (
            "Conditional online proof after PACE Gate1 GO. "
            "DICE NO-GO and EC Gate2 artifacts are historical; not relabeled as PACE wins. "
            "Capacity confound (μ+K) must be stated explicitly."
        ),
    }
    (args.out_dir / "pace_gate2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        out_dir=args.out_dir,
        title=args.title,
        experiment=args.experiment,
        args=args,
        cfg=cfg,
        methods=methods,
        results=results,
        method_summary=method_summary,
        summary=summary,
        baseline_runs=baseline_runs,
        baseline_summary=baseline_summary,
    )
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
