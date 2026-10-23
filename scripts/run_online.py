#!/usr/bin/env python3
"""Minimal Gate-2-style evolutionary validation (NOT a DICE rescue).

Pre-registered DICE Gate-1 negative result remains. This script only checks that
NSGA-II + reliable evaluators (Uniform / Racing) can run a small online search
against Random Search on AG News + Qwen2.5-7B.
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
    crossover_prompts,
    generate_offspring,
    mutate_prompt,
    nsga2_select,
)
from inference import QwenPredictor, preload_gate1_matrix_into_cache  # noqa: E402
from metrics import preference_winner  # noqa: E402
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


def allocate_budget(
    policy_name: str,
    state: SearchState,
    oracle: LiveOracle,
    max_new_calls: int,
) -> int:
    """Spend up to max_new_calls with the named policy (no DICE)."""
    if max_new_calls <= 0:
        return 0
    # Temporarily shrink remaining budget window
    hard_cap = state.calls_used + max_new_calls
    old_budget = state.calls_budget
    state.calls_budget = min(state.calls_budget, hard_cap)
    policy = get_policy(policy_name)
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
        tag = TAG_UNIFORM if (policy_name == "uniform" or any(len(v) >= 3 for v in by_s.values())) else TAG_PAIR
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
    """Batched test evaluation."""
    items = []
    index = []  # (prompt_id, local row idx) — gather per prompt
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


def run_nsga2(
    *,
    policy_name: str,
    seed: int,
    cfg: dict,
    init_inds: list[Individual],
    search_rows: list[dict],
    test_rows: list[dict],
    predictor: QwenPredictor,
) -> dict:
    rng = np.random.default_rng(seed)
    pop = [Individual(i.prompt_id, i.text, 0) for i in init_inds]
    texts = {i.prompt_id: i.text for i in pop}
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

    for gen in range(n_gen):
        # Spend a slice of remaining budget on current population (+ children after gen0)
        rem = state.remaining_calls()
        gens_left = n_gen - gen
        slice_budget = rem // gens_left if gens_left else rem
        allocate_budget(policy_name, state, oracle, slice_budget)

        ids = [ind.prompt_id for ind in pop]
        acc, cost = estimates_for(state, ids)
        selected_ids = nsga2_select(ids, acc, cost, pop_size)
        survivors = [ind for ind in pop if ind.prompt_id in selected_ids]
        order = {pid: i for i, pid in enumerate(selected_ids)}
        survivors.sort(key=lambda ind: order[ind.prompt_id])
        history.append({"generation": gen, "ids": [s.prompt_id for s in survivors], "acc": acc, "cost": cost})

        if gen == n_gen - 1:
            pop = survivors
            break

        children = generate_offspring(survivors, rng, pop_size, id_prefix=f"{policy_name}_s{seed}", gen=gen + 1)
        # Register children in oracle/state
        for ch in children:
            oracle.add_prompt(ch.prompt_id, ch.text)
        ensure_prompts_in_state(state, survivors + children)
        # Warm-start children with a few shared random samples if budget allows
        unused = [s for s in state.sample_ids]
        rng.shuffle(unused)
        warm = unused[: min(4, len(unused))]
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

        combined = survivors + children
        rem = state.remaining_calls()
        allocate_budget(policy_name, state, oracle, rem // max(1, (n_gen - gen)))
        ids = [ind.prompt_id for ind in combined]
        acc2, cost2 = estimates_for(state, ids)
        keep = nsga2_select(ids, acc2, cost2, pop_size)
        pop = [ind for ind in combined if ind.prompt_id in keep]
        order = {pid: i for i, pid in enumerate(keep)}
        pop.sort(key=lambda ind: order[ind.prompt_id])

    # Final pick by primary lambda on search estimates
    ids = [ind.prompt_id for ind in pop]
    acc, cost = estimates_for(state, ids)
    lam = float(cfg["primary_lambda"])
    winner_id = preference_winner(ids, acc, cost, lam, float(cfg["cost_norm_min"]), float(cfg["cost_norm_max"]))
    winner = next(ind for ind in pop if ind.prompt_id == winner_id)

    # Protocol: evaluate only the chosen winner on sealed test (saves ~pop×n_test calls).
    test_acc, test_cost = evaluate_on_test(predictor, [winner], test_rows)
    # Search-side regret within final pop (descriptive; not test-oracle regret).
    search_oracle = preference_winner(
        ids, acc, cost, lam, float(cfg["cost_norm_min"]), float(cfg["cost_norm_max"])
    )

    return {
        "method": f"nsga2_{policy_name}",
        "seed": seed,
        "calls_used": state.calls_used,
        "winner": winner_id,
        "winner_text": winner.text,
        "search_acc": acc.get(winner_id),
        "search_cost": cost.get(winner_id),
        "test_acc": test_acc[winner_id],
        "test_cost": test_cost[winner_id],
        "search_oracle_in_final_pop": search_oracle,
        "test_eval_mode": "winner_only",
        "history": history,
        "final_pop_search": [
            {"id": ind.prompt_id, "search_acc": acc[ind.prompt_id], "search_cost": cost[ind.prompt_id]}
            for ind in pop
        ],
    }


def run_random_search(
    *,
    seed: int,
    cfg: dict,
    init_inds: list[Individual],
    search_rows: list[dict],
    test_rows: list[dict],
    predictor: QwenPredictor,
) -> dict:
    """Random mutations without NSGA selection; Uniform evaluation."""
    rng = np.random.default_rng(seed)
    # Start from init texts; keep generating random mutants up to pop trajectory size
    pool = [Individual(i.prompt_id, i.text, 0) for i in init_inds]
    texts = {i.prompt_id: i.text for i in pool}
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

    n_extra = int(cfg["population_size"]) * int(cfg["n_generations"])
    for k in range(n_extra):
        parent = pool[int(rng.integers(0, len(pool)))]
        child = Individual(
            prompt_id=f"rand_s{seed}_c{k:02d}",
            text=mutate_prompt(parent.text, rng),
            generation=k + 1,
            parent_ids=(parent.prompt_id,),
        )
        pool.append(child)
        oracle.add_prompt(child.prompt_id, child.text)
        ensure_prompts_in_state(state, [child])

    allocate_budget("uniform", state, oracle, state.remaining_calls())
    # Select best by search utility among all generated
    ids = [ind.prompt_id for ind in pool]
    acc, cost = estimates_for(state, ids)
    lam = float(cfg["primary_lambda"])
    # Keep a final set of pop_size by utility ranking (not Pareto) for fair-ish report
    ranked = sorted(ids, key=lambda p: (-(lam * acc[p] - (1 - lam) * (cost[p] / float(cfg["cost_norm_max"]))), p))
    final_ids = ranked[: int(cfg["population_size"])]
    final_pop = [ind for ind in pool if ind.prompt_id in final_ids]
    winner_id = preference_winner(
        final_ids,
        {i: acc[i] for i in final_ids},
        {i: cost[i] for i in final_ids},
        lam,
        float(cfg["cost_norm_min"]),
        float(cfg["cost_norm_max"]),
    )
    winner = next(ind for ind in final_pop if ind.prompt_id == winner_id)
    test_acc, test_cost = evaluate_on_test(predictor, [winner], test_rows)
    return {
        "method": "random_search",
        "seed": seed,
        "calls_used": state.calls_used,
        "winner": winner_id,
        "winner_text": winner.text,
        "test_acc": test_acc[winner_id],
        "test_cost": test_cost[winner_id],
        "test_eval_mode": "winner_only",
        "final_pop_search": [
            {"id": ind.prompt_id, "search_acc": acc[ind.prompt_id], "search_cost": cost[ind.prompt_id]}
            for ind in final_pop
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "gate2.yaml")
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompt_pool.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "split_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gate2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--go", action="store_true", help="Acknowledge this is EC validation, not DICE expansion")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 gens, tiny test for wiring")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--gate1-npz",
        type=Path,
        default=ROOT / "outputs" / "gate1" / "oracle_matrices.npz",
        help="Warm cache for initial prompts from Gate-1 matrix",
    )
    args = ap.parse_args()
    if not args.go:
        print("Pass --go to run minimal EC validation (does not claim DICE success).")
        sys.exit(2)

    cfg = yaml.safe_load(args.config.read_text())
    # Override policies: no DICE
    cfg["policies"] = ["uniform", "racing"]
    cfg["n_bootstrap"] = min(int(cfg.get("n_bootstrap", 100)), 20)
    if args.smoke:
        cfg["online_seeds"] = [42]
        cfg["n_generations"] = 2
        cfg["population_size"] = 4
        cfg["calls_cap"] = 256
        cfg["init_common"] = 4

    prompts = load_prompt_pool(args.prompts)
    init_ids = cfg.get("initial_prompt_ids") or sorted(prompts.keys())[: cfg["population_size"]]
    init_inds = [Individual(pid, prompts[pid], 0) for pid in init_ids]
    manifest = load_json(args.manifest)
    search_rows = manifest["search"]
    test_rows = manifest["test"]
    if args.smoke:
        test_rows = test_rows[:40]

    decode = DecodeConfig(max_new_tokens=int(cfg["max_new_tokens"]), do_sample=False)
    cache = EvaluationCache()
    n_warm = preload_gate1_matrix_into_cache(
        cache, cfg["model_id"], decode, args.gate1_npz, prompt_ids=init_ids, split="search"
    )
    print(f"Warmed {n_warm} Gate-1 search cells into cache for init prompts {init_ids}", flush=True)
    predictor = QwenPredictor(
        cfg["model_id"],
        decode,
        cache,
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=args.batch_size,
        device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for seed in cfg["online_seeds"]:
        for pol in cfg["policies"]:
            print(f"=== NSGA-II + {pol} seed={seed} ===", flush=True)
            r = run_nsga2(
                policy_name=pol,
                seed=int(seed),
                cfg=cfg,
                init_inds=init_inds,
                search_rows=search_rows,
                test_rows=test_rows,
                predictor=predictor,
            )
            results.append(r)
            (args.out_dir / f"{r['method']}_s{seed}.json").write_text(json.dumps(r, indent=2) + "\n")
            print(
                json.dumps(
                    {k: r[k] for k in r if k not in ("history", "winner_text", "final_pop_search")},
                    indent=2,
                ),
                flush=True,
            )

        print(f"=== Random Search seed={seed} ===", flush=True)
        r = run_random_search(
            seed=int(seed),
            cfg=cfg,
            init_inds=init_inds,
            search_rows=search_rows,
            test_rows=test_rows,
            predictor=predictor,
        )
        results.append(r)
        (args.out_dir / f"random_search_s{seed}.json").write_text(json.dumps(r, indent=2) + "\n")
        print(
            json.dumps(
                {k: r[k] for k in r if k not in ("winner_text", "final_pop_search")},
                indent=2,
            ),
            flush=True,
        )

    summary = {
        "note": "EC validation only; Gate-1 DICE NO-GO unchanged; batched+winner-only test+gate1 warmup",
        "elapsed_sec": round(time.time() - t0, 1),
        "n_cache": len(cache),
        "gate1_warm_cells": n_warm,
        "batch_size": args.batch_size,
        "runs": [
            {
                "method": r["method"],
                "seed": r["seed"],
                "calls_used": r["calls_used"],
                "test_acc": r["test_acc"],
                "test_cost": r["test_cost"],
                "winner": r["winner"],
            }
            for r in results
        ],
    }
    (args.out_dir / "online_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
