#!/usr/bin/env python3
"""Gate 1 offline replay: 4 policies × budget nodes × replay seeds (no new LLM calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_prompt_pool  # noqa: E402
from metrics import bootstrap_mean_ci, hv_regret  # noqa: E402
from pareto import nondominated_ids  # noqa: E402
from policies import get_policy  # noqa: E402
from replay import OracleMatrix, oracle_test_metrics, replay_once  # noqa: E402


def load_oracles(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    prompt_ids = [str(x) for x in data["prompt_ids"].tolist()]
    search = OracleMatrix.from_arrays(
        prompt_ids,
        [str(x) for x in data["search_ids"].tolist()],
        data["search_correct"],
        data["search_input_tokens"],
        data["search_output_tokens"],
    )
    test = OracleMatrix.from_arrays(
        prompt_ids,
        [str(x) for x in data["test_ids"].tolist()],
        data["test_correct"],
        data["test_input_tokens"],
        data["test_output_tokens"],
    )
    return search, test


def test_estimates(oracle: OracleMatrix) -> tuple[dict, dict]:
    acc, cost = {}, {}
    for p in oracle.prompt_ids:
        i = oracle.prompt_index[p]
        acc[p] = float(oracle.correct[i].mean())
        cost[p] = float((oracle.input_tokens[i] + oracle.output_tokens[i]).astype(float).mean())
    return acc, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "gate1.yaml")
    ap.add_argument("--npz", type=Path, default=ROOT / "outputs" / "gate1" / "oracle_matrices.npz")
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompt_pool.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gate1")
    ap.add_argument("--synthetic", action="store_true", help="Use built-in 16x160 synthetic matrix for dry-run")
    ap.add_argument("--smoke", action="store_true", help="Fast dry-run: 3 seeds, n_bootstrap=40")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg["replay_seeds"] = list(cfg["replay_seeds"])[:3]
        cfg["n_bootstrap"] = 40
        cfg["budgets"] = cfg.get("budgets", [320, 640, 1280])
    prompts = load_prompt_pool(args.prompts)

    if args.synthetic or not args.npz.exists():
        print("Using synthetic oracle for dry-run (Gate 0 / pipeline smoke).")
        rng = np.random.default_rng(6707)
        prompt_ids = sorted(prompts.keys())[: cfg["n_prompts"]]
        prompts = {k: prompts[k] for k in prompt_ids}
        sample_ids = [f"syn_s{i:03d}" for i in range(cfg["n_search"])]
        n_p, n_s = len(prompt_ids), len(sample_ids)
        rates = np.linspace(0.75, 0.45, n_p)
        correct = np.vstack([(rng.random(n_s) < r).astype(np.int8) for r in rates])
        tin = np.tile(np.arange(50, 50 + n_p * 4, 4, dtype=np.int32)[:, None], (1, n_s))
        tout = np.full((n_p, n_s), 4, dtype=np.int32)
        search = OracleMatrix.from_arrays(prompt_ids, sample_ids, correct, tin, tout)
        # Test: same rates + noise
        t_ids = [f"syn_t{i:03d}" for i in range(cfg["n_test"])]
        t_correct = np.vstack([(rng.random(len(t_ids)) < r).astype(np.int8) for r in rates])
        t_in = np.tile(np.arange(50, 50 + n_p * 4, 4, dtype=np.int32)[:, None], (1, len(t_ids)))
        t_out = np.full((n_p, len(t_ids)), 4, dtype=np.int32)
        test = OracleMatrix.from_arrays(prompt_ids, t_ids, t_correct, t_in, t_out)
    else:
        search, test = load_oracles(args.npz)

    test_acc, test_cost = test_estimates(test)
    rows = []
    logs_dir = args.out_dir / "replay_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for policy_name in cfg["policies"]:
        for budget in cfg["budgets"]:
            for seed in cfg["replay_seeds"]:
                policy = get_policy(policy_name)
                state, summary = replay_once(
                    policy,
                    search,
                    prompts,
                    budget=int(budget),
                    seed=int(seed),
                    init_common=int(cfg["init_common"]),
                    cost_norm_min=float(cfg["cost_norm_min"]),
                    cost_norm_max=float(cfg["cost_norm_max"]),
                    n_bootstrap=int(cfg["n_bootstrap"]),
                    n_contenders=int(cfg["n_contenders"]),
                    max_pairs=int(cfg["max_pairs"]),
                    pair_batch_samples=int(cfg["pair_batch_samples"]),
                    racing_stop_r=float(cfg["racing_stop_r"]),
                    lambdas=tuple(cfg["lambdas"]),
                )
                regrets = {}
                for lam in cfg["lambdas"]:
                    w = summary["winners"][f"lambda_{lam}"]
                    m = oracle_test_metrics(
                        search,
                        test,
                        w,
                        float(lam),
                        float(cfg["cost_norm_min"]),
                        float(cfg["cost_norm_max"]),
                    )
                    regrets[str(lam)] = m["test_regret"]
                acc_dev, cost_dev = state.snapshot_estimates(use_pop=True)
                nd = nondominated_ids(
                    [p for p in state.prompt_ids if p in acc_dev and not np.isnan(acc_dev[p])],
                    acc_dev,
                    cost_dev,
                )
                hvr = hv_regret(
                    acc_dev,
                    cost_dev,
                    test_acc,
                    test_cost,
                    float(cfg["hv_ref_accuracy"]),
                    float(cfg["hv_ref_cost_norm"]),
                    float(cfg["cost_norm_min"]),
                    float(cfg["cost_norm_max"]),
                    selected_ids=nd or list(state.prompt_ids),
                )
                row = {
                    "policy": policy_name,
                    "budget": budget,
                    "seed": seed,
                    "calls_used": summary["calls_used"],
                    "regret_macro": float(np.mean(list(regrets.values()))),
                    "hv_regret": hvr,
                    **{f"regret_{k}": v for k, v in regrets.items()},
                    **{f"winner_{k}": summary["winners"][f"lambda_{k}"] for k in cfg["lambdas"]},
                }
                rows.append(row)
                (logs_dir / f"{policy_name}_b{budget}_s{seed}.json").write_text(
                    json.dumps({"summary": summary, "queries": state.query_log}, indent=2)
                )

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "replay_results.csv"
    df.to_csv(out_csv, index=False)

    # Aggregate primary budget table
    primary = int(cfg["primary_budget"])
    sub = df[df["budget"] == primary]
    agg_rows = []
    rng = np.random.default_rng(0)
    for policy, g in sub.groupby("policy"):
        vals = g["regret_macro"].tolist()
        ci = bootstrap_mean_ci(vals, rng, n_boot=1000)
        agg_rows.append({"policy": policy, "budget": primary, **ci, "n": len(vals)})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(args.out_dir / "table1_primary_budget.csv", index=False)
    print(agg.to_string(index=False))
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
