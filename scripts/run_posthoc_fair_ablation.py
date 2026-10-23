#!/usr/bin/env python3
"""Post-hoc diagnostic: fair Racing vs fair DICE on Gate-1 matrix (0 new LLM).

Equalizes early-stop (racing_stop_r) and Uniform column fallback.
Sole algorithmic difference: use_impact True (fair_dice) vs False (fair_racing).

Does NOT replace the pre-registered Gate-1 NO-GO result.
"""

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
from metrics import bootstrap_mean_ci  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "gate1.yaml")
    ap.add_argument("--npz", type=Path, default=ROOT / "outputs" / "gate1" / "oracle_matrices.npz")
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompt_pool.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gate1" / "posthoc_fair")
    ap.add_argument("--primary-budget", type=int, default=640)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompts = load_prompt_pool(args.prompts)
    search, test = load_oracles(args.npz)

    # Fair scaffold policies only (+ original dice/racing for side-by-side)
    policies = ["fair_racing", "fair_dice", "fair_risk_only", "racing", "dice"]
    budgets = [int(args.primary_budget)]  # focus registered primary; optional expand
    # Also report 320/1280 for trend under fair scaffold
    budgets = [320, 640, 1280]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    logs_dir = args.out_dir / "replay_logs"
    logs_dir.mkdir(exist_ok=True)

    for policy_name in policies:
        for budget in budgets:
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
                row = {
                    "policy": policy_name,
                    "budget": budget,
                    "seed": seed,
                    "calls_used": summary["calls_used"],
                    "budget_util": summary["calls_used"] / budget,
                    "regret_macro": float(np.mean(list(regrets.values()))),
                    **{f"regret_{k}": v for k, v in regrets.items()},
                    **{f"winner_{k}": summary["winners"][f"lambda_{k}"] for k in cfg["lambdas"]},
                    "experiment": "posthoc_fair_ablation",
                }
                rows.append(row)
                (logs_dir / f"{policy_name}_b{budget}_s{seed}.json").write_text(
                    json.dumps({"summary": summary, "n_queries": len(state.query_log)}, indent=2)
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "posthoc_fair_results.csv", index=False)

    primary = int(args.primary_budget)
    sub = df[df["budget"] == primary]
    rng = np.random.default_rng(0)
    agg = []
    for policy, g in sub.groupby("policy"):
        ci = bootstrap_mean_ci(g["regret_macro"].tolist(), rng, n_boot=1000)
        agg.append(
            {
                "policy": policy,
                "budget": primary,
                "regret_mean": ci["mean"],
                "regret_sd": ci["sd"],
                "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"],
                "util_mean": float(g["budget_util"].mean()),
                "util_sd": float(g["budget_util"].std(ddof=1)),
                "n": len(g),
            }
        )
    agdf = pd.DataFrame(agg).sort_values("regret_mean")
    agdf.to_csv(args.out_dir / "posthoc_fair_primary.csv", index=False)

    # Paired seed deltas at primary
    piv = sub.pivot(index="seed", columns="policy", values="regret_macro")
    delta = None
    if "fair_dice" in piv.columns and "fair_racing" in piv.columns:
        d = piv["fair_dice"] - piv["fair_racing"]
        delta = {
            "mean_fair_dice_minus_fair_racing": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "frac_dice_better": float((d < 0).mean()),
            "frac_tie": float((d == 0).mean()),
            "frac_racing_better": float((d > 0).mean()),
        }

    util_piv = sub.pivot(index="seed", columns="policy", values="budget_util")

    md = [
        "# Post-hoc Fair Ablation (0 new LLM calls)",
        "",
        "**Status**: diagnostic / exploratory — does **not** replace pre-registered Gate-1 NO-GO.",
        "",
        "## Design",
        "",
        "- Shared: `racing_stop_r`, Uniform column fallback when no pair query, same matrix/seeds/budgets.",
        "- Sole difference: `use_impact` True (`fair_dice`) vs False (`fair_racing`).",
        "- `fair_risk_only` should match `fair_racing` (sanity).",
        "- Original `dice` / `racing` included for side-by-side.",
        "",
        f"## Primary budget {primary} — macro test preference regret",
        "",
        agdf.round(5).to_markdown(index=False),
        "",
        "## Budget utilization (primary)",
        "",
        sub.groupby("policy")["budget_util"].agg(["mean", "std"]).round(4).to_markdown(),
        "",
    ]
    if delta:
        md += [
            "## Paired seed: fair_dice − fair_racing (regret; lower better for method)",
            "",
            f"- mean Δ = **{delta['mean_fair_dice_minus_fair_racing']:.5f}** (positive ⇒ fair_dice worse)",
            f"- P(fair_dice better) = {delta['frac_dice_better']:.2f}, "
            f"P(tie)={delta['frac_tie']:.2f}, P(racing better)={delta['frac_racing_better']:.2f}",
            "",
        ]
    # Trend table fair only
    fair = df[df["policy"].isin(["fair_racing", "fair_dice"])]
    trend = fair.groupby(["budget", "policy"])["regret_macro"].agg(["mean", "std"]).round(5)
    md += ["## Trend (fair policies only)", "", trend.to_markdown(), ""]

    # Verdict
    fr = agdf.set_index("policy")
    if "fair_dice" in fr.index and "fair_racing" in fr.index:
        rd = float(fr.loc["fair_dice", "regret_mean"] - fr.loc["fair_racing", "regret_mean"])
        ud = float(fr.loc["fair_dice", "util_mean"] - fr.loc["fair_racing", "util_mean"])
        if rd < -1e-6:
            verdict = (
                "Under equal early-stop/fallback, fair_dice has **lower** mean regret than fair_racing. "
                "Suggests prior NO-GO may have been partly driven by scaffold mismatch; still mark exploratory."
            )
        elif abs(rd) <= 1e-6:
            verdict = "Under equal scaffold, fair_dice ≈ fair_racing (no impact gain)."
        else:
            verdict = (
                "Under equal early-stop/fallback, fair_dice still has **higher** mean regret than fair_racing. "
                "Supports that Decision-Critical Scoring itself lacks benefit in this setting "
                "(not only the missing stop/fallback)."
            )
        md += [
            "## Verdict",
            "",
            f"- util gap (fair_dice − fair_racing) = {ud:.4f}",
            f"- regret gap (fair_dice − fair_racing) = {rd:.5f}",
            f"- {verdict}",
            "",
        ]

    (args.out_dir / "POSTHOC_FAIR.md").write_text("\n".join(md), encoding="utf-8")
    print(agdf.round(5).to_string(index=False))
    if delta:
        print("delta", delta)
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
