#!/usr/bin/env python3
"""PACE Gate 1a/1b on frozen Gate-1 matrix (0 new LLM calls).

Matched design: one Paired-Racing purchase trajectory per (budget, seed);
then B0 point-estimate truncate vs PACE evidence-gated survival on the *same* state.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_prompt_pool  # noqa: E402
from pace.audit import (  # noqa: E402
    false_elimination_rate,
    full_matrix_estimates,
    reference_pareto_ids,
    reference_pareto_recall,
    retained_set_regret,
)
from pace.survivor import pace_bounded_survive, point_estimate_truncate  # noqa: E402
from policies import get_policy  # noqa: E402
from replay import OracleMatrix, initialize_common_anchor, run_policy_until_budget  # noqa: E402
from state import empty_state  # noqa: E402


def load_search_oracle(npz_path: Path) -> OracleMatrix:
    data = np.load(npz_path, allow_pickle=True)
    prompt_ids = [str(x) for x in data["prompt_ids"].tolist()]
    return OracleMatrix.from_arrays(
        prompt_ids,
        [str(x) for x in data["search_ids"].tolist()],
        data["search_correct"],
        data["search_input_tokens"],
        data["search_output_tokens"],
    )


def build_state(oracle: OracleMatrix, prompts: dict, cfg: dict, budget: int, seed: int):
    state = empty_state(
        oracle.prompt_ids,
        oracle.sample_ids,
        {p: prompts[p] for p in oracle.prompt_ids},
        calls_budget=budget,
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
    rng = np.random.default_rng(seed)
    initialize_common_anchor(state, oracle, int(cfg["init_common"]), rng)
    run_policy_until_budget(get_policy("racing"), state, oracle)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pace_gate1.yaml")
    ap.add_argument("--npz", type=Path, default=ROOT / "outputs" / "gate1" / "oracle_matrices.npz")
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompt_pool.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "pace" / "gate1")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompts = load_prompt_pool(args.prompts)
    oracle = load_search_oracle(args.npz)
    ref_pareto = set(reference_pareto_ids(oracle))
    acc_full, cost_full = full_matrix_estimates(oracle)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reference_pareto.json").write_text(
        json.dumps(
            {
                "reference_pareto": sorted(ref_pareto),
                "full_acc": acc_full,
                "full_cost": cost_full,
                "note": "Audit-only; never used by policies",
            },
            indent=2,
        )
        + "\n"
    )

    rows = []
    traces = []
    for budget in cfg["budgets"]:
        for seed in cfg["replay_seeds"]:
            state = build_state(oracle, prompts, cfg, int(budget), int(seed))
            pool = list(state.prompt_ids)

            # B0
            b0 = point_estimate_truncate(state, pool, mu=int(cfg["mu"]))
            b0_ret = set(b0.retained)
            b0_elim = set(pool) - b0_ret

            # PACE (same state)
            rng = np.random.default_rng(10_000 + int(seed) + int(budget))
            pace = pace_bounded_survive(
                state,
                pool,
                mu=int(cfg["mu"]),
                archive_k=int(cfg["archive_k"]),
                epsilon=float(cfg["epsilon"]),
                n_min_common=int(cfg["n_min_common"]),
                n_bootstrap=int(cfg["n_bootstrap"]),
                rng=rng,
            )
            pace_ret = set(pace.retained)
            pace_conf = {e.eliminated for e in pace.confidence_elims}
            pace_forced = {e.prompt_id for e in pace.forced_evictions}
            pace_elim = pace_conf | pace_forced

            for method, retained, elim, extra in [
                ("b0_point_nsga", b0_ret, b0_elim, {"n_conf_elim": 0, "n_forced": 0}),
                (
                    "pace",
                    pace_ret,
                    pace_elim,
                    {
                        "n_conf_elim": len(pace.confidence_elims),
                        "n_forced": len(pace.forced_evictions),
                    },
                ),
            ]:
                row = {
                    "method": method,
                    "budget": budget,
                    "seed": seed,
                    "calls_used": state.calls_used,
                    "n_retained": len(retained),
                    "false_elim_rate": false_elimination_rate(elim, ref_pareto),
                    "ref_pareto_recall": reference_pareto_recall(retained, ref_pareto),
                    "retained_regret": retained_set_regret(
                        sorted(retained),
                        oracle,
                        float(cfg["primary_lambda"]),
                        float(cfg["cost_norm_min"]),
                        float(cfg["cost_norm_max"]),
                    ),
                    "retained": ",".join(sorted(retained)),
                    "eliminated": ",".join(sorted(elim)),
                    **extra,
                }
                rows.append(row)

            traces.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "calls_used": state.calls_used,
                    "b0_retained": sorted(b0_ret),
                    "pace_retained": sorted(pace_ret),
                    "pace_confidence_elims": [
                        {
                            "eliminated": e.eliminated,
                            "by": e.by,
                            "L": e.L,
                            "U": e.U,
                            "delta": e.delta_hat,
                            "n": e.n_common,
                        }
                        for e in pace.confidence_elims
                    ],
                    "pace_forced": sorted(pace_forced),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "pace_gate1_results.csv", index=False)
    (args.out_dir / "pace_gate1_traces.jsonl").write_text(
        "\n".join(json.dumps(t) for t in traces) + "\n"
    )

    primary = int(cfg["primary_budget"])
    sub = df[df["budget"] == primary]
    summary = (
        sub.groupby("method")[["false_elim_rate", "ref_pareto_recall", "retained_regret", "n_retained", "n_conf_elim", "n_forced"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(args.out_dir / "pace_gate1_primary_summary.csv")

    # Premise H0: nontrivial false elim under B0?
    b0 = sub[sub["method"] == "b0_point_nsga"]
    pace_s = sub[sub["method"] == "pace"]
    h0_rate = float(b0["false_elim_rate"].mean())
    d_fer = float(pace_s["false_elim_rate"].mean() - b0["false_elim_rate"].mean())
    d_rec = float(pace_s["ref_pareto_recall"].mean() - b0["ref_pareto_recall"].mean())
    d_reg = float(pace_s["retained_regret"].mean() - b0["retained_regret"].mean())

    # GO if PACE reduces false elim and does not worsen regret materially
    go = (h0_rate >= 0.05) and (d_fer < 0) and (d_reg <= 0.002)

    decision = {
        "primary_budget": primary,
        "reference_pareto": sorted(ref_pareto),
        "H0_b0_mean_false_elim_rate": h0_rate,
        "H0_premise_nontrivial": h0_rate >= 0.05,
        "delta_false_elim_pace_minus_b0": d_fer,
        "delta_recall_pace_minus_b0": d_rec,
        "delta_regret_pace_minus_b0": d_reg,
        "GO": go,
        "rule": "GO iff H0_rate>=0.05 and PACE false_elim < B0 and retained_regret not worse by >0.002",
        "note": "Post-matched Racing trajectory; exploratory GO for Gate 2 only. Negative findings retained.",
    }
    (args.out_dir / "PACE_GO_NOGO.json").write_text(json.dumps(decision, indent=2) + "\n")

    md = [
        "# PACE Gate 1 Results (0 new LLM)",
        "",
        f"Reference Pareto (audit-only full search matrix): `{sorted(ref_pareto)}`",
        "",
        f"## Primary budget {primary} — mean±std over 20 replay seeds",
        "",
        summary.to_markdown(),
        "",
        "## GO/NO-GO",
        "",
        f"- B0 mean false-elim rate (H0): **{h0_rate:.4f}** (nontrivial≥0.05? {h0_rate >= 0.05})",
        f"- Δ false-elim (PACE−B0): **{d_fer:.4f}** (want <0)",
        f"- Δ recall (PACE−B0): **{d_rec:.4f}**",
        f"- Δ retained regret (PACE−B0): **{d_reg:.4f}** (want ≤0.002)",
        f"- **Decision: {'GO' if go else 'NO-GO'}**",
        "",
        "DICE historical negatives remain unchanged and are not PACE wins.",
        "",
    ]
    (args.out_dir / "PACE_GATE1.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
