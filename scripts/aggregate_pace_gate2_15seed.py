#!/usr/bin/env python3
"""Aggregate PACE vs B0+Archive 15-seed Gate2 cap-align results.

Labels:
  - seeds 42/123/456: exploratory (kept verbatim from prior runs)
  - all 15: primary online extension for paper reporting

Primary metric: paired test accuracy difference (PACE − B0+Archive).
Does not retune protocol; reads finished JSON only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPLORATORY = {42, 123, 456}

ALL_SEEDS = [
    42,
    123,
    456,
    6707,
    2026,
    31415,
    27182,
    16180,
    90210,
    13579,
    86753,
    24680,
    97531,
    11235,
    42424,
]


def _load_run(out_dir: Path, method: str, seed: int) -> dict:
    path = out_dir / f"{method}_s{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _search_selection_regret(run: dict, lam: float, cmin: float, cmax: float) -> float:
    """Auxiliary: λ-utility gap on search estimates (winner vs best in final retained)."""
    from metrics import preference_selection_regret  # noqa: WPS433

    pop = run.get("final_pop_search") or []
    if not pop:
        return float("nan")
    acc = {r["id"]: float(r["search_acc"]) for r in pop}
    cost = {r["id"]: float(r["search_cost"]) for r in pop}
    return float(
        preference_selection_regret(run["winner"], acc, cost, lam, cmin, cmax)
    )


def _n_conf_forced(run: dict) -> tuple[int, int]:
    hist = run.get("history") or []
    return (
        int(sum(h.get("n_conf_elim", 0) for h in hist)),
        int(sum(h.get("n_forced", 0) for h in hist)),
    )


def paired_permutation_pvalue(deltas: np.ndarray, n_perm: int = 100_000, seed: int = 6707) -> float:
    """Two-sided paired sign-flip permutation test on mean(delta)."""
    d = np.asarray(deltas, dtype=float)
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = signs @ d / len(d)
    return float(np.mean(np.abs(null) >= abs(obs)))


def mean_ci_t(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and Student-t approx 95% CI (n-1 df)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return m, float("nan"), float("nan")
    s = float(np.std(x, ddof=1))
    # tcrit approx for df=14 (alpha=0.05 two-sided) ≈ 2.145; use scipy-free approx
    # Normal 1.96 underestimates slightly; use df-aware via crude formula:
    # For df>=10, t ≈ 1.96 + 2.4/(df) works poorly; hardcode common dfs.
    tcrit = {
        2: 12.706,
        3: 4.303,
        14: 2.145,
        15: 2.131,
    }.get(n - 1, 1.96 + 2.0 / max(n - 1, 1))
    half = tcrit * s / math.sqrt(n)
    return m, m - half, m + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pace" / "gate2_cap_align_15seed",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pace_gate2_cap_align_15seed.yaml",
    )
    ap.add_argument("--n-perm", type=int, default=100_000)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    lam = float(cfg["primary_lambda"])
    cmin = float(cfg["cost_norm_min"])
    cmax = float(cfg["cost_norm_max"])
    calls_cap = int(cfg["calls_cap"])
    seeds = list(cfg.get("online_seeds") or ALL_SEEDS)

    import sys

    sys.path.insert(0, str(ROOT / "src"))

    rows = []
    for seed in seeds:
        b0 = _load_run(args.out_dir, "b0_archive", seed)
        pace = _load_run(args.out_dir, "pace", seed)
        b0_conf, b0_forced = _n_conf_forced(b0)
        p_conf, p_forced = _n_conf_forced(pace)
        d_acc = float(pace["test_acc"]) - float(b0["test_acc"])
        d_cost = float(pace["test_cost"]) - float(b0["test_cost"])
        if abs(d_acc) < 1e-12:
            wtl = "Tie"
        elif d_acc > 0:
            wtl = "Win"
        else:
            wtl = "Loss"
        rows.append(
            {
                "seed": seed,
                "cohort": "exploratory" if seed in EXPLORATORY else "extension",
                "b0_test_acc": float(b0["test_acc"]),
                "pace_test_acc": float(pace["test_acc"]),
                "delta_acc": d_acc,
                "b0_test_cost": float(b0["test_cost"]),
                "pace_test_cost": float(pace["test_cost"]),
                "delta_cost": d_cost,
                "b0_calls": int(b0["calls_used"]),
                "pace_calls": int(pace["calls_used"]),
                "b0_budget_util": float(b0["calls_used"]) / calls_cap,
                "pace_budget_util": float(pace["calls_used"]) / calls_cap,
                "b0_search_sel_regret": _search_selection_regret(b0, lam, cmin, cmax),
                "pace_search_sel_regret": _search_selection_regret(pace, lam, cmin, cmax),
                "b0_conf_elim": b0_conf,
                "pace_conf_elim": p_conf,
                "b0_forced": b0_forced,
                "pace_forced": p_forced,
                "wtl": wtl,
                "b0_winner": b0["winner"],
                "pace_winner": pace["winner"],
            }
        )

    deltas = np.array([r["delta_acc"] for r in rows], dtype=float)
    mean_d, lo, hi = mean_ci_t(deltas)
    p_perm = paired_permutation_pvalue(deltas, n_perm=args.n_perm)
    wins = sum(1 for r in rows if r["wtl"] == "Win")
    ties = sum(1 for r in rows if r["wtl"] == "Tie")
    losses = sum(1 for r in rows if r["wtl"] == "Loss")

    # Exploratory-only (3-seed) for explicit retention
    exp_rows = [r for r in rows if r["cohort"] == "exploratory"]
    exp_d = np.array([r["delta_acc"] for r in exp_rows], dtype=float)
    exp_mean = float(np.mean(exp_d)) if len(exp_d) else float("nan")

    summary = {
        "track": "PACE Gate 2 cap-align 15-seed primary",
        "role": "primary_online_extension",
        "contrast": "pace (μ+K, gate) vs b0_archive (μ+K, no gate)",
        "protocol_frozen": True,
        "n_seeds": len(rows),
        "seeds": seeds,
        "exploratory_seeds": sorted(EXPLORATORY),
        "extension_seeds": [s for s in seeds if s not in EXPLORATORY],
        "primary_metric": "paired_test_accuracy_difference (pace - b0_archive)",
        "mean_delta_acc": mean_d,
        "ci95_delta_acc": [lo, hi],
        "std_delta_acc": float(np.std(deltas, ddof=1)),
        "win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
        "paired_permutation_pvalue_two_sided": p_perm,
        "n_permutations": args.n_perm,
        "exploratory_3seed_mean_delta_acc": exp_mean,
        "exploratory_note": (
            "Seeds 42/123/456 are retained exploratory runs; "
            "15-seed analysis is the subsequent extension and paper primary online result."
        ),
        "method_means": {
            "b0_archive": {
                "mean_test_acc": float(np.mean([r["b0_test_acc"] for r in rows])),
                "mean_test_cost": float(np.mean([r["b0_test_cost"] for r in rows])),
                "mean_calls": float(np.mean([r["b0_calls"] for r in rows])),
                "mean_budget_util": float(np.mean([r["b0_budget_util"] for r in rows])),
                "mean_search_sel_regret": float(
                    np.nanmean([r["b0_search_sel_regret"] for r in rows])
                ),
                "mean_conf_elim": float(np.mean([r["b0_conf_elim"] for r in rows])),
                "mean_forced": float(np.mean([r["b0_forced"] for r in rows])),
            },
            "pace": {
                "mean_test_acc": float(np.mean([r["pace_test_acc"] for r in rows])),
                "mean_test_cost": float(np.mean([r["pace_test_cost"] for r in rows])),
                "mean_calls": float(np.mean([r["pace_calls"] for r in rows])),
                "mean_budget_util": float(np.mean([r["pace_budget_util"] for r in rows])),
                "mean_search_sel_regret": float(
                    np.nanmean([r["pace_search_sel_regret"] for r in rows])
                ),
                "mean_conf_elim": float(np.mean([r["pace_conf_elim"] for r in rows])),
                "mean_forced": float(np.mean([r["pace_forced"] for r in rows])),
            },
        },
        "acceptance": (
            "Not required to be statistically significant positive; "
            "deliver credible mean lift, CI, and full cost record. "
            "This 15-seed table is the paper primary online result regardless of sign."
        ),
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pace_gate2_15seed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # CSV
    csv_path = args.out_dir / "pace_gate2_15seed_paired.csv"
    cols = [
        "seed",
        "cohort",
        "b0_test_acc",
        "pace_test_acc",
        "delta_acc",
        "wtl",
        "b0_test_cost",
        "pace_test_cost",
        "delta_cost",
        "b0_calls",
        "pace_calls",
        "b0_budget_util",
        "pace_budget_util",
        "b0_search_sel_regret",
        "pace_search_sel_regret",
        "b0_conf_elim",
        "pace_conf_elim",
        "b0_forced",
        "pace_forced",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    csv_path.write_text("\n".join(lines) + "\n")

    md = [
        "# PACE Gate 2 — Cap-Align 15-Seed Primary Online Result",
        "",
        "**Role:** subsequent extension / paper **primary** online experiment.",
        "**Exploratory retained:** seeds `42, 123, 456` (3-seed exploratory; not discarded).",
        "**Contrast:** PACE (μ=8, K=8, confidence gate on) vs B0+Archive (μ=8, K=8, gate off).",
        "**Protocol:** frozen (AG News × Qwen2.5-7B, pop=8, gen=4, 1280 calls, Racing eval).",
        "",
        "## Primary: paired test accuracy difference (PACE − B0+Archive)",
        "",
        f"- n = {len(rows)} fixed seeds",
        f"- mean Δacc = **{mean_d:+.4f}**",
        f"- 95% CI (Student-t) = **[{lo:+.4f}, {hi:+.4f}]**",
        f"- Win / Tie / Loss = **{wins} / {ties} / {losses}**",
        f"- Paired sign-flip permutation p (two-sided, {args.n_perm:,} perms) = **{p_perm:.4f}**",
        f"- Exploratory 3-seed mean Δacc (retained) = {exp_mean:+.4f}",
        "",
        "## Method means (all 15)",
        "",
        "| Method | test_acc | test_cost (tok) | calls | budget util | search sel. regret | conf_elim | forced |",
        "|--------|----------|-----------------|-------|-------------|--------------------|-----------|--------|",
    ]
    for m in ("b0_archive", "pace"):
        s = summary["method_means"][m]
        md.append(
            f"| {m} | {s['mean_test_acc']:.4f} | {s['mean_test_cost']:.1f} | "
            f"{s['mean_calls']:.0f} | {s['mean_budget_util']:.3f} | "
            f"{s['mean_search_sel_regret']:.4f} | {s['mean_conf_elim']:.1f} | "
            f"{s['mean_forced']:.1f} |"
        )
    md.extend(
        [
            "",
            "## Per-seed paired table",
            "",
            "| seed | cohort | B0 acc | PACE acc | Δacc | W/T/L | B0 cost | PACE cost | B0 calls | PACE calls |",
            "|------|--------|--------|----------|------|-------|---------|-----------|----------|------------|",
        ]
    )
    for r in rows:
        md.append(
            f"| {r['seed']} | {r['cohort']} | {r['b0_test_acc']:.4f} | {r['pace_test_acc']:.4f} | "
            f"{r['delta_acc']:+.4f} | {r['wtl']} | {r['b0_test_cost']:.1f} | {r['pace_test_cost']:.1f} | "
            f"{r['b0_calls']} | {r['pace_calls']} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- Acceptance is a credible mean/CI/cost record, **not** a required significant win.",
            "- `search_sel_regret` is auxiliary on **search** estimates within the final retained set "
            "(sealed-test selection regret over all generated candidates was not purchased).",
            "- Original 3-seed tree `outputs/pace/gate2_cap_align/` is untouched.",
            "",
        ]
    )
    (args.out_dir / "PACE_GATE2_15SEED.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
