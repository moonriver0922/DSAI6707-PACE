#!/usr/bin/env python3
"""Zero-GPU archive / genealogy / false-retention audit on frozen Gate2 logs.

Answers (from existing 15-seed cap-align JSONs only):
  1) Do gate-protected / archive-only candidates later breed or win?
  2) How much μ+K capacity is occupied by non-breeding passengers?
  3) Soft-selection counterfactual is limited to *selection on observed pools*
     (no fabricated unobserved offspring) — reported separately as non-causal.

Does not modify frozen runs. Writes under outputs/pace/gate2_cap_align_15seed/analysis/.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [
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


def _hist_steps(run: dict) -> list[dict]:
    return [h for h in run.get("history") or [] if h.get("phase") not in (None, "leftover_racing")]


def _parse_gen(pid: str) -> int:
    # init prompts p00.. → gen 0; *_gK_c* → K
    if pid.startswith("p") and "_g" not in pid:
        return 0
    if "_g" in pid:
        try:
            return int(pid.split("_g")[1].split("_")[0])
        except ValueError:
            return -1
    return -1


def candidate_trajectories(run: dict) -> dict[str, dict]:
    """Per-candidate survival / breeding exposure from logged survive steps."""
    traj: dict[str, dict] = {}
    steps = _hist_steps(run)

    def ensure(pid: str) -> dict:
        if pid not in traj:
            traj[pid] = {
                "pid": pid,
                "born_gen": _parse_gen(pid),
                "steps_seen": 0,
                "steps_retained": 0,
                "steps_active": 0,
                "steps_archive_only": 0,  # retained \ active
                "conf_elim": 0,
                "forced_elim": 0,
                "first_retained_step": None,
                "last_retained_step": None,
                "ever_active": False,
                "bred_opportunities": 0,  # times in active at a pre-offspring racing_then_survive
            }
        return traj[pid]

    for si, h in enumerate(steps):
        retained = set(h.get("retained") or [])
        active = set(h.get("active") or [])
        cands = set(h.get("candidate_ids") or []) | retained | active
        for pid in cands:
            ensure(pid)
        for e in h.get("confidence_elims") or []:
            ensure(e["eliminated"])["conf_elim"] += 1
        for e in h.get("forced_evictions") or []:
            ensure(e["prompt_id"])["forced_elim"] += 1

        for pid in cands:
            t = traj[pid]
            t["steps_seen"] += 1
            if pid in retained:
                t["steps_retained"] += 1
                if t["first_retained_step"] is None:
                    t["first_retained_step"] = si
                t["last_retained_step"] = si
                if pid in active:
                    t["steps_active"] += 1
                    t["ever_active"] = True
                else:
                    t["steps_archive_only"] += 1
            # Breeding parents = active after racing_then_survive, when offspring follow
            if (
                h.get("phase") == "racing_then_survive"
                and pid in active
                and si + 1 < len(steps)
                and steps[si + 1].get("phase") == "offspring_racing_then_survive"
            ):
                t["bred_opportunities"] += 1

    return traj


def final_quality(run: dict) -> dict[str, dict]:
    return {r["id"]: r for r in (run.get("final_pop_search") or [])}


def analyze_run(method: str, seed: int, run: dict) -> dict:
    traj = candidate_trajectories(run)
    qual = final_quality(run)
    winner = run["winner"]
    final_ret = set(run.get("final_retained") or [])
    final_act = set(run.get("final_active") or [])
    archive_end = final_ret - final_act

    # Passengers: ever archive-only; never-bred: bred_opportunities==0
    ever_archive_only = [t for t in traj.values() if t["steps_archive_only"] > 0]
    never_bred = [t for t in traj.values() if t["steps_retained"] > 0 and t["bred_opportunities"] == 0]
    archive_then_active = [
        t for t in ever_archive_only if t["ever_active"] and t["bred_opportunities"] > 0
    ]
    pure_passengers = [
        t for t in ever_archive_only if t["bred_opportunities"] == 0
    ]

    # Archive slot-steps (capacity pressure proxy)
    archive_slot_steps = sum(t["steps_archive_only"] for t in traj.values())
    retained_slot_steps = sum(t["steps_retained"] for t in traj.values())
    passenger_slot_steps = sum(t["steps_archive_only"] for t in pure_passengers)

    # Final retained quality (search estimates)
    final_rows = []
    for pid in final_ret:
        q = qual.get(pid, {})
        t = traj.get(pid, {})
        final_rows.append(
            {
                "pid": pid,
                "search_acc": q.get("search_acc"),
                "search_cost": q.get("search_cost"),
                "in_active": pid in final_act,
                "archive_only_end": pid in archive_end,
                "bred_opportunities": t.get("bred_opportunities", 0),
                "steps_archive_only": t.get("steps_archive_only", 0),
                "is_winner": pid == winner,
            }
        )

    low_final = [r for r in final_rows if r["search_acc"] is not None and r["search_acc"] < 0.5]
    low_archive = [r for r in low_final if r["archive_only_end"]]

    # Winner path
    wt = traj.get(winner, {})
    winner_info = {
        "winner": winner,
        "test_acc": run["test_acc"],
        "test_cost": run["test_cost"],
        "born_gen": _parse_gen(winner),
        "bred_opportunities": wt.get("bred_opportunities", 0),
        "steps_archive_only": wt.get("steps_archive_only", 0),
        "ever_active": wt.get("ever_active", False),
        "in_final_active": winner in final_act,
        "in_final_archive_only": winner in archive_end,
    }

    # Conf vs forced totals
    steps = _hist_steps(run)
    n_conf = sum(h.get("n_conf_elim", 0) for h in steps)
    n_forced = sum(h.get("n_forced", 0) for h in steps)

    # Soft-selection counterfactual on FINAL retained pool only (observational):
    # Compare hard top-μ by search preference utility vs soft weights ∝ bootstrap-free
    # rank score. This does NOT claim online lift.
    soft = _soft_vs_hard_final(final_rows, mu=8)

    return {
        "method": method,
        "seed": seed,
        "n_candidates_ever": len(traj),
        "n_conf_elim": n_conf,
        "n_forced": n_forced,
        "forced_share_of_elims": (
            n_forced / (n_conf + n_forced) if (n_conf + n_forced) else float("nan")
        ),
        "n_ever_archive_only": len(ever_archive_only),
        "n_pure_passengers": len(pure_passengers),
        "n_archive_then_bred": len(archive_then_active),
        "archive_then_bred_rate": (
            len(archive_then_active) / len(ever_archive_only) if ever_archive_only else float("nan")
        ),
        "archive_slot_steps": archive_slot_steps,
        "retained_slot_steps": retained_slot_steps,
        "passenger_slot_steps": passenger_slot_steps,
        "passenger_archive_share": (
            passenger_slot_steps / archive_slot_steps if archive_slot_steps else float("nan")
        ),
        "n_final_retained": len(final_ret),
        "n_final_archive": len(archive_end),
        "n_final_low_acc": len(low_final),
        "n_final_low_in_archive": len(low_archive),
        "mean_final_search_acc": float(
            np.mean([r["search_acc"] for r in final_rows if r["search_acc"] is not None])
        )
        if final_rows
        else float("nan"),
        "mean_archive_end_search_acc": float(
            np.mean(
                [
                    r["search_acc"]
                    for r in final_rows
                    if r["archive_only_end"] and r["search_acc"] is not None
                ]
            )
        )
        if any(r["archive_only_end"] for r in final_rows)
        else float("nan"),
        "mean_active_end_search_acc": float(
            np.mean(
                [
                    r["search_acc"]
                    for r in final_rows
                    if r["in_active"] and r["search_acc"] is not None
                ]
            )
        )
        if any(r["in_active"] for r in final_rows)
        else float("nan"),
        "winner": winner_info,
        "soft_final_counterfactual": soft,
        "pure_passenger_ids": [t["pid"] for t in pure_passengers][:20],
        "low_final_ids": [r["pid"] for r in low_final],
    }


def _soft_vs_hard_final(final_rows: list[dict], mu: int) -> dict:
    """Non-causal: on final observed pool, hard keep-μ vs soft weight mass on low-acc."""
    rows = [r for r in final_rows if r["search_acc"] is not None]
    if len(rows) < 2:
        return {"note": "insufficient_final_rows"}
    # Rank by search_acc then -cost
    ranked = sorted(
        rows,
        key=lambda r: (-float(r["search_acc"]), float(r["search_cost"] or 1e9), r["pid"]),
    )
    hard = {r["pid"] for r in ranked[:mu]}
    # Soft weights via softmax on search_acc
    accs = np.array([float(r["search_acc"]) for r in ranked], dtype=float)
    # temperature: scale by std
    s = float(np.std(accs)) + 1e-6
    z = (accs - accs.max()) / s
    w = np.exp(z)
    w = w / w.sum()
    soft_mass_outside_hard = float(
        sum(wi for r, wi in zip(ranked, w) if r["pid"] not in hard)
    )
    soft_mass_on_low = float(
        sum(wi for r, wi in zip(ranked, w) if float(r["search_acc"]) < 0.5)
    )
    hard_drops_low = sum(1 for r in ranked[mu:] if float(r["search_acc"]) < 0.5)
    return {
        "n_pool": len(ranked),
        "soft_mass_outside_top_mu": soft_mass_outside_hard,
        "soft_mass_on_search_acc_lt_0.5": soft_mass_on_low,
        "hard_truncate_would_drop_low_acc": hard_drops_low,
        "note": (
            "Observational on final retained estimates only; "
            "does not simulate ungenerated offspring or online trajectories."
        ),
    }


def paired_pace_vs_b0(pace_a: dict, b0_a: dict) -> dict:
    return {
        "seed": pace_a["seed"],
        "delta_test_acc": pace_a["winner"]["test_acc"] - b0_a["winner"]["test_acc"],
        "pace_forced_share": pace_a["forced_share_of_elims"],
        "b0_forced_share": b0_a["forced_share_of_elims"],
        "pace_passenger_archive_share": pace_a["passenger_archive_share"],
        "b0_passenger_archive_share": b0_a["passenger_archive_share"],
        "pace_archive_then_bred_rate": pace_a["archive_then_bred_rate"],
        "pace_n_final_low_in_archive": pace_a["n_final_low_in_archive"],
        "b0_n_final_low_in_archive": b0_a["n_final_low_in_archive"],
        "pace_winner_ever_archive_only": pace_a["winner"]["steps_archive_only"] > 0,
        "pace_winner_bred_ops": pace_a["winner"]["bred_opportunities"],
        "pace_mean_archive_acc": pace_a["mean_archive_end_search_acc"],
        "pace_mean_active_acc": pace_a["mean_active_end_search_acc"],
    }


def go_nogo(summary: dict) -> dict:
    """Precommitted qualitative rule from advisor brief (mechanism bottleneck)."""
    pace = summary["pace_means"]
    # Bottleneck signals
    passenger_heavy = pace["mean_passenger_archive_share"] >= 0.40
    archive_weak = (
        not math.isnan(pace["mean_archive_end_search_acc"])
        and not math.isnan(pace["mean_active_end_search_acc"])
        and pace["mean_archive_end_search_acc"] + 0.10 < pace["mean_active_end_search_acc"]
    )
    rarely_promoted = pace["mean_archive_then_bred_rate"] < 0.35
    forced_dominates = pace["mean_forced_share"] >= 0.70
    winner_needs_archive = pace["frac_winner_ever_archive_only"] >= 0.40

    bottleneck = passenger_heavy and archive_weak and rarely_promoted
    # If winners often passed through archive-only, protection may be valuable
    protection_valuable = winner_needs_archive and pace["mean_archive_then_bred_rate"] >= 0.35

    if bottleneck and not protection_valuable:
        decision = "GO_SPaCE_MINIMAL"
        rationale = (
            "Archive passengers occupy substantial capacity, end weaker than active, "
            "and seldom convert into breeding opportunities — false-retention / opportunity-cost "
            "bottleneck supported → justify minimal soft-selection extension."
        )
    elif protection_valuable and not bottleneck:
        decision = "NO_GO_KEEP_PACE"
        rationale = (
            "Protected/archive path appears to contribute to winners or later breeding; "
            "primary issue may be variance / mutation / sample depth, not hard survival rule."
        )
    else:
        decision = "CONDITIONAL"
        rationale = (
            "Mixed signals: document false-retention pressure and forced-eviction dominance, "
            "but do not claim soft-selection will fix online Δacc without a new frozen protocol. "
            "Default course delivery = freeze PACE v1; SPaCE only as labeled extension if pursued."
        )

    return {
        "decision": decision,
        "signals": {
            "passenger_heavy": passenger_heavy,
            "archive_weak_vs_active": archive_weak,
            "rarely_promoted_to_breeding": rarely_promoted,
            "forced_dominates_elims": forced_dominates,
            "winner_often_touched_archive": winner_needs_archive,
            "protection_valuable": protection_valuable,
            "bottleneck": bottleneck,
        },
        "rationale": rationale,
        "delivery_stance": (
            "PACE v1 (15-seed cap-align) remains the course primary deliverable regardless. "
            "SPaCE is a conditional research extension, not a replacement of frozen results."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=ROOT / "outputs" / "pace" / "gate2_cap_align_15seed",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    out_dir = args.out_dir or (args.in_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    pace_runs = []
    b0_runs = []
    paired = []
    for seed in SEEDS:
        pace = json.loads((args.in_dir / f"pace_s{seed}.json").read_text())
        b0 = json.loads((args.in_dir / f"b0_archive_s{seed}.json").read_text())
        pa = analyze_run("pace", seed, pace)
        ba = analyze_run("b0_archive", seed, b0)
        pace_runs.append(pa)
        b0_runs.append(ba)
        paired.append(paired_pace_vs_b0(pa, ba))

    def mean_key(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None and not (
            isinstance(r[key], float) and math.isnan(r[key])
        )]
        return float(np.mean(vals)) if vals else float("nan")

    pace_means = {
        "mean_forced_share": mean_key(pace_runs, "forced_share_of_elims"),
        "mean_passenger_archive_share": mean_key(pace_runs, "passenger_archive_share"),
        "mean_archive_then_bred_rate": mean_key(pace_runs, "archive_then_bred_rate"),
        "mean_n_pure_passengers": mean_key(pace_runs, "n_pure_passengers"),
        "mean_n_final_low_in_archive": mean_key(pace_runs, "n_final_low_in_archive"),
        "mean_archive_end_search_acc": mean_key(pace_runs, "mean_archive_end_search_acc"),
        "mean_active_end_search_acc": mean_key(pace_runs, "mean_active_end_search_acc"),
        "mean_conf_elim": mean_key(pace_runs, "n_conf_elim"),
        "mean_forced": mean_key(pace_runs, "n_forced"),
        "frac_winner_ever_archive_only": float(
            np.mean([1.0 if r["winner"]["steps_archive_only"] > 0 else 0.0 for r in pace_runs])
        ),
        "frac_winner_in_final_archive_only": float(
            np.mean([1.0 if r["winner"]["in_final_archive_only"] else 0.0 for r in pace_runs])
        ),
        "mean_winner_bred_ops": float(
            np.mean([r["winner"]["bred_opportunities"] for r in pace_runs])
        ),
    }
    soft_masses = [
        r["soft_final_counterfactual"].get("soft_mass_outside_top_mu", float("nan"))
        for r in pace_runs
    ]
    pace_means["mean_soft_mass_outside_top_mu"] = float(
        np.nanmean(np.array(soft_masses, dtype=float))
    )

    b0_means = {
        "mean_forced_share": mean_key(b0_runs, "forced_share_of_elims"),
        "mean_passenger_archive_share": mean_key(b0_runs, "passenger_archive_share"),
        "mean_archive_then_bred_rate": mean_key(b0_runs, "archive_then_bred_rate"),
        "mean_n_final_low_in_archive": mean_key(b0_runs, "n_final_low_in_archive"),
        "mean_archive_end_search_acc": mean_key(b0_runs, "mean_archive_end_search_acc"),
        "mean_active_end_search_acc": mean_key(b0_runs, "mean_active_end_search_acc"),
        "mean_conf_elim": mean_key(b0_runs, "n_conf_elim"),
        "mean_forced": mean_key(b0_runs, "n_forced"),
    }

    # Correlation: passenger pressure vs Δacc
    d_acc = np.array([p["delta_test_acc"] for p in paired], dtype=float)
    pass_share = np.array([a["passenger_archive_share"] for a in pace_runs], dtype=float)
    if len(d_acc) >= 3 and np.std(pass_share) > 1e-12:
        corr = float(np.corrcoef(pass_share, d_acc)[0, 1])
    else:
        corr = float("nan")

    summary = {
        "n_seeds": len(SEEDS),
        "source": str(args.in_dir),
        "limitations": [
            "parent_ids not logged; breeding inferred from active set at offspring steps only.",
            "No mid-run per-candidate accuracy in history; soft counterfactual is final-pool only.",
            "PACE vs B0 trajectories diverge; cannot attribute the same child to alternate parents.",
            "Zero new LLM calls; not a causal proof of SPaCE online gains.",
        ],
        "pace_means": pace_means,
        "b0_means": b0_means,
        "corr_passenger_share_vs_delta_acc": corr,
        "paired": paired,
        "pace_runs": pace_runs,
        "b0_runs": b0_runs,
    }
    decision = go_nogo(summary)
    summary["go_nogo"] = decision

    (out_dir / "false_retention_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    md = [
        "# PACE False-Retention / Archive Opportunity-Cost Audit (0 GPU)",
        "",
        f"Source: `{args.in_dir}` · n={len(SEEDS)} seeds · PACE vs B0+Archive logs only.",
        "",
        "## Limitations (read first)",
        "",
    ]
    for lim in summary["limitations"]:
        md.append(f"- {lim}")
    md.extend(
        [
            "",
            "## Headline means (PACE)",
            "",
            f"| Signal | PACE | B0+Archive |",
            f"|--------|------|------------|",
            f"| Forced share of eliminations | {pace_means['mean_forced_share']:.3f} | {b0_means['mean_forced_share']:.3f} |",
            f"| Mean conf_elim / run | {pace_means['mean_conf_elim']:.2f} | {b0_means['mean_conf_elim']:.2f} |",
            f"| Mean forced / run | {pace_means['mean_forced']:.2f} | {b0_means['mean_forced']:.2f} |",
            f"| Pure passenger share of archive-slot-steps | {pace_means['mean_passenger_archive_share']:.3f} | {b0_means['mean_passenger_archive_share']:.3f} |",
            f"| Archive-only → later breeding rate | {pace_means['mean_archive_then_bred_rate']:.3f} | {b0_means['mean_archive_then_bred_rate']:.3f} |",
            f"| Mean end archive search_acc | {pace_means['mean_archive_end_search_acc']:.3f} | {b0_means['mean_archive_end_search_acc']:.3f} |",
            f"| Mean end active search_acc | {pace_means['mean_active_end_search_acc']:.3f} | {b0_means['mean_active_end_search_acc']:.3f} |",
            f"| Final low-acc (<0.5) sitting in archive | {pace_means['mean_n_final_low_in_archive']:.2f} | {b0_means['mean_n_final_low_in_archive']:.2f} |",
            f"| Frac winners ever archive-only | {pace_means['frac_winner_ever_archive_only']:.3f} | — |",
            f"| Frac winners final archive-only (never needed active to win) | {pace_means['frac_winner_in_final_archive_only']:.3f} | — |",
            f"| Corr(passenger_share, Δacc) | {corr:+.3f} | — |",
            "",
            "## Interpretation checklist",
            "",
            "1. **Protected → better offspring?** Exact parent_ids absent; proxy = archive-only candidates "
            "that later appear in `active` before an offspring step (`archive_then_bred_rate`).",
            "2. **False retention occupying archive?** `passenger_archive_share` + final low-acc in archive.",
            "3. **Soft vs hard?** Final-pool soft mass outside top-μ is observational only "
            f"(mean soft mass outside top-μ ≈ {pace_means['mean_soft_mass_outside_top_mu']:.3f}).",
            "",
            "## Go / No-Go (SPaCE extension)",
            "",
            f"**Decision: `{decision['decision']}`**",
            "",
            decision["rationale"],
            "",
            "Signals:",
            "",
        ]
    )
    for k, v in decision["signals"].items():
        md.append(f"- `{k}`: **{v}**")
    md.extend(
        [
            "",
            decision["delivery_stance"],
            "",
            "## Per-seed paired snapshot",
            "",
            "| seed | Δacc | PACE pass.share | arch→bred | PACE low@arch | winner touched archive |",
            "|------|------|-----------------|-----------|---------------|------------------------|",
        ]
    )
    for p, a in zip(paired, pace_runs):
        md.append(
            f"| {p['seed']} | {p['delta_test_acc']:+.3f} | {a['passenger_archive_share']:.2f} | "
            f"{a['archive_then_bred_rate']:.2f} | {a['n_final_low_in_archive']} | "
            f"{a['winner']['steps_archive_only'] > 0} |"
        )
    md.extend(
        [
            "",
            "## Insight (for report discussion, not a new claim of SOTA)",
            "",
            "> Under a fixed evolutionary search budget, minimizing false elimination alone "
            "does not maximize the long-term value of the surviving population.",
            "",
            "DICE asked whether each *evaluation* was worth buying; PACE asked whether each "
            "*elimination* was justified; the open question is whether each *survivor* deserves "
            "continued breeding / archive mass.",
            "",
        ]
    )
    (out_dir / "FALSE_RETENTION_GO_NOGO.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
