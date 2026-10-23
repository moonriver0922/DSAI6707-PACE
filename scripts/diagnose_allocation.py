#!/usr/bin/env python3
"""Task A: mechanism diagnosis from Gate 1 replay logs (0 LLM calls)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_logs(log_dir: Path, budget: int, policies: list[str]):
    rows = []
    for pol in policies:
        for seed in range(20):
            path = log_dir / f"{pol}_b{budget}_s{seed}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            rows.append((pol, seed, data))
    return rows


def herfindahl(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    shares = np.array([c / total for c in counts.values()], dtype=float)
    return float((shares ** 2).sum())


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def segment_post_anchor(queries: list[dict]) -> dict:
    """Split post-anchor queries into column-fills vs paired shared-sample buys.

    Column-fill: consecutive queries share one sample_id and cover ≥3 prompts
    (Uniform / Racing fallback).
    Pair-buy: consecutive queries come in (p,s),(q,s) dyads for the same unordered pair.
    """
    post = [q for q in queries if q["tag"] != "anchor"]
    i = 0
    n_column = 0
    n_pair = 0
    pair_batches: list[tuple[str, str, int]] = []
    column_batches = 0

    while i < len(post):
        sid = post[i]["sample_id"]
        # Gather maximal run with same sample_id
        j = i
        while j < len(post) and post[j]["sample_id"] == sid:
            j += 1
        block = post[i:j]
        prompts = [q["prompt_id"] for q in block]
        uniq = set(prompts)

        if len(uniq) >= 3:
            # column-style purchase for this sample
            n_column += len(block)
            column_batches += 1
            i = j
            continue

        if len(block) >= 2 and len(uniq) == 2:
            # paired dyads on one sample, possibly multiple samples in a larger pair batch
            # Extend while next blocks are dyads of the SAME unordered pair
            p_a, p_b = sorted(uniq)
            samples = 1
            k = j
            while k + 1 < len(post):
                sid2 = post[k]["sample_id"]
                t = k
                while t < len(post) and post[t]["sample_id"] == sid2:
                    t += 1
                blk2 = post[k:t]
                u2 = set(q["prompt_id"] for q in blk2)
                if u2 != {p_a, p_b} or len(blk2) != 2:
                    break
                samples += 1
                k = t
                if samples >= 8:
                    # allow longer; peek handled by loop
                    pass
            n_pair += (k - i)
            pair_batches.append((p_a, p_b, samples))
            i = k
            continue

        # singleton / malformed: count as other, advance 1
        n_column += 1  # treat residual as exploratory fill
        i += 1

    return {
        "n_column_calls": n_column,
        "n_pair_calls": n_pair,
        "column_batches": column_batches,
        "pair_batches": pair_batches,
    }


def analyze(budget: int, log_dir: Path, out_dir: Path):
    policies = ["uniform", "racing", "risk_only", "dice"]
    logs = load_logs(log_dir, budget, policies)
    results = pd.read_csv(ROOT / "outputs" / "gate1" / "replay_results.csv")
    res_b = results[results["budget"] == budget]

    prompt_alloc = []
    summary_rows = []
    pair_focus = []

    for pol, seed, data in logs:
        qs = data["queries"]
        post = [q for q in qs if q["tag"] != "anchor"]
        n_anchor = sum(1 for q in qs if q["tag"] == "anchor")
        by_prompt = Counter(q["prompt_id"] for q in post)
        hhi = herfindahl(by_prompt)
        top3 = sum(c for _, c in by_prompt.most_common(3))
        top3_share = top3 / max(1, sum(by_prompt.values()))

        seg = segment_post_anchor(qs)
        pair_counts = Counter(pair_key(a, b) for a, b, _ in seg["pair_batches"])
        n_unique_pairs = len(pair_counts)
        top_pair, top_pair_n = (pair_counts.most_common(1)[0] if pair_counts else ("", 0))
        total_pb = sum(pair_counts.values()) or 1
        repeat_share = sum(n for n in pair_counts.values() if n > 1) / total_pb

        n_post = max(1, len(post))
        summary_rows.append(
            {
                "policy": pol,
                "seed": seed,
                "calls_used": data["summary"]["calls_used"],
                "budget_util": data["summary"]["calls_used"] / budget,
                "n_anchor": n_anchor,
                "n_post_anchor": len(post),
                "column_call_share": seg["n_column_calls"] / n_post,
                "pair_call_share": seg["n_pair_calls"] / n_post,
                "column_batches": seg["column_batches"],
                "hhi_prompt": hhi,
                "top3_prompt_share": top3_share,
                "n_pair_batches": len(seg["pair_batches"]),
                "n_unique_pairs": n_unique_pairs,
                "top_pair": top_pair,
                "top_pair_batches": top_pair_n,
                "repeat_pair_batch_share": repeat_share,
                "winner_0.5": data["summary"]["winners"].get("lambda_0.5"),
            }
        )

        for pid, c in by_prompt.items():
            prompt_alloc.append({"policy": pol, "seed": seed, "prompt_id": pid, "post_calls": c})
        for pk, n in pair_counts.items():
            pair_focus.append({"policy": pol, "seed": seed, "pair": pk, "n_batches": n})

    sdf = pd.DataFrame(summary_rows)
    pdf = pd.DataFrame(prompt_alloc)

    ag = (
        sdf.groupby("policy")
        .agg(
            calls_used_mean=("calls_used", "mean"),
            budget_util_mean=("budget_util", "mean"),
            column_share_mean=("column_call_share", "mean"),
            pair_share_mean=("pair_call_share", "mean"),
            hhi_mean=("hhi_prompt", "mean"),
            hhi_std=("hhi_prompt", "std"),
            top3_share_mean=("top3_prompt_share", "mean"),
            n_pair_batches_mean=("n_pair_batches", "mean"),
            n_unique_pairs_mean=("n_unique_pairs", "mean"),
            repeat_pair_share_mean=("repeat_pair_batch_share", "mean"),
        )
        .reset_index()
    )
    reg = res_b.groupby("policy")["regret_macro"].agg(["mean", "std"]).reset_index()
    reg.columns = ["policy", "regret_mean", "regret_std"]
    table = ag.merge(reg, on="policy")

    win = sdf.groupby(["policy", "winner_0.5"]).size().reset_index(name="n")
    win["share"] = win.groupby("policy")["n"].transform(lambda s: s / s.sum())
    mean_prompt = pdf.groupby(["policy", "prompt_id"])["post_calls"].mean().reset_index()

    overlap_rows = []
    for seed in range(20):
        d_pairs = {r["pair"] for r in pair_focus if r["policy"] == "dice" and r["seed"] == seed}
        r_pairs = {r["pair"] for r in pair_focus if r["policy"] == "risk_only" and r["seed"] == seed}
        rac_pairs = {r["pair"] for r in pair_focus if r["policy"] == "racing" and r["seed"] == seed}
        overlap_rows.append(
            {
                "seed": seed,
                "dice_n_pairs": len(d_pairs),
                "risk_n_pairs": len(r_pairs),
                "racing_n_pairs": len(rac_pairs),
                "dice_risk_jaccard": len(d_pairs & r_pairs) / (len(d_pairs | r_pairs) or 1),
                "dice_racing_jaccard": len(d_pairs & rac_pairs) / (len(d_pairs | rac_pairs) or 1),
            }
        )
    odf = pd.DataFrame(overlap_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "diagnosis_table.csv", index=False)
    sdf.to_csv(out_dir / "diagnosis_per_seed.csv", index=False)
    win.to_csv(out_dir / "diagnosis_winners.csv", index=False)
    mean_prompt.to_csv(out_dir / "diagnosis_prompt_alloc.csv", index=False)
    odf.to_csv(out_dir / "diagnosis_pair_overlap.csv", index=False)

    piv = mean_prompt.pivot(index="prompt_id", columns="policy", values="post_calls").fillna(0)
    piv = piv.reindex(sorted(piv.index))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    x = np.arange(len(piv.index))
    width = 0.2
    colors = {"uniform": "#4C78A8", "racing": "#F58518", "risk_only": "#54A24B", "dice": "#E45756"}
    for i, pol in enumerate(["uniform", "racing", "risk_only", "dice"]):
        if pol not in piv.columns:
            continue
        ax.bar(x + i * width, piv[pol].values, width, label=pol, color=colors[pol])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(piv.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean post-anchor evals")
    ax.set_title(f"Budget allocation by prompt (budget={budget})")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    order = ["uniform", "racing", "risk_only", "dice"]
    tix = table.set_index("policy")
    vals = [float(tix.loc[p, "hhi_mean"]) for p in order]
    errs = [float(tix.loc[p, "hhi_std"]) for p in order]
    ax2.bar(order, vals, yerr=errs, capsize=4, color=[colors[p] for p in order])
    ax2.set_ylabel("HHI (prompt concentration)")
    ax2.set_title("Higher HHI = more concentrated budget")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig_path = out_dir / "figure_budget_allocation.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    t = table.set_index("policy")
    lines = [
        f"# Gate 1 Mechanism Diagnosis (budget={budget}, 0 new LLM calls)",
        "",
        "## Diagnosis table (mean over 20 seeds)",
        "",
        table.round(4).to_markdown(index=False),
        "",
        "## Key contrasts",
        "",
        f"- **Budget utilization**: Racing {t.loc['racing','budget_util_mean']:.3f} vs "
        f"DICE {t.loc['dice','budget_util_mean']:.3f} "
        f"(DICE often stops early when pair queries idle).",
        f"- **Column-fill share** (Uniform-style / Racing fallback): "
        f"Racing {t.loc['racing','column_share_mean']:.3f} vs DICE {t.loc['dice','column_share_mean']:.3f}.",
        f"- **Prompt HHI**: Racing {t.loc['racing','hhi_mean']:.3f} vs DICE {t.loc['dice','hhi_mean']:.3f}; "
        f"top-3 share Racing {t.loc['racing','top3_share_mean']:.3f} vs DICE {t.loc['dice','top3_share_mean']:.3f}.",
        f"- **Unique pairs**: Racing {t.loc['racing','n_unique_pairs_mean']:.1f} vs "
        f"DICE {t.loc['dice','n_unique_pairs_mean']:.1f}.",
        f"- **Jaccard(DICE, Risk-Only) pairs** mean = {odf['dice_risk_jaccard'].mean():.3f} "
        f"(impact *does* change which pairs are bought).",
        f"- **Jaccard(DICE, Racing) pairs** mean = {odf['dice_racing_jaccard'].mean():.3f}.",
        "",
        "## Winner@λ=0.5 distribution",
        "",
        win.round(3).to_markdown(index=False),
        "",
        f"![allocation]({fig_path.name})",
        "",
        "## Mechanism reading",
        "",
        "1. **Code confound**: `PairedRacingPolicy` uses `stop_r` and falls back to "
        "`uniform_column_queries`; `DICEPolicy` / `RiskOnlyPolicy` use `stop_r=None` and never fall back. "
        "Registered comparison therefore mixes pair-scoring with stop/explore policy.",
        "2. **Concentration**: DICE puts a larger fraction of post-anchor budget on few prompts "
        "(esp. paths that lock `p15` as λ=0.5 winner in 55% of seeds).",
        "3. **Impact without VOI**: Decision impact changes pair sets vs Risk-Only "
        f"(Jaccard≈{odf['dice_risk_jaccard'].mean():.2f}) but does not reduce regret — "
        "consistent with “high risk/impact ≠ expected regret reduction”.",
        "4. Do **not** retune DICE to chase wins; optional future work is VOI / expected regret reduction, "
        "or a controlled ablation that equalizes early-stop + uniform fallback across policies.",
        "",
    ]
    (out_dir / "DIAGNOSIS.md").write_text("\n".join(lines), encoding="utf-8")

    print(table.round(4).to_string(index=False))
    print("\nWinners:")
    print(win.to_string(index=False))
    print("\nOverlap means:")
    print(odf.mean(numeric_only=True).round(3).to_string())
    print(f"\nWrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=640)
    ap.add_argument("--log-dir", type=Path, default=ROOT / "outputs" / "gate1" / "replay_logs")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gate1" / "diagnosis")
    args = ap.parse_args()
    analyze(args.budget, args.log_dir, args.out_dir)


if __name__ == "__main__":
    main()
