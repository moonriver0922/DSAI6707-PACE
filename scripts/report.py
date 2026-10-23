#!/usr/bin/env python3
"""Regenerate Gate 1 summary tables/figures from replay_results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("outputs/gate1/replay_results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gate1"))
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    # Figure 2 data: mean regret vs budget
    g = (
        df.groupby(["policy", "budget"])["regret_macro"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g.to_csv(args.out_dir / "figure2_regret_vs_budget.csv", index=False)
    # Table 2 ablation: risk_only vs dice at primary budgets
    abl = df[df["policy"].isin(["risk_only", "dice"])]
    abl_g = (
        abl.groupby(["policy", "budget"])["regret_macro"].agg(["mean", "std"]).reset_index()
    )
    abl_g.to_csv(args.out_dir / "table2_ablation.csv", index=False)
    print(g.to_string(index=False))
    print("--- ablation ---")
    print(abl_g.to_string(index=False))


if __name__ == "__main__":
    main()
