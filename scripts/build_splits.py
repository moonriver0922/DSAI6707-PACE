#!/usr/bin/env python3
"""Freeze AG News search/test sample IDs into data/split_manifest.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import (  # noqa: E402
    AG_NEWS_LABELS,
    SEARCH_SPLIT,
    TEST_SPLIT,
    Sample,
    assert_disjoint,
    sample_id_for,
    save_json,
    stratified_indices,
)


def load_ag_news():
    from datasets import load_dataset

    # Hub requires namespace/name; classic "ag_news" short id is rejected by current clients.
    train = load_dataset("fancyzhx/ag_news", split="train")
    test = load_dataset("fancyzhx/ag_news", split="test")
    return train, test


def rows_from_hf(split_name: str, ds, indices: list[int]) -> list[Sample]:
    out = []
    for idx in indices:
        text = ds[int(idx)]["text"]
        label = int(ds[int(idx)]["label"])
        sid = sample_id_for(split_name, int(idx), text)
        out.append(
            Sample(
                sample_id=sid,
                split=SEARCH_SPLIT if split_name == "train" else TEST_SPLIT,
                source_split=split_name,
                source_index=int(idx),
                text=text,
                label_id=label,
                label_name=AG_NEWS_LABELS[label],
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-search", type=int, default=160)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=6707)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "split_manifest.json")
    args = ap.parse_args()

    train, test = load_ag_news()
    train_labels = [int(x) for x in train["label"]]
    test_labels = [int(x) for x in test["label"]]
    search_idx = stratified_indices(train_labels, args.n_search, args.seed)
    test_idx = stratified_indices(test_labels, args.n_test, args.seed + 1)
    search = rows_from_hf("train", train, search_idx)
    test_rows = rows_from_hf("test", test, test_idx)
    for r in search:
        object.__setattr__(r, "split", SEARCH_SPLIT) if False else None
    # Sample is frozen dataclass — rebuild with correct split tag
    search = [
        Sample(
            sample_id=s.sample_id,
            split=SEARCH_SPLIT,
            source_split=s.source_split,
            source_index=s.source_index,
            text=s.text,
            label_id=s.label_id,
            label_name=s.label_name,
        )
        for s in search
    ]
    test_rows = [
        Sample(
            sample_id=s.sample_id,
            split=TEST_SPLIT,
            source_split=s.source_split,
            source_index=s.source_index,
            text=s.text,
            label_id=s.label_id,
            label_name=s.label_name,
        )
        for s in test_rows
    ]
    assert_disjoint(search, test_rows)
    from dataset import samples_to_manifest

    manifest = samples_to_manifest(search, test_rows)
    manifest["seed"] = args.seed
    save_json(args.out, manifest)
    print(f"Wrote {args.out} search={len(search)} test={len(test_rows)}")


if __name__ == "__main__":
    main()
