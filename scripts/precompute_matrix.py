#!/usr/bin/env python3
"""Gate 1: one-shot LLM batch over 16 prompts × (160 search + 200 test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache import DecodeConfig, EvaluationCache  # noqa: E402
from dataset import load_json, load_prompt_pool  # noqa: E402
from inference import QwenPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "gate1.yaml")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "split_manifest.json")
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompt_pool.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gate1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", type=int, default=0, help="If >0, only evaluate this many samples per split")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompts = load_prompt_pool(args.prompts)
    prompt_ids = sorted(prompts.keys())
    manifest = load_json(args.manifest)
    search = manifest["search"]
    test = manifest["test"]
    if args.smoke > 0:
        search = search[: args.smoke]
        test = test[: args.smoke]

    decode = DecodeConfig(max_new_tokens=cfg["max_new_tokens"], do_sample=cfg["do_sample"])
    cache = EvaluationCache()
    predictor = QwenPredictor(
        cfg["model_id"],
        decode,
        cache,
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", 8),
        device=args.device,
    )

    def eval_split(rows, tag: str):
        sample_ids = [r["sample_id"] for r in rows]
        n_p, n_s = len(prompt_ids), len(sample_ids)
        correct = np.zeros((n_p, n_s), dtype=np.int8)
        tin = np.zeros((n_p, n_s), dtype=np.int32)
        tout = np.zeros((n_p, n_s), dtype=np.int32)
        logs = []
        for j, row in enumerate(tqdm(rows, desc=tag)):
            for i, pid in enumerate(prompt_ids):
                pred = predictor.predict(
                    pid, prompts[pid], row["sample_id"], row["text"], int(row["label_id"])
                )
                correct[i, j] = pred.correct
                tin[i, j] = pred.input_tokens
                tout[i, j] = pred.output_tokens
                logs.append(
                    {
                        "split": tag,
                        "prompt_id": pid,
                        "sample_id": row["sample_id"],
                        "correct": pred.correct,
                        "input_tokens": pred.input_tokens,
                        "output_tokens": pred.output_tokens,
                        "raw_output": pred.raw_output,
                    }
                )
        return sample_ids, correct, tin, tout, logs

    args.out_dir.mkdir(parents=True, exist_ok=True)
    s_ids, s_c, s_in, s_out, s_logs = eval_split(search, "search")
    t_ids, t_c, t_in, t_out, t_logs = eval_split(test, "test")

    np.savez_compressed(
        args.out_dir / "oracle_matrices.npz",
        prompt_ids=np.array(prompt_ids),
        search_ids=np.array(s_ids),
        test_ids=np.array(t_ids),
        search_correct=s_c,
        search_input_tokens=s_in,
        search_output_tokens=s_out,
        test_correct=t_c,
        test_input_tokens=t_in,
        test_output_tokens=t_out,
    )
    # Prefer parquet when pandas/pyarrow exist; otherwise JSONL (GPU envs may lack pandas).
    all_logs = s_logs + t_logs
    try:
        import pandas as pd

        pd.DataFrame(all_logs).to_parquet(args.out_dir / "queries.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        jsonl = args.out_dir / "queries.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for row in all_logs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"parquet unavailable ({exc}); wrote {jsonl}")
    meta = {
        "model_id": cfg["model_id"],
        "n_prompts": len(prompt_ids),
        "n_search": len(s_ids),
        "n_test": len(t_ids),
        "total_evals": len(s_ids) * len(prompt_ids) + len(t_ids) * len(prompt_ids),
        "smoke": args.smoke,
    }
    (args.out_dir / "precompute_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
