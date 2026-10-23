#!/usr/bin/env python3
"""Load Qwen2.5-7B-Instruct and run a single AG News classification smoke test."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache import DecodeConfig, EvaluationCache  # noqa: E402
from inference import QwenPredictor, parse_ag_news_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))

    decode = DecodeConfig(max_new_tokens=8, do_sample=False)
    cache = EvaluationCache()
    pred = QwenPredictor(args.model, decode, cache, device=args.device)
    text = (
        "The quarterback threw three touchdowns as the home team won the championship game."
    )
    template = (
        "Classify the news article into one of: World, Sports, Business, Sci/Tech.\n"
        "Article: {text}\nLabel:"
    )
    t0 = time.time()
    out = pred.predict("smoke", template, "smoke0", text, gold_label=1)
    dt = time.time() - t0
    print(
        {
            "raw": out.raw_output,
            "pred": out.pred_label,
            "parsed_ok": parse_ag_news_label(out.raw_output),
            "correct": out.correct,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "seconds": round(dt, 2),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else None,
        }
    )


if __name__ == "__main__":
    main()
