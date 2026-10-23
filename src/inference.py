"""Local Qwen inference: greedy, max_new_tokens=8, true token counts, batched generate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from cache import CachedEval, DecodeConfig, EvaluationCache, cache_key
from dataset import AG_NEWS_LABELS

LABEL_PATTERNS = [
    (3, re.compile(r"sci\s*/?\s*tech|science|technology", re.I)),
    (1, re.compile(r"\bsports?\b", re.I)),
    (2, re.compile(r"\bbusiness\b|\bmarkets?\b|\beconomy\b", re.I)),
    (0, re.compile(r"\bworld\b|\binternational\b", re.I)),
]
NUM_MAP = {"1": 0, "2": 1, "3": 2, "4": 3}


def parse_ag_news_label(raw: str) -> Optional[int]:
    text = raw.strip()
    if not text:
        return None
    head = text.splitlines()[0][:80]
    m = re.search(r"\{?\s*['\"]?label['\"]?\s*[:=]\s*['\"]?([A-Za-z/]+)", head, re.I)
    if m:
        head = m.group(1)
    for lab_id, pat in LABEL_PATTERNS:
        if pat.search(head):
            return lab_id
    digit = re.search(r"[1-4]", head)
    if digit:
        return NUM_MAP[digit.group(0)]
    lower = head.lower()
    for i, name in enumerate(AG_NEWS_LABELS):
        if name.lower() in lower:
            return i
    return None


def render_prompt(template: str, text: str) -> str:
    return template.replace("{text}", text)


@dataclass
class Prediction:
    prompt_id: str
    sample_id: str
    raw_output: str
    pred_label: Optional[int]
    correct: int
    input_tokens: int
    output_tokens: int


class Predictor(Protocol):
    def predict(
        self,
        prompt_id: str,
        prompt_template: str,
        sample_id: str,
        sample_text: str,
        gold_label: int,
    ) -> Prediction: ...


class MockPredictor:
    """Deterministic predictor for Gate 0 / offline tests. No GPU."""

    def __init__(self, oracle_correct: dict[tuple[str, str], int], oracle_tokens: dict[tuple[str, str], tuple[int, int]]):
        self.oracle_correct = oracle_correct
        self.oracle_tokens = oracle_tokens

    def predict(self, prompt_id, prompt_template, sample_id, sample_text, gold_label) -> Prediction:
        key = (prompt_id, sample_id)
        correct = int(self.oracle_correct[key])
        tin, tout = self.oracle_tokens[key]
        raw = AG_NEWS_LABELS[gold_label] if correct else AG_NEWS_LABELS[(gold_label + 1) % 4]
        return Prediction(prompt_id, sample_id, raw, gold_label if correct else (gold_label + 1) % 4, correct, tin, tout)


class QwenPredictor:
    def __init__(
        self,
        model_id: str,
        decode: DecodeConfig,
        cache: EvaluationCache,
        dtype: str = "bfloat16",
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.decode = decode
        self.cache = cache
        self.dtype = dtype
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = torch.bfloat16 if self.dtype in ("bfloat16", "bf16") else torch.float16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=False)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch_dtype, device_map=None, **kwargs
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch_dtype, device_map=None, **kwargs
            )
        self._model.to(self.device)
        self._model.eval()

    def _from_cache(self, prompt_id: str, sample_id: str) -> Optional[Prediction]:
        key = cache_key(self.model_id, prompt_id, sample_id, self.decode)
        hit = self.cache.get(key)
        if hit is None:
            return None
        pred = parse_ag_news_label(hit.raw_output) if hit.raw_output else None
        return Prediction(
            prompt_id,
            sample_id,
            hit.raw_output,
            pred,
            hit.correct,
            hit.input_tokens,
            hit.output_tokens,
        )

    def predict(self, prompt_id, prompt_template, sample_id, sample_text, gold_label) -> Prediction:
        hit = self._from_cache(prompt_id, sample_id)
        if hit is not None:
            return hit
        return self.predict_batch(
            [(prompt_id, prompt_template, sample_id, sample_text, gold_label)]
        )[0]

    def predict_batch(
        self,
        items: Sequence[tuple[str, str, str, str, int]],
    ) -> list[Prediction]:
        """Batched greedy generate. items: (prompt_id, template, sample_id, text, gold)."""
        if not items:
            return []
        outs: list[Optional[Prediction]] = [None] * len(items)
        miss_idx: list[int] = []
        miss_items: list[tuple[str, str, str, str, int]] = []
        for i, it in enumerate(items):
            hit = self._from_cache(it[0], it[2])
            if hit is not None:
                outs[i] = hit
            else:
                miss_idx.append(i)
                miss_items.append(it)
        if not miss_items:
            return [o for o in outs if o is not None]

        self._load()
        import torch

        for start in range(0, len(miss_items), self.batch_size):
            chunk = miss_items[start : start + self.batch_size]
            chunk_pos = miss_idx[start : start + self.batch_size]
            texts = []
            for prompt_id, template, sample_id, sample_text, gold in chunk:
                user = render_prompt(template, sample_text)
                messages = [{"role": "user", "content": user}]
                texts.append(
                    self._tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                )
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = enc["input_ids"].to(self._model.device)
            attn = enc["attention_mask"].to(self._model.device)
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=self.decode.max_new_tokens,
                do_sample=self.decode.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            if self.decode.do_sample:
                gen_kwargs["temperature"] = self.decode.temperature
            with torch.no_grad():
                out = self._model.generate(**gen_kwargs)
            in_lens = attn.sum(dim=1).tolist()
            for row, (prompt_id, template, sample_id, sample_text, gold), in_len, pos in zip(
                range(len(chunk)), chunk, in_lens, chunk_pos
            ):
                # left-padded: generated tokens start after full sequence width
                gen = out[row, input_ids.shape[1] :]
                # Also strip any accidental pad in decode
                raw = self._tokenizer.decode(gen, skip_special_tokens=True)
                pred = parse_ag_news_label(raw)
                correct = int(pred is not None and pred == gold)
                tin = int(in_len)
                tout = int(gen.shape[0])
                # Prefer counting non-pad new tokens if available
                if self._tokenizer.pad_token_id is not None:
                    tout = int((gen != self._tokenizer.pad_token_id).sum().item())
                rec = CachedEval(correct, tin, max(tout, 1), raw)
                self.cache.put(cache_key(self.model_id, prompt_id, sample_id, self.decode), rec)
                outs[pos] = Prediction(prompt_id, sample_id, raw, pred, correct, tin, rec.output_tokens)
        return [o for o in outs if o is not None]


def preload_gate1_matrix_into_cache(
    cache: EvaluationCache,
    model_id: str,
    decode: DecodeConfig,
    npz_path: str | Path,
    prompt_ids: Sequence[str] | None = None,
    split: str = "search",
) -> int:
    """Warm cache from Gate-1 oracle matrix (correct/token counts; raw may be empty)."""
    import numpy as np

    path = Path(npz_path)
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=True)
    all_pids = [str(x) for x in data["prompt_ids"].tolist()]
    if split == "search":
        sids = [str(x) for x in data["search_ids"].tolist()]
        correct = data["search_correct"]
        tin = data["search_input_tokens"]
        tout = data["search_output_tokens"]
    else:
        sids = [str(x) for x in data["test_ids"].tolist()]
        correct = data["test_correct"]
        tin = data["test_input_tokens"]
        tout = data["test_output_tokens"]
    want = set(prompt_ids) if prompt_ids is not None else set(all_pids)
    n = 0
    for i, pid in enumerate(all_pids):
        if pid not in want:
            continue
        for j, sid in enumerate(sids):
            key = cache_key(model_id, pid, sid, decode)
            if key in cache:
                continue
            cache.put(
                key,
                CachedEval(
                    correct=int(correct[i, j]),
                    input_tokens=int(tin[i, j]),
                    output_tokens=int(tout[i, j]),
                    raw_output="",
                ),
            )
            n += 1
    return n
