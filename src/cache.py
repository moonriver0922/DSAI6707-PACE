"""Evaluation cache keyed by (model, prompt, sample, decode config)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DecodeConfig:
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 0.0

    def fingerprint(self) -> str:
        payload = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cache_key(model_id: str, prompt_id: str, sample_id: str, decode: DecodeConfig) -> str:
    raw = f"{model_id}|{prompt_id}|{sample_id}|{decode.fingerprint()}"
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CachedEval:
    correct: int
    input_tokens: int
    output_tokens: int
    raw_output: str = ""


class EvaluationCache:
    """In-memory / dict cache. Hits cost 0 search-evaluation calls."""

    def __init__(self) -> None:
        self._store: dict[str, CachedEval] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Optional[CachedEval]:
        return self._store.get(key)

    def put(self, key: str, value: CachedEval) -> None:
        self._store[key] = value

    def has(self, model_id: str, prompt_id: str, sample_id: str, decode: DecodeConfig) -> bool:
        return cache_key(model_id, prompt_id, sample_id, decode) in self._store

    def __len__(self) -> int:
        return len(self._store)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for k, v in self._store.items():
            rows.append({"key": k, **asdict(v)})
        return rows
