"""Frozen AG News splits. Test IDs are never given to policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SEARCH_SPLIT = "search"
TEST_SPLIT = "test"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    source_split: str
    source_index: int
    text: str
    label_id: int
    label_name: str


class SplitAccessError(Exception):
    """Raised when code without allow_test tries to load the sealed test split."""


def sample_id_for(source_split: str, source_index: int, text: str) -> str:
    h = hashlib.sha1(f"{source_split}|{source_index}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source_split}-{source_index:06d}-{h}"


def stratified_indices(labels: list[int], n: int, rng_seed: int) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(int(y), []).append(i)
    for y in by:
        rng.shuffle(by[y])
    classes = sorted(by)
    chosen: list[int] = []
    ptr = {y: 0 for y in classes}
    while len(chosen) < n:
        progress = False
        for y in classes:
            if len(chosen) >= n:
                break
            k = ptr[y]
            if k < len(by[y]):
                chosen.append(by[y][k])
                ptr[y] = k + 1
                progress = True
        if not progress:
            break
    return chosen[:n]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_prompt_pool(path: Path) -> dict[str, str]:
    data = load_json(path)
    texts = {p["id"]: p["text"] for p in data["prompts"]}
    return texts


def load_search_manifest(path: Path) -> list[Sample]:
    data = load_json(path)
    rows = []
    for r in data["search"]:
        rows.append(Sample(**r))
        if rows[-1].split != SEARCH_SPLIT:
            raise ValueError("search manifest contains non-search rows")
    return rows


def load_test_manifest(path: Path, *, allow_test: bool) -> list[Sample]:
    if not allow_test:
        raise SplitAccessError("policies and tuning must not load the test split")
    data = load_json(path)
    rows = []
    for r in data["test"]:
        rows.append(Sample(**r))
        if rows[-1].split != TEST_SPLIT:
            raise ValueError("test manifest contains non-test rows")
    return rows


def samples_to_manifest(search: list[Sample], test: list[Sample]) -> dict[str, Any]:
    return {
        "task": "ag_news",
        "search": [asdict(s) for s in search],
        "test": [asdict(s) for s in test],
        "n_search": len(search),
        "n_test": len(test),
        "search_ids": [s.sample_id for s in search],
        "test_ids": [s.sample_id for s in test],
    }


def assert_disjoint(search: list[Sample], test: list[Sample]) -> None:
    a = {s.sample_id for s in search}
    b = {s.sample_id for s in test}
    if a & b:
        raise ValueError("search/test sample IDs overlap")
    sa = {(s.source_split, s.source_index) for s in search}
    sb = {(s.source_split, s.source_index) for s in test}
    if sa & sb:
        raise ValueError("search/test source indices overlap")
