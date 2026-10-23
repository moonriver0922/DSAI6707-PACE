"""Observed search state. Never contains test rows or hidden oracle values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

TAG_ANCHOR = "anchor"
TAG_UNIFORM = "uniform"
TAG_PAIR = "pair"


@dataclass
class TokenCount:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> float:
        return float(self.input_tokens + self.output_tokens)


@dataclass
class SearchState:
    prompt_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    prompt_texts: dict[str, str]
    observed: dict[tuple[str, str], int]
    tokens: dict[tuple[str, str], TokenCount]
    tags: dict[tuple[str, str], str]
    calls_used: int
    calls_budget: int
    rng_seed: int
    cost_norm_min: float
    cost_norm_max: float
    w_in: float = 1.0
    w_out: float = 1.0
    n_bootstrap: int = 200
    lambdas: tuple[float, ...] = (0.25, 0.5, 0.75)
    n_contenders: int = 6
    max_pairs: int = 8
    pair_batch_samples: int = 8
    racing_stop_r: float = 0.05
    query_log: list[dict] = field(default_factory=list)

    def remaining_calls(self) -> int:
        return int(self.calls_budget - self.calls_used)

    def rng(self) -> np.random.Generator:
        ss = np.random.SeedSequence([int(self.rng_seed), int(self.calls_used)])
        return np.random.default_rng(ss)

    def is_observed(self, prompt_id: str, sample_id: str) -> bool:
        return (prompt_id, sample_id) in self.observed

    def common_samples(self, p: str, q: str) -> list[str]:
        return [
            s
            for s in self.sample_ids
            if (p, s) in self.observed and (q, s) in self.observed
        ]

    def _mean_correct(self, prompt_id: str, allowed_tags: Optional[set[str]]) -> float:
        vals = []
        for s in self.sample_ids:
            key = (prompt_id, s)
            if key not in self.observed:
                continue
            if allowed_tags is not None and self.tags.get(key) not in allowed_tags:
                continue
            vals.append(self.observed[key])
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def _mean_cost(self, prompt_id: str, allowed_tags: Optional[set[str]]) -> float:
        vals = []
        for s in self.sample_ids:
            key = (prompt_id, s)
            if key not in self.tokens:
                continue
            if allowed_tags is not None and self.tags.get(key) not in allowed_tags:
                continue
            tok = self.tokens[key]
            vals.append(self.w_in * tok.input_tokens + self.w_out * tok.output_tokens)
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def accuracy_pop(self, prompt_id: str) -> float:
        """Population estimator on uniform / random common samples.

        v1 pair batches are random shared items (not difficulty-targeted), so
        TAG_PAIR is included. Future active item selection must exclude
        non-uniform tags here and use a separate estimator.
        """
        return self._mean_correct(prompt_id, {TAG_ANCHOR, TAG_UNIFORM, TAG_PAIR})

    def cost_pop(self, prompt_id: str) -> float:
        return self._mean_cost(prompt_id, {TAG_ANCHOR, TAG_UNIFORM, TAG_PAIR})

    def accuracy_all(self, prompt_id: str) -> float:
        """SRS mean of all purchased items (pair rows are still random items in v1)."""
        return self._mean_correct(prompt_id, None)

    def cost_all(self, prompt_id: str) -> float:
        return self._mean_cost(prompt_id, None)

    def cost_norm(self, cost: float) -> float:
        from pareto import normalize_cost

        return normalize_cost(cost, self.cost_norm_min, self.cost_norm_max)

    def n_obs(self, prompt_id: str) -> int:
        return sum(1 for s in self.sample_ids if (prompt_id, s) in self.observed)

    def snapshot_estimates(self, use_pop: bool = True) -> tuple[dict[str, float], dict[str, float]]:
        acc = {}
        cost = {}
        for p in self.prompt_ids:
            acc[p] = self.accuracy_pop(p) if use_pop else self.accuracy_all(p)
            cost[p] = self.cost_pop(p) if use_pop else self.cost_all(p)
        return acc, cost


def empty_state(
    prompt_ids: Iterable[str],
    sample_ids: Iterable[str],
    prompt_texts: dict[str, str],
    *,
    calls_budget: int,
    rng_seed: int,
    cost_norm_min: float,
    cost_norm_max: float,
    **kwargs,
) -> SearchState:
    return SearchState(
        prompt_ids=tuple(prompt_ids),
        sample_ids=tuple(sample_ids),
        prompt_texts=dict(prompt_texts),
        observed={},
        tokens={},
        tags={},
        calls_used=0,
        calls_budget=calls_budget,
        rng_seed=rng_seed,
        cost_norm_min=cost_norm_min,
        cost_norm_max=cost_norm_max,
        **kwargs,
    )
