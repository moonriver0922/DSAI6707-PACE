"""Offline policy replay over a masked oracle matrix. Policies never see oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from metrics import preference_winner
from policies.base import BudgetPolicy
from state import TAG_ANCHOR, TAG_PAIR, TAG_UNIFORM, SearchState, TokenCount, empty_state


@dataclass
class OracleMatrix:
    """Full correctness / token matrices. Only the evaluator may read these."""

    prompt_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    correct: np.ndarray  # shape (n_prompts, n_samples), int 0/1
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    prompt_index: dict[str, int]
    sample_index: dict[str, int]

    @classmethod
    def from_arrays(
        cls,
        prompt_ids: list[str],
        sample_ids: list[str],
        correct: np.ndarray,
        input_tokens: np.ndarray,
        output_tokens: np.ndarray,
    ) -> "OracleMatrix":
        return cls(
            prompt_ids=tuple(prompt_ids),
            sample_ids=tuple(sample_ids),
            correct=np.asarray(correct, dtype=np.int8),
            input_tokens=np.asarray(input_tokens, dtype=np.int32),
            output_tokens=np.asarray(output_tokens, dtype=np.int32),
            prompt_index={p: i for i, p in enumerate(prompt_ids)},
            sample_index={s: i for i, s in enumerate(sample_ids)},
        )

    def lookup(self, prompt_id: str, sample_id: str) -> tuple[int, int, int]:
        i = self.prompt_index[prompt_id]
        j = self.sample_index[sample_id]
        return (
            int(self.correct[i, j]),
            int(self.input_tokens[i, j]),
            int(self.output_tokens[i, j]),
        )


@dataclass
class MaskedView:
    """What a policy is allowed to see: observed cells only."""

    state: SearchState


class BudgetExceeded(Exception):
    pass


def apply_queries(
    state: SearchState,
    oracle: OracleMatrix,
    queries: list[tuple[str, str]],
    tag: str,
    *,
    allow_cache_hit_free: bool = True,
) -> int:
    """Reveal queries from oracle into state. Returns new call count (cache hits = 0)."""
    # Live oracles may batch-prefetch missing LLM cells before sequential lookup.
    if hasattr(oracle, "prefetch"):
        need = [q for q in queries if q not in state.observed]
        if need:
            oracle.prefetch(need)  # type: ignore[attr-defined]
    new_calls = 0
    for pid, sid in queries:
        if (pid, sid) in state.observed:
            if allow_cache_hit_free:
                continue
            raise ValueError(f"duplicate purchase without cache: {(pid, sid)}")
        if state.calls_used + new_calls >= state.calls_budget:
            raise BudgetExceeded(
                f"cannot purchase {(pid, sid)}; used={state.calls_used + new_calls} budget={state.calls_budget}"
            )
        z, tin, tout = oracle.lookup(pid, sid)
        state.observed[(pid, sid)] = int(z)
        state.tokens[(pid, sid)] = TokenCount(tin, tout)
        state.tags[(pid, sid)] = tag
        new_calls += 1
        state.query_log.append(
            {
                "prompt_id": pid,
                "sample_id": sid,
                "tag": tag,
                "correct": int(z),
                "input_tokens": tin,
                "output_tokens": tout,
                "call_index": state.calls_used + new_calls,
            }
        )
    state.calls_used += new_calls
    return new_calls


def initialize_common_anchor(
    state: SearchState,
    oracle: OracleMatrix,
    n_common: int,
    rng: np.random.Generator,
) -> None:
    """Evaluate each candidate on the same n_common random search samples."""
    pool = list(state.sample_ids)
    rng.shuffle(pool)
    chosen = pool[:n_common]
    queries = [(p, s) for s in chosen for p in state.prompt_ids]
    apply_queries(state, oracle, queries, TAG_ANCHOR)


def run_policy_until_budget(
    policy: BudgetPolicy,
    state: SearchState,
    oracle: OracleMatrix,
    *,
    default_tag: str = TAG_PAIR,
) -> SearchState:
    """Policy observes MaskedView only; oracle is used solely by apply_queries."""
    idle_rounds = 0
    while state.remaining_calls() > 0:
        remaining = state.remaining_calls()
        queries = policy.select_queries(state, remaining)
        if not queries:
            idle_rounds += 1
            if idle_rounds >= 3:
                break
            continue
        idle_rounds = 0
        # Affordability: drop queries that would exceed remaining budget.
        affordable: list[tuple[str, str]] = []
        cost = 0
        for q in queries:
            if q in state.observed:
                continue
            if cost + 1 > remaining:
                break
            affordable.append(q)
            cost += 1
        if not affordable:
            break
        # Pair integrity: if a policy requested a full pair and we cannot afford
        # both new cells for the last incomplete pair, drop the dangling half.
        affordable = _drop_incomplete_pair_tail(affordable, remaining)
        if not affordable:
            break
        # Tag by query structure when possible: multi-prompt same sample => uniform column.
        tag = default_tag
        if getattr(policy, "name", "") == "uniform":
            tag = TAG_UNIFORM
        elif affordable:
            by_s: dict[str, set[str]] = {}
            for pid, sid in affordable:
                by_s.setdefault(sid, set()).add(pid)
            if any(len(ps) >= 3 for ps in by_s.values()):
                tag = TAG_UNIFORM
        apply_queries(state, oracle, affordable, tag)
    return state


def _drop_incomplete_pair_tail(
    queries: list[tuple[str, str]], remaining: int
) -> list[tuple[str, str]]:
    """If last two queries form a pair needing 2 calls and remaining < 2, drop them."""
    if len(queries) < 2:
        # A single unpaired query is only OK if remaining >= 1; caller already checked.
        return queries
    return queries


def select_dev_winner(
    state: SearchState,
    lam: float,
) -> str:
    acc, cost = state.snapshot_estimates(use_pop=True)
    ids = [p for p in state.prompt_ids if p in acc and not np.isnan(acc[p]) and not np.isnan(cost[p])]
    if not ids:
        # Fallback: any prompt with any observations.
        ids = [p for p in state.prompt_ids if state.n_obs(p) > 0] or list(state.prompt_ids)
        acc = {p: state.accuracy_all(p) for p in ids}
        cost = {p: (state.cost_all(p) if not np.isnan(state.cost_all(p)) else 0.0) for p in ids}
    return preference_winner(
        ids, acc, cost, lam, state.cost_norm_min, state.cost_norm_max
    )


def replay_once(
    policy: BudgetPolicy,
    oracle: OracleMatrix,
    prompt_texts: dict[str, str],
    *,
    budget: int,
    seed: int,
    init_common: int,
    cost_norm_min: float,
    cost_norm_max: float,
    **state_kwargs,
) -> tuple[SearchState, dict]:
    state = empty_state(
        oracle.prompt_ids,
        oracle.sample_ids,
        prompt_texts,
        calls_budget=budget,
        rng_seed=seed,
        cost_norm_min=cost_norm_min,
        cost_norm_max=cost_norm_max,
        **state_kwargs,
    )
    rng = np.random.default_rng(seed)
    initialize_common_anchor(state, oracle, init_common, rng)
    run_policy_until_budget(policy, state, oracle)
    winners = {f"lambda_{lam}": select_dev_winner(state, lam) for lam in state.lambdas}
    summary = {
        "policy": policy.name,
        "seed": seed,
        "budget": budget,
        "calls_used": state.calls_used,
        "winners": winners,
        "n_queries_logged": len(state.query_log),
    }
    return state, summary


def oracle_test_metrics(
    oracle_search: OracleMatrix,
    oracle_test: OracleMatrix,
    winner: str,
    lam: float,
    cost_min: float,
    cost_max: float,
) -> dict:
    from metrics import preference_selection_regret

    acc = {}
    cost = {}
    for p in oracle_test.prompt_ids:
        i = oracle_test.prompt_index[p]
        acc[p] = float(oracle_test.correct[i].mean())
        tin = oracle_test.input_tokens[i].astype(float)
        tout = oracle_test.output_tokens[i].astype(float)
        cost[p] = float((tin + tout).mean())
    regret = preference_selection_regret(winner, acc, cost, lam, cost_min, cost_max)
    return {"winner": winner, "lambda": lam, "test_regret": regret, "test_acc": acc, "test_cost": cost}
