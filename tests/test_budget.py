"""Budget accounting: cache hits free; remaining=1 forbids a 2-call pair."""

from __future__ import annotations

import numpy as np
import pytest

from policies import get_policy
from replay import BudgetExceeded, apply_queries, initialize_common_anchor, run_policy_until_budget
from state import TAG_PAIR, empty_state


def test_cache_hit_does_not_consume_budget(synthetic_oracle, fresh_state):
    rng = np.random.default_rng(0)
    initialize_common_anchor(fresh_state, synthetic_oracle, 2, rng)
    used = fresh_state.calls_used
    # Re-request an already observed cell
    p = fresh_state.prompt_ids[0]
    s = next(iter({sid for (pid, sid) in fresh_state.observed if pid == p}))
    n = apply_queries(fresh_state, synthetic_oracle, [(p, s)], TAG_PAIR)
    assert n == 0
    assert fresh_state.calls_used == used


def test_cannot_exceed_budget(synthetic_oracle, prompt_ids, sample_ids, prompt_texts):
    state = empty_state(
        prompt_ids,
        sample_ids,
        prompt_texts,
        calls_budget=5,
        rng_seed=0,
        cost_norm_min=0.0,
        cost_norm_max=256.0,
    )
    queries = [(prompt_ids[0], sample_ids[i]) for i in range(6)]
    with pytest.raises(BudgetExceeded):
        apply_queries(state, synthetic_oracle, queries, TAG_PAIR)


def test_remaining_one_rejects_full_new_pair(synthetic_oracle, prompt_ids, sample_ids, prompt_texts):
    state = empty_state(
        prompt_ids,
        sample_ids,
        prompt_texts,
        calls_budget=100,
        rng_seed=3,
        cost_norm_min=0.0,
        cost_norm_max=256.0,
        pair_batch_samples=8,
        n_contenders=4,
        max_pairs=4,
        n_bootstrap=20,
    )
    rng = np.random.default_rng(3)
    initialize_common_anchor(state, synthetic_oracle, 3, rng)
    # Artificially leave only 1 call
    state.calls_budget = state.calls_used + 1
    for name in ("racing", "risk_only", "dice"):
        policy = get_policy(name)
        q = policy.select_queries(state, remaining_calls=1)
        # Either empty, or only queries that are already cached (cost 0 after apply)
        new_needed = sum(1 for item in q if item not in state.observed)
        assert new_needed <= 1
        # Must not request a brand-new 2-call pair
        assert new_needed < 2 or len(q) == 0


def test_run_never_exceeds_budget(synthetic_oracle, prompt_ids, sample_ids, prompt_texts):
    state = empty_state(
        prompt_ids,
        sample_ids,
        prompt_texts,
        calls_budget=48,
        rng_seed=9,
        cost_norm_min=0.0,
        cost_norm_max=256.0,
        n_bootstrap=20,
        pair_batch_samples=4,
    )
    rng = np.random.default_rng(9)
    initialize_common_anchor(state, synthetic_oracle, 2, rng)
    run_policy_until_budget(get_policy("dice"), state, synthetic_oracle)
    assert state.calls_used <= state.calls_budget
