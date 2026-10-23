"""Policies must not change when hidden (unobserved) oracle cells are flipped."""

from __future__ import annotations

import copy

import numpy as np

from policies import get_policy
from replay import apply_queries, initialize_common_anchor
from state import TAG_ANCHOR


def _clone_oracle_flip_hidden(oracle, state, seed=7):
    """Flip correctness of cells that are still unobserved."""
    flipped = copy.deepcopy(oracle)
    rng = np.random.default_rng(seed)
    for i, p in enumerate(oracle.prompt_ids):
        for j, s in enumerate(oracle.sample_ids):
            if (p, s) in state.observed:
                continue
            if rng.random() < 0.5:
                flipped.correct[i, j] = 1 - flipped.correct[i, j]
    return flipped


def test_policy_output_invariant_to_hidden_oracle(synthetic_oracle, fresh_state, prompt_texts):
    rng = np.random.default_rng(1)
    initialize_common_anchor(fresh_state, synthetic_oracle, n_common=4, rng=rng)

    for name in ("uniform", "racing", "risk_only", "dice"):
        state_a = copy.deepcopy(fresh_state)
        state_b = copy.deepcopy(fresh_state)
        # Flip hidden cells in a twin oracle; states are identical observations.
        _ = _clone_oracle_flip_hidden(synthetic_oracle, state_a)
        # Policies only receive state; flipping oracle cannot affect them unless leaked.
        policy = get_policy(name)
        q_a = policy.select_queries(state_a, remaining_calls=16)
        q_b = policy.select_queries(state_b, remaining_calls=16)
        assert q_a == q_b, f"{name} leaked oracle information"


def test_changing_unobserved_matrix_after_select_does_not_alter_prior_choice(
    synthetic_oracle, fresh_state
):
    rng = np.random.default_rng(2)
    initialize_common_anchor(fresh_state, synthetic_oracle, n_common=4, rng=rng)
    policy = get_policy("dice")
    before = policy.select_queries(fresh_state, 16)
    # Mutate entire unobserved region
    for i, p in enumerate(synthetic_oracle.prompt_ids):
        for j, s in enumerate(synthetic_oracle.sample_ids):
            if (p, s) not in fresh_state.observed:
                synthetic_oracle.correct[i, j] = 1 - synthetic_oracle.correct[i, j]
    after = policy.select_queries(fresh_state, 16)
    assert before == after
