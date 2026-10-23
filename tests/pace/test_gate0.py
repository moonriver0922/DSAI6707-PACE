"""PACE Gate 0 — CPU correctness."""

from __future__ import annotations

import numpy as np
import pytest

from pace.evidence import confidence_eliminate, paired_delta_ci
from pace.survivor import pace_bounded_survive, point_estimate_truncate
from replay import OracleMatrix, apply_queries, initialize_common_anchor
from state import TAG_ANCHOR, empty_state


@pytest.fixture
def toy_state_and_oracle():
    prompt_ids = [f"p{i}" for i in range(6)]
    sample_ids = [f"s{i:02d}" for i in range(20)]
    texts = {p: "{text}" for p in prompt_ids}
    rng = np.random.default_rng(0)
    # Make p0 dominate p5 on accuracy with lower cost
    correct = np.zeros((6, 20), dtype=np.int8)
    correct[0] = 1
    correct[1] = (rng.random(20) < 0.7).astype(np.int8)
    correct[2] = (rng.random(20) < 0.65).astype(np.int8)
    correct[3] = (rng.random(20) < 0.6).astype(np.int8)
    correct[4] = (rng.random(20) < 0.55).astype(np.int8)
    correct[5] = 0
    tin = np.tile(np.arange(40, 70, 5, dtype=np.int32)[:, None], (1, 20))
    tout = np.full((6, 20), 4, dtype=np.int32)
    oracle = OracleMatrix.from_arrays(prompt_ids, sample_ids, correct, tin, tout)
    state = empty_state(
        prompt_ids,
        sample_ids,
        texts,
        calls_budget=200,
        rng_seed=1,
        cost_norm_min=0.0,
        cost_norm_max=256.0,
        n_bootstrap=50,
    )
    initialize_common_anchor(state, oracle, n_common=10, rng=np.random.default_rng(1))
    return state, oracle


def test_paired_sample_ids_match(toy_state_and_oracle):
    state, _ = toy_state_and_oracle
    common = state.common_samples("p0", "p5")
    assert len(common) >= 8
    assert all((("p0", s) in state.observed and ("p5", s) in state.observed) for s in common)


def test_confidence_elim_with_clear_dominance(toy_state_and_oracle):
    state, _ = toy_state_and_oracle
    rng = np.random.default_rng(2)
    elims = confidence_eliminate(
        state,
        list(state.prompt_ids),
        epsilon=0.0,
        n_min_common=8,
        n_bootstrap=80,
        rng=rng,
    )
    elim_q = {e[0] for e in elims}
    # p5 is always wrong and costly-ish; should often be eliminated by cheaper/better p0
    assert "p5" in elim_q


def test_insufficient_common_means_undecided():
    prompt_ids = ["a", "b"]
    sample_ids = [f"s{i}" for i in range(10)]
    state = empty_state(
        prompt_ids,
        sample_ids,
        {"a": "{text}", "b": "{text}"},
        calls_budget=50,
        rng_seed=0,
        cost_norm_min=0.0,
        cost_norm_max=100.0,
    )
    # Only 2 common — below n_min
    state.observed[("a", "s0")] = 1
    state.observed[("b", "s0")] = 0
    state.observed[("a", "s1")] = 1
    state.observed[("b", "s1")] = 0
    from state import TokenCount, TAG_ANCHOR

    for s in ("s0", "s1"):
        state.tokens[("a", s)] = TokenCount(10, 2)
        state.tokens[("b", s)] = TokenCount(12, 2)
        state.tags[("a", s)] = TAG_ANCHOR
        state.tags[("b", s)] = TAG_ANCHOR
    elims = confidence_eliminate(
        state, prompt_ids, epsilon=0.0, n_min_common=8, n_bootstrap=20, rng=np.random.default_rng(0)
    )
    assert elims == []


def test_point_truncate_capacity(toy_state_and_oracle):
    state, _ = toy_state_and_oracle
    r = point_estimate_truncate(state, list(state.prompt_ids), mu=3)
    assert len(r.retained) == 3


def test_pace_logs_forced_eviction_under_tiny_cap(toy_state_and_oracle):
    state, _ = toy_state_and_oracle
    # Raise epsilon so confidence elim unlikely; force capacity eviction
    r = pace_bounded_survive(
        state,
        list(state.prompt_ids),
        mu=2,
        archive_k=0,
        epsilon=10.0,  # impossible L > 10
        n_min_common=8,
        n_bootstrap=30,
        rng=np.random.default_rng(3),
    )
    assert len(r.retained) <= 2
    assert len(r.forced_evictions) >= 1


def test_no_oracle_in_confidence_gate(toy_state_and_oracle):
    state, oracle = toy_state_and_oracle
    # Flip unobserved cells — should not change gate given same observations
    rng = np.random.default_rng(4)
    before = confidence_eliminate(
        state, list(state.prompt_ids), epsilon=0.0, n_min_common=8, n_bootstrap=40, rng=rng
    )
    for i, p in enumerate(oracle.prompt_ids):
        for j, s in enumerate(oracle.sample_ids):
            if (p, s) not in state.observed:
                oracle.correct[i, j] = 1 - oracle.correct[i, j]
    rng2 = np.random.default_rng(4)
    after = confidence_eliminate(
        state, list(state.prompt_ids), epsilon=0.0, n_min_common=8, n_bootstrap=40, rng=rng2
    )
    assert [(a[0], a[1]) for a in before] == [(a[0], a[1]) for a in after]
