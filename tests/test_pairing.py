"""Paired sample IDs: same rows for both candidates; deterministic seeds."""

from __future__ import annotations

import numpy as np

from policies.common import pair_queries
from replay import initialize_common_anchor, replay_once
from policies import get_policy
from state import empty_state


def test_pair_queries_share_sample_ids(fresh_state, synthetic_oracle):
    rng = np.random.default_rng(5)
    initialize_common_anchor(fresh_state, synthetic_oracle, 2, rng)
    p, q = fresh_state.prompt_ids[0], fresh_state.prompt_ids[1]
    queries = pair_queries(fresh_state, p, q, remaining_calls=16, rng=rng)
    assert queries
    # Group by sample: each new sample should appear for both prompts
    by_sample = {}
    for pid, sid in queries:
        by_sample.setdefault(sid, set()).add(pid)
    for sid, prompts in by_sample.items():
        assert prompts == {p, q}


def test_common_anchor_uses_identical_samples(fresh_state, synthetic_oracle):
    rng = np.random.default_rng(11)
    initialize_common_anchor(fresh_state, synthetic_oracle, 4, rng)
    sets = []
    for p in fresh_state.prompt_ids:
        sets.append({s for (pid, s) in fresh_state.observed if pid == p})
    assert all(s == sets[0] for s in sets)
    assert len(sets[0]) == 4


def test_deterministic_replay(synthetic_oracle, prompt_texts):
    kwargs = dict(
        budget=64,
        init_common=4,
        cost_norm_min=0.0,
        cost_norm_max=256.0,
        n_bootstrap=30,
        pair_batch_samples=4,
        n_contenders=4,
        max_pairs=4,
    )
    s1, sum1 = replay_once(get_policy("dice"), synthetic_oracle, prompt_texts, seed=123, **kwargs)
    s2, sum2 = replay_once(get_policy("dice"), synthetic_oracle, prompt_texts, seed=123, **kwargs)
    assert sum1["calls_used"] == sum2["calls_used"]
    assert sum1["winners"] == sum2["winners"]
    assert [(q["prompt_id"], q["sample_id"]) for q in s1.query_log] == [
        (q["prompt_id"], q["sample_id"]) for q in s2.query_log
    ]
