"""Shared Gate 0 fixtures: synthetic 6×20 oracle matrix."""

from __future__ import annotations

import numpy as np
import pytest

from replay import OracleMatrix
from state import empty_state


@pytest.fixture
def prompt_ids():
    return [f"p{i}" for i in range(6)]


@pytest.fixture
def sample_ids():
    return [f"s{i:02d}" for i in range(20)]


@pytest.fixture
def prompt_texts(prompt_ids):
    return {p: f"Classify. Article: {{text}}\nLabel: [{p}]" for p in prompt_ids}


@pytest.fixture
def synthetic_oracle(prompt_ids, sample_ids):
    """6×20 binary correctness with distinct costs; deterministic."""
    rng = np.random.default_rng(0)
    n_p, n_s = len(prompt_ids), len(sample_ids)
    # Accuracies roughly decreasing with prompt index; costs increasing.
    base = np.linspace(0.85, 0.55, n_p)
    correct = np.zeros((n_p, n_s), dtype=np.int8)
    for i, rate in enumerate(base):
        correct[i] = (rng.random(n_s) < rate).astype(np.int8)
    # Make a few prompts nearly tied on accuracy but different cost (H3-like).
    correct[1] = correct[0].copy()
    correct[1, :3] = 1 - correct[1, :3]
    input_tokens = np.tile(np.arange(40, 40 + n_p * 5, 5, dtype=np.int32)[:, None], (1, n_s))
    output_tokens = np.full((n_p, n_s), 4, dtype=np.int32)
    return OracleMatrix.from_arrays(prompt_ids, sample_ids, correct, input_tokens, output_tokens)


@pytest.fixture
def fresh_state(prompt_ids, sample_ids, prompt_texts):
    return empty_state(
        prompt_ids,
        sample_ids,
        prompt_texts,
        calls_budget=128,
        rng_seed=42,
        cost_norm_min=0.0,
        cost_norm_max=256.0,
        n_bootstrap=50,
        n_contenders=4,
        max_pairs=6,
        pair_batch_samples=4,
    )
