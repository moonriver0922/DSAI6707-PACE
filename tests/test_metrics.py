"""Pareto signs, utility normalization, HV, bootstrap edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_reversal_rate,
    preference_selection_regret,
    preference_winner,
)
from pareto import dominates_max_a_min_c, hypervolume_2d, normalize_cost, utility
from dataset import SplitAccessError, load_test_manifest


def test_dominance_max_a_min_c():
    assert dominates_max_a_min_c(0.9, 10, 0.8, 12)
    assert not dominates_max_a_min_c(0.8, 10, 0.9, 12)
    assert not dominates_max_a_min_c(0.9, 12, 0.8, 10)  # better acc worse cost: not dominate
    assert dominates_max_a_min_c(0.9, 10, 0.9, 12)


def test_utility_and_cost_norm():
    assert normalize_cost(128, 0, 256) == 0.5
    with pytest.raises(ValueError):
        normalize_cost(1, 5, 5)
    assert utility(1.0, 0.0, 0.5) == 0.5
    assert utility(0.0, 1.0, 0.5) == -0.5


def test_hypervolume_positive_and_monotone():
    ref_a, ref_c = 0.0, 1.05
    pts = [(0.8, 0.4), (0.6, 0.2)]
    hv = hypervolume_2d(pts, ref_a, ref_c)
    assert hv > 0
    hv2 = hypervolume_2d(pts + [(0.9, 0.3)], ref_a, ref_c)
    assert hv2 >= hv


def test_preference_regret_nonnegative():
    acc = {"a": 0.9, "b": 0.7}
    cost = {"a": 100.0, "b": 50.0}
    w = preference_winner(["a", "b"], acc, cost, 0.5, 0.0, 200.0)
    r = preference_selection_regret(w, acc, cost, 0.5, 0.0, 200.0)
    assert r == pytest.approx(0.0)
    r_bad = preference_selection_regret("b" if w == "a" else "a", acc, cost, 0.5, 0.0, 200.0)
    assert r_bad >= 0


def test_bootstrap_constant_and_empty():
    rng = np.random.default_rng(0)
    z = np.ones(5)
    r = paired_bootstrap_reversal_rate(z, z, 0.2, 0.3, rng, 50, "p", "q")
    assert r == 0.0
    r0 = paired_bootstrap_reversal_rate(np.array([]), np.array([]), 0.2, 0.3, rng, 10, "p", "q")
    assert r0 == 1.0
    summary = bootstrap_mean_ci([1, 2, 3], rng, n_boot=200)
    assert summary["ci_lo"] <= summary["mean"] <= summary["ci_hi"]


def test_test_split_guard(tmp_path):
    path = tmp_path / "split_manifest.json"
    path.write_text('{"test": []}')
    with pytest.raises(SplitAccessError):
        load_test_manifest(path, allow_test=False)
