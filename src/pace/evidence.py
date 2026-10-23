"""Paired bootstrap CI and evidence-gated elimination (heuristic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from state import SearchState


@dataclass(frozen=True)
class PairCI:
    p: str
    q: str
    n_common: int
    delta_hat: float
    L: float
    U: float


def paired_delta_ci(
    state: SearchState,
    p: str,
    q: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> PairCI | None:
    common = state.common_samples(p, q)
    n = len(common)
    if n == 0:
        return None
    d = np.asarray(
        [state.observed[(p, s)] - state.observed[(q, s)] for s in common],
        dtype=float,
    )
    delta = float(d.mean())
    if n == 1 or np.all(d == d[0]):
        return PairCI(p, q, n, delta, delta, delta)
    boots = []
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boots = d[idx].mean(axis=1)
    L = float(np.quantile(boots, 0.025))
    U = float(np.quantile(boots, 0.975))
    return PairCI(p, q, n, delta, L, U)


def confidence_eliminate(
    state: SearchState,
    candidates: list[str],
    *,
    epsilon: float,
    n_min_common: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[tuple[str, str, PairCI]]:
    """Return list of (eliminated_q, dominator_p, ci) under the v1 rule.

    q is eliminated by p if cost_pop(p)<=cost_pop(q) and L_pq > epsilon,
    with |S_pq| >= n_min_common. Uses only observed state.
    """
    from policies.common import finite

    costs = {}
    for pid in candidates:
        c = state.cost_pop(pid)
        if not finite(c):
            c = state.cost_all(pid)
        costs[pid] = float(c) if finite(c) else float("inf")

    eliminations: list[tuple[str, str, PairCI]] = []
    eliminated: set[str] = set()
    # Deterministic candidate order for reproducibility
    ordered = sorted(candidates)
    for q in ordered:
        if q in eliminated:
            continue
        for p in ordered:
            if p == q or p in eliminated:
                continue
            if costs[p] > costs[q]:
                continue
            ci = paired_delta_ci(state, p, q, rng, n_bootstrap)
            if ci is None or ci.n_common < n_min_common:
                continue
            if ci.L > epsilon:
                eliminations.append((q, p, ci))
                eliminated.add(q)
                break
    return eliminations
