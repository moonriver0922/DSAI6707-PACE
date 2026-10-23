"""Shared contender construction, paired risk, decision impact, pair queries."""

from __future__ import annotations

import math

import numpy as np

from metrics import paired_bootstrap_reversal_rate
from pareto import nondominated_ids, normalize_cost, utility
from state import SearchState, TAG_PAIR


def finite(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def contender_ids(state: SearchState) -> list[str]:
    acc, cost = state.snapshot_estimates(use_pop=True)
    ids = [p for p in state.prompt_ids if finite(acc[p]) and finite(cost[p])]
    if not ids:
        return list(state.prompt_ids)[: state.n_contenders]
    nd = nondominated_ids(ids, acc, cost)
    rest = [p for p in ids if p not in nd]
    # Neighbours of the frontier in normalized (a,c) space.
    if rest and nd:
        def dist(p: str) -> float:
            return min(
                (acc[p] - acc[q]) ** 2
                + (normalize_cost(cost[p], state.cost_norm_min, state.cost_norm_max)
                   - normalize_cost(cost[q], state.cost_norm_min, state.cost_norm_max)) ** 2
                for q in nd
            )
        rest = sorted(rest, key=dist)
    ordered = nd + rest
    return ordered[: state.n_contenders]


def neighboring_pairs(sorted_ids: list[str], max_pairs: int) -> list[tuple[str, str]]:
    n = len(sorted_ids)
    pairs: list[tuple[str, str]] = []
    for dist in range(1, n):
        for i in range(n - dist):
            a, b = sorted_ids[i], sorted_ids[i + dist]
            pair = (a, b) if a < b else (b, a)
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def candidate_pairs(state: SearchState) -> list[tuple[str, str]]:
    ids = contender_ids(state)
    acc, cost = state.snapshot_estimates(use_pop=True)
    ids = sorted(ids, key=lambda p: (cost.get(p, 0.0), p))
    return neighboring_pairs(ids, state.max_pairs)


def pair_arrays(state: SearchState, p: str, q: str) -> tuple[np.ndarray, np.ndarray]:
    common = state.common_samples(p, q)
    z_p = np.asarray([state.observed[(p, s)] for s in common], dtype=float)
    z_q = np.asarray([state.observed[(q, s)] for s in common], dtype=float)
    return z_p, z_q


def cost_norm_of(state: SearchState, pid: str) -> float:
    c = state.cost_all(pid)
    if not finite(c):
        c = state.cost_pop(pid)
    if not finite(c):
        return 0.5
    return normalize_cost(c, state.cost_norm_min, state.cost_norm_max)


def reversal_rate(state: SearchState, p: str, q: str, rng: np.random.Generator) -> float:
    z_p, z_q = pair_arrays(state, p, q)
    return paired_bootstrap_reversal_rate(
        z_p,
        z_q,
        cost_norm_of(state, p),
        cost_norm_of(state, q),
        rng,
        state.n_bootstrap,
        p,
        q,
        lam=0.5,
    )


def utilities_all(state: SearchState, lam: float) -> dict[str, float]:
    out = {}
    for pid in state.prompt_ids:
        a = state.accuracy_all(pid)
        c = state.cost_all(pid)
        if not finite(a) or not finite(c):
            out[pid] = float("-inf")
            continue
        out[pid] = utility(a, normalize_cost(c, state.cost_norm_min, state.cost_norm_max), lam)
    return out


def winner_of(utils: dict[str, float]) -> str:
    return sorted(utils, key=lambda p: (-utils[p], p))[0]


def decision_impact(state: SearchState, p: str, q: str) -> float:
    imps = []
    for lam in state.lambdas:
        u = utilities_all(state, lam)
        if not finite(u[p]) or not finite(u[q]) or u[p] == float("-inf"):
            imps.append(0.0)
            continue
        if u[p] == u[q]:
            imps.append(0.0)
            continue
        w = winner_of(u)
        u2 = dict(u)
        u2[p], u2[q] = u[q], u[p]
        w2 = winner_of(u2)
        if w2 == w:
            imps.append(0.0)
        else:
            imps.append(abs(u[w] - u[w2]))
    return float(np.mean(imps)) if imps else 0.0


def expected_new_calls(state: SearchState, p: str, q: str, n_samples: int) -> int:
    unused = [
        s
        for s in state.sample_ids
        if (p, s) not in state.observed and (q, s) not in state.observed
    ]
    take = min(n_samples, len(unused))
    return 2 * take


def pair_queries(
    state: SearchState, p: str, q: str, remaining_calls: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    unused = [
        s
        for s in state.sample_ids
        if (p, s) not in state.observed and (q, s) not in state.observed
    ]
    if not unused or remaining_calls < 2:
        return []
    rng.shuffle(unused)
    n_want = min(state.pair_batch_samples, len(unused), remaining_calls // 2)
    if n_want <= 0:
        return []
    queries: list[tuple[str, str]] = []
    for s in unused[:n_want]:
        queries.append((p, s))
        queries.append((q, s))
    return queries


def score_pair_from_parts(r: float, impact: float, delta_calls: int) -> float:
    return float(r * impact / max(1, delta_calls))


def select_best_pair_queries(
    state: SearchState,
    remaining_calls: int,
    use_impact: bool,
    stop_r: float | None,
) -> list[tuple[str, str]]:
    rng = state.rng()
    pairs = candidate_pairs(state)
    if not pairs or remaining_calls < 2:
        return []
    scored = []
    for p, q in pairs:
        r = reversal_rate(state, p, q, rng)
        if stop_r is not None and r < stop_r:
            continue
        impact = decision_impact(state, p, q) if use_impact else 1.0
        delta = expected_new_calls(state, p, q, state.pair_batch_samples)
        s = score_pair_from_parts(r, impact, delta)
        n_p, n_q = state.n_obs(p), state.n_obs(q)
        starve = 1.0 / max(1, min(n_p, n_q))
        scored.append((s, starve, p, q))
    if not scored:
        return []
    max_s = max(t[0] for t in scored)
    if max_s <= 0:
        scored.sort(key=lambda t: (-t[1], t[2], t[3]))
    else:
        scored.sort(key=lambda t: (-t[0], t[2], t[3]))
    _, _, p, q = scored[0]
    return pair_queries(state, p, q, remaining_calls, rng)


def uniform_column_queries(state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
    """Next search sample not fully observed for all prompts, if affordable."""
    rng = state.rng()
    order = list(state.sample_ids)
    rng.shuffle(order)
    # Deterministic scan in shuffled-once order stored via seed+calls_used.
    for s in order:
        missing = [p for p in state.prompt_ids if (p, s) not in state.observed]
        if not missing:
            continue
        if len(missing) <= remaining_calls:
            return [(p, s) for p in missing]
        return []
    return []
