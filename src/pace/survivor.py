"""Point-estimate truncation (B0) vs PACE evidence-gated bounded survival."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evolution import nsga2_select
from pace.evidence import PairCI, confidence_eliminate
from policies.common import finite
from state import SearchState


@dataclass
class ConfidenceElimination:
    eliminated: str
    by: str
    L: float
    U: float
    delta_hat: float
    n_common: int


@dataclass
class ForcedEviction:
    prompt_id: str
    reason: str = "capacity"


@dataclass
class SurviveResult:
    retained: list[str]
    active: list[str] = field(default_factory=list)
    confidence_elims: list[ConfidenceElimination] = field(default_factory=list)
    forced_evictions: list[ForcedEviction] = field(default_factory=list)


def _estimates(state: SearchState, ids: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    acc, cost = {}, {}
    for pid in ids:
        a = state.accuracy_pop(pid)
        c = state.cost_pop(pid)
        if not finite(a):
            a = state.accuracy_all(pid)
        if not finite(c):
            c = state.cost_all(pid)
        acc[pid] = 0.0 if not finite(a) else float(a)
        cost[pid] = 1e9 if not finite(c) else float(c)
    return acc, cost


def point_estimate_truncate(state: SearchState, candidates: list[str], mu: int) -> SurviveResult:
    """B0: NSGA-II rank/crowding on point estimates → keep mu."""
    if len(candidates) <= mu:
        keep = sorted(candidates)
        return SurviveResult(retained=keep, active=keep)
    acc, cost = _estimates(state, candidates)
    keep = nsga2_select(candidates, acc, cost, mu)
    return SurviveResult(retained=keep, active=keep)


def pace_bounded_survive(
    state: SearchState,
    candidates: list[str],
    *,
    mu: int,
    archive_k: int,
    epsilon: float,
    n_min_common: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    disable_confidence_gate: bool = False,
) -> SurviveResult:
    """Confidence eliminate, then capacity-truncate with logged forced evictions.

    A1 ablation: disable_confidence_gate=True skips evidence elimination (conf_elim=0).
    With archive_k=0 this collapses to NSGA truncate at μ (should match B0).
    """
    if disable_confidence_gate:
        conf: list[ConfidenceElimination] = []
        remaining = list(candidates)
    else:
        raw = confidence_eliminate(
            state,
            candidates,
            epsilon=epsilon,
            n_min_common=n_min_common,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        conf = [
            ConfidenceElimination(q, p, ci.L, ci.U, ci.delta_hat, ci.n_common)
            for q, p, ci in raw
        ]
        elim_ids = {e.eliminated for e in conf}
        remaining = [c for c in candidates if c not in elim_ids]
    cap = mu + archive_k
    forced: list[ForcedEviction] = []
    if len(remaining) > cap:
        acc, cost = _estimates(state, remaining)
        keep = nsga2_select(remaining, acc, cost, cap)
        keep_set = set(keep)
        for pid in sorted(remaining):
            if pid not in keep_set:
                forced.append(ForcedEviction(pid))
        remaining = keep
    # Active = top-mu; retained = active ∪ archive (≤ μ+K)
    if len(remaining) > mu:
        acc, cost = _estimates(state, remaining)
        active = nsga2_select(remaining, acc, cost, mu)
        retained = remaining
    else:
        active = list(remaining)
        retained = remaining
    return SurviveResult(
        retained=retained,
        active=active,
        confidence_elims=conf,
        forced_evictions=forced,
    )
