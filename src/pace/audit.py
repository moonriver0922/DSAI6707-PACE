"""Audit-only reference labels from the full search matrix (never for policy)."""

from __future__ import annotations

import numpy as np

from pareto import nondominated_ids, normalize_cost, utility
from replay import OracleMatrix


def full_matrix_estimates(oracle: OracleMatrix) -> tuple[dict[str, float], dict[str, float]]:
    acc, cost = {}, {}
    for p in oracle.prompt_ids:
        i = oracle.prompt_index[p]
        acc[p] = float(oracle.correct[i].mean())
        cost[p] = float((oracle.input_tokens[i] + oracle.output_tokens[i]).astype(float).mean())
    return acc, cost


def reference_pareto_ids(oracle: OracleMatrix) -> list[str]:
    acc, cost = full_matrix_estimates(oracle)
    ids = list(oracle.prompt_ids)
    return nondominated_ids(ids, acc, cost)


def false_elimination_rate(
    eliminated: set[str],
    reference_pareto: set[str],
) -> float:
    """Fraction of reference-Pareto candidates that were eliminated."""
    if not reference_pareto:
        return float("nan")
    hit = eliminated & reference_pareto
    return float(len(hit) / len(reference_pareto))


def reference_pareto_recall(retained: set[str], reference_pareto: set[str]) -> float:
    if not reference_pareto:
        return float("nan")
    return float(len(retained & reference_pareto) / len(reference_pareto))


def retained_set_regret(
    retained: list[str],
    oracle: OracleMatrix,
    lam: float,
    cost_min: float,
    cost_max: float,
) -> float:
    """Preference regret if we pick best retained by full-matrix U_λ vs full-pool oracle."""
    from metrics import preference_selection_regret, preference_winner

    acc, cost = full_matrix_estimates(oracle)
    if not retained:
        return float("nan")
    # Choose among retained using full matrix (audit only) — for reporting retained-set quality
    # Protocol: auditor evaluates retained set quality on full search matrix reference.
    w = preference_winner(retained, acc, cost, lam, cost_min, cost_max)
    return preference_selection_regret(w, acc, cost, lam, cost_min, cost_max)
