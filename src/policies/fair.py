"""Post-hoc fair scaffolds: same early-stop + Uniform fallback as Racing.

Only the pair-scoring rule differs:
  - fair_racing / PairedRacing: risk-only score + stop_r + uniform fallback
  - fair_dice: impact-aware score + SAME stop_r + SAME uniform fallback

These policies are for diagnostic ablation only; they do not replace the
pre-registered Gate-1 NO-GO comparison.
"""

from __future__ import annotations

from policies.base import BudgetPolicy
from policies.common import select_best_pair_queries, uniform_column_queries
from state import SearchState


class FairRacingPolicy(BudgetPolicy):
    """Identical to PairedRacingPolicy (explicit name for post-hoc tables)."""

    name = "fair_racing"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        q = select_best_pair_queries(
            state, remaining_calls, use_impact=False, stop_r=state.racing_stop_r
        )
        if q:
            return q
        return uniform_column_queries(state, remaining_calls)


class FairDICEPolicy(BudgetPolicy):
    """DICE scoring with Racing's early-stop and Uniform fallback."""

    name = "fair_dice"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        q = select_best_pair_queries(
            state, remaining_calls, use_impact=True, stop_r=state.racing_stop_r
        )
        if q:
            return q
        return uniform_column_queries(state, remaining_calls)


class FairRiskOnlyPolicy(BudgetPolicy):
    """Risk-only score + same scaffold (should ≈ fair_racing; sanity check)."""

    name = "fair_risk_only"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        q = select_best_pair_queries(
            state, remaining_calls, use_impact=False, stop_r=state.racing_stop_r
        )
        if q:
            return q
        return uniform_column_queries(state, remaining_calls)
