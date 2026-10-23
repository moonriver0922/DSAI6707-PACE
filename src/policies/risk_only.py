from __future__ import annotations

from policies.base import BudgetPolicy
from policies.common import select_best_pair_queries
from state import SearchState


class RiskOnlyPolicy(BudgetPolicy):
    """Uncertainty-only paired allocation; impact is fixed at 1 (H2 ablation)."""

    name = "risk_only"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        return select_best_pair_queries(state, remaining_calls, use_impact=False, stop_r=None)
