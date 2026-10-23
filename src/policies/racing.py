from __future__ import annotations

from policies.base import BudgetPolicy
from policies.common import select_best_pair_queries, uniform_column_queries
from state import SearchState


class PairedRacingPolicy(BudgetPolicy):
    name = "racing"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        q = select_best_pair_queries(
            state, remaining_calls, use_impact=False, stop_r=state.racing_stop_r
        )
        if q:
            return q
        return uniform_column_queries(state, remaining_calls)
