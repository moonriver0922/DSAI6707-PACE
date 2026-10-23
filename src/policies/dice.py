from __future__ import annotations

from policies.base import BudgetPolicy
from policies.common import select_best_pair_queries
from state import SearchState


class DICEPolicy(BudgetPolicy):
    name = "dice"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        return select_best_pair_queries(state, remaining_calls, use_impact=True, stop_r=None)
