from __future__ import annotations

from policies.base import BudgetPolicy
from policies.common import uniform_column_queries
from state import SearchState


class UniformPolicy(BudgetPolicy):
    name = "uniform"

    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        return uniform_column_queries(state, remaining_calls)
