from __future__ import annotations

from abc import ABC, abstractmethod

from state import SearchState


class BudgetPolicy(ABC):
    name: str = "base"

    @abstractmethod
    def select_queries(self, state: SearchState, remaining_calls: int) -> list[tuple[str, str]]:
        """Only observed, non-test state; each returned uncached query costs 1."""
