"""Agent wrapper for bounded iterative-deepening minimax search."""

from __future__ import annotations

import time

from nubia_ai.evaluation import DEFAULT_WEIGHTS, EvaluationWeights
from nubia_ai.iterative import (
    Clock,
    IterativeSearchConfig,
    IterativeSearchResult,
    iterative_search_state,
)
from nubia_engine import Action, Empire, GameState


class IterativeMinimaxAgent:
    """A bounded minimax policy returning its last completed iteration.

    Node-only and unlimited searches are deterministic. With a wall-clock
    budget, completed depth can vary with machine speed and system load.
    """

    def __init__(
        self,
        empire: Empire,
        config: IterativeSearchConfig,
        weights: EvaluationWeights = DEFAULT_WEIGHTS,
        *,
        clock: Clock = time.monotonic,
        name: str | None = None,
    ) -> None:
        if not isinstance(empire, Empire):
            raise TypeError("empire must be an Empire")
        if not isinstance(config, IterativeSearchConfig):
            raise TypeError("config must be an IterativeSearchConfig")
        if not isinstance(weights, EvaluationWeights):
            raise TypeError("weights must be EvaluationWeights")
        if not callable(clock):
            raise TypeError("clock must be callable")
        resolved_name = self._default_name(config) if name is None else name
        if not isinstance(resolved_name, str) or not resolved_name.strip():
            raise ValueError("name must be a non-empty string")
        self.empire = empire
        self.config = config
        self.weights = weights
        self.name = resolved_name
        self._clock = clock

    @staticmethod
    def _default_name(config: IterativeSearchConfig) -> str:
        budgets: list[str] = []
        if config.time_limit_seconds is not None:
            budgets.append(f"time={config.time_limit_seconds:g}s")
        if config.node_limit is not None:
            budgets.append(f"nodes={config.node_limit}")
        suffix = f",{','.join(budgets)}" if budgets else ""
        return f"IterativeMinimaxAgent(max_depth={config.max_depth}{suffix})"

    def search(self, state: GameState) -> IterativeSearchResult:
        """Return the bounded decision and complete search telemetry."""

        return iterative_search_state(
            state,
            self.empire,
            self.config,
            self.weights,
            clock=self._clock,
        )

    def choose_action(self, state: GameState) -> Action:
        """Return the deepest completed decision or deterministic fallback."""

        return self.search(state).action
