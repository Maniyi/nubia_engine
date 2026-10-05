"""Agent wrapper for deterministic fixed-depth minimax search."""

from __future__ import annotations

from nubia_ai.evaluation import DEFAULT_WEIGHTS, EvaluationWeights
from nubia_ai.search import SearchConfig, SearchResult, search_state
from nubia_engine import Action, Empire, GameState


class MinimaxAgent:
    """A deterministic minimax policy with optional alpha-beta pruning."""

    def __init__(
        self,
        empire: Empire,
        config: SearchConfig,
        weights: EvaluationWeights = DEFAULT_WEIGHTS,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(empire, Empire):
            raise TypeError("empire must be an Empire")
        if not isinstance(config, SearchConfig):
            raise TypeError("config must be a SearchConfig")
        if not isinstance(weights, EvaluationWeights):
            raise TypeError("weights must be EvaluationWeights")
        resolved_name = f"MinimaxAgent(depth={config.depth})" if name is None else name
        if not isinstance(resolved_name, str) or not resolved_name.strip():
            raise ValueError("name must be a non-empty string")
        self.empire = empire
        self.config = config
        self.weights = weights
        self.name = resolved_name

    def search(self, state: GameState) -> SearchResult:
        """Return full deterministic diagnostics for one state."""

        return search_state(state, self.empire, self.config, self.weights)

    def choose_action(self, state: GameState) -> Action:
        """Choose the first action of the principal variation."""

        return self.search(state).action
