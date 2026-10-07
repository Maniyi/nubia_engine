"""Agent-compatible direct neural-policy and MCTS players."""

from __future__ import annotations

import math

import numpy as np

from nubia_ai.agent import available_actions_for
from nubia_engine import Action, Empire, GameState
from nubia_training import action_to_index
from nubia_training.neural.evaluator import PositionEvaluator
from nubia_training.neural.mcts import MCTSConfig, MCTSSearchResult, search_state


class NeuralPolicyAgent:
    """Select directly from evaluator priors, without tree search."""

    def __init__(
        self,
        empire: Empire,
        evaluator: PositionEvaluator,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        name: str = "NeuralPolicyAgent",
    ) -> None:
        if not isinstance(empire, Empire):
            raise TypeError("empire must be an Empire")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be numeric")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if temperature > 0.0 and seed is None:
            raise ValueError("positive-temperature selection requires a seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.empire = empire
        self.evaluator = evaluator
        self.temperature = temperature
        self.name = name
        self._rng = np.random.default_rng(seed)

    def choose_action(self, state: GameState) -> Action:
        actions = available_actions_for(self, state)
        evaluation = self.evaluator.evaluate(state)
        if evaluation.perspective is not state.side_to_move:
            raise ValueError("evaluation perspective must equal the side to move")
        expected_indices = tuple(action_to_index(state, action) for action in actions)
        if evaluation.action_indices != expected_indices:
            raise ValueError("evaluation priors do not align with legal actions")
        if self.temperature == 0.0:
            selected = max(
                range(len(actions)),
                key=lambda index: (evaluation.priors[index], -index),
            )
        else:
            weights = np.asarray(evaluation.priors, dtype=np.float64) ** (
                1.0 / self.temperature
            )
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                raise ValueError("temperature transform produced invalid prior weights")
            selected = int(self._rng.choice(len(actions), p=weights / weights.sum()))
        return actions[selected]


class MCTSAgent:
    """Run an independent deterministic PUCT search for every move."""

    def __init__(
        self,
        empire: Empire,
        evaluator: PositionEvaluator,
        config: MCTSConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(empire, Empire):
            raise TypeError("empire must be an Empire")
        self.empire = empire
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.name = (
            f"MCTSAgent(simulations={self.config.simulations})"
            if name is None
            else name
        )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        self._last_search_result: MCTSSearchResult | None = None

    @property
    def last_search_result(self) -> MCTSSearchResult | None:
        return self._last_search_result

    def choose_action(self, state: GameState) -> Action:
        available_actions_for(self, state)
        result = search_state(state, self.evaluator, self.config)
        self._last_search_result = result
        return result.action
