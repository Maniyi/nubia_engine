"""Deterministic one-ply heuristic agent."""

from __future__ import annotations

from dataclasses import dataclass

from nubia_ai.agent import available_actions_for
from nubia_ai.evaluation import (
    DEFAULT_WEIGHTS,
    EvaluationBreakdown,
    EvaluationWeights,
    evaluate_state,
)
from nubia_engine import Action, Empire, GameState, apply_action


@dataclass(frozen=True, slots=True)
class ScoredAction:
    """One legal action and the evaluation of its immediate successor."""

    action: Action
    evaluation: EvaluationBreakdown

    @property
    def score(self) -> int:
        return self.evaluation.total


class HeuristicAgent:
    """A deterministic baseline that looks ahead exactly one action."""

    def __init__(
        self,
        empire: Empire,
        *,
        weights: EvaluationWeights = DEFAULT_WEIGHTS,
        name: str = "HeuristicAgent",
    ) -> None:
        if not isinstance(empire, Empire):
            raise TypeError("empire must be an Empire")
        if not isinstance(weights, EvaluationWeights):
            raise TypeError("weights must be EvaluationWeights")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.empire = empire
        self.name = name
        self.weights = weights

    def rank_actions(self, state: GameState) -> tuple[ScoredAction, ...]:
        """Score actions in engine order, preserving that order for ties."""

        return tuple(
            ScoredAction(
                action,
                evaluate_state(apply_action(state, action), self.empire, self.weights),
            )
            for action in available_actions_for(self, state)
        )

    def choose_action(self, state: GameState) -> Action:
        """Choose the first engine-ordered action with the greatest score."""

        ranked = self.rank_actions(state)
        return max(ranked, key=lambda scored: scored.score).action
