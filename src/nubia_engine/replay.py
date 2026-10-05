"""Pure immutable replay of already-constructed NUBIA actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nubia_engine.actions import Action, IllegalActionError
from nubia_engine.models import GameState
from nubia_engine.transitions import apply_action


class ReplayError(ValueError):
    """Raised when an action sequence cannot be replayed."""

    def __init__(self, action_index: int, action: object, cause: Exception) -> None:
        self.action_index = action_index
        self.action = action
        self.cause = cause
        super().__init__(f"replay failed at action index {action_index}: {cause}")


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """The immutable inputs and outputs of a successful replay."""

    initial_state: GameState
    actions: tuple[Action, ...]
    states: tuple[GameState, ...]
    final_state: GameState

    @property
    def applied_actions(self) -> tuple[Action, ...]:
        """Alias emphasizing that every recorded action was applied."""

        return self.actions

    @property
    def state_sequence(self) -> tuple[GameState, ...]:
        """Alias for the ordered states, including the initial state."""

        return self.states


def replay_actions(initial_state: GameState, actions: Iterable[Action]) -> ReplayResult:
    """Apply ``actions`` in order using the engine's public transition API."""

    if not isinstance(initial_state, GameState):
        raise TypeError("initial_state must be a GameState")
    if not isinstance(actions, Iterable):
        raise TypeError("actions must be iterable")
    current = initial_state
    applied: list[Action] = []
    states = [initial_state]
    for index, action in enumerate(actions):
        try:
            current = apply_action(current, action)
        except (IllegalActionError, TypeError) as error:
            raise ReplayError(index, action, error) from error
        applied.append(action)
        states.append(current)
    return ReplayResult(initial_state, tuple(applied), tuple(states), current)
