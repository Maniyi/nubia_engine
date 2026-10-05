"""Perft-style verification using only public generation and transition APIs."""

from __future__ import annotations

from dataclasses import dataclass

from nubia_engine.actions import Action
from nubia_engine.models import GameState
from nubia_engine.move_generation import legal_actions
from nubia_engine.notation import format_action
from nubia_engine.transitions import apply_action


@dataclass(frozen=True, slots=True)
class PerftEntry:
    """One top-level action and its descendant leaf count."""

    action: Action
    label: str
    count: int

    @property
    def descendant_count(self) -> int:
        """Return the descendant leaf count for this top-level action."""

        return self.count


def _validate(state: GameState, depth: int) -> None:
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth < 0:
        raise ValueError("depth must be non-negative")


def perft(state: GameState, depth: int) -> int:
    """Count move-tree leaves at exactly ``depth`` plies."""

    _validate(state, depth)
    if depth == 0:
        return 1
    return sum(
        perft(apply_action(state, action), depth - 1) for action in legal_actions(state)
    )


def perft_divide(state: GameState, depth: int) -> tuple[PerftEntry, ...]:
    """Return legal-order top-level contributions for a positive depth."""

    _validate(state, depth)
    if depth == 0:
        return ()
    return tuple(
        PerftEntry(
            action,
            format_action(state, action),
            perft(apply_action(state, action), depth - 1),
        )
        for action in legal_actions(state)
    )
