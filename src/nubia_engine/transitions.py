"""Validated immutable state transitions for ordinary NUBIA actions."""

from __future__ import annotations

from nubia_engine.actions import Action, IllegalActionError
from nubia_engine.coordinates import BOARD_SIZE
from nubia_engine.enums import ActionKind, Empire
from nubia_engine.models import GameState
from nubia_engine.move_generation import legal_actions_from


def apply_action(state: GameState, action: Action) -> GameState:
    """Apply a currently legal ordinary action and return its successor state."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(action, Action):
        raise IllegalActionError("action must be an Action")
    if action not in legal_actions_from(state, action.source):
        raise IllegalActionError("action is not legal in the supplied state")

    source_index = action.source.row * BOARD_SIZE + action.source.column
    destination_index = action.destination.row * BOARD_SIZE + action.destination.column
    moving_piece = state.board[source_index]
    if moving_piece is None:  # Defensive: legality already establishes this.
        raise IllegalActionError("action source is empty")

    board = list(state.board)
    board[source_index] = None
    board[destination_index] = moving_piece
    next_side = Empire.B if state.side_to_move is Empire.A else Empire.A
    quiet_ply_count = (
        0 if action.kind is ActionKind.CAPTURE else state.quiet_ply_count + 1
    )
    return GameState(
        board=tuple(board),
        side_to_move=next_side,
        quiet_ply_count=quiet_ply_count,
        ply_number=state.ply_number + 1,
    )
