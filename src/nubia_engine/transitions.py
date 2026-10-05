"""Validated immutable state transitions for NUBIA actions."""

from __future__ import annotations

from dataclasses import replace

from nubia_engine.actions import Action, GameAlreadyOverError, IllegalActionError
from nubia_engine.adjudication import adjudicate
from nubia_engine.coordinates import BOARD_SIZE
from nubia_engine.enums import ActionKind, BrainwashAvailability, Empire
from nubia_engine.models import GameState
from nubia_engine.move_generation import legal_actions_from
from nubia_engine.position import position_key, record_position


def apply_action(state: GameState, action: Action) -> GameState:
    """Apply a currently legal action and return its successor state."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if state.result is not None:
        raise GameAlreadyOverError("the game is already over")
    if not isinstance(action, Action):
        raise IllegalActionError("action must be an Action")
    if action not in legal_actions_from(state, action.source):
        raise IllegalActionError("action is not legal in the supplied state")

    source_index = action.source.row * BOARD_SIZE + action.source.column
    moving_piece = state.board[source_index]
    if moving_piece is None:  # Defensive: legality already establishes this.
        raise IllegalActionError("action source is empty")

    board = list(state.board)
    if action.kind is ActionKind.GBESELE:
        for target in action.targets:
            board[target.square.row * BOARD_SIZE + target.square.column] = None
    elif action.kind in (ActionKind.BRAINWASH, ActionKind.REBRAINWASH):
        target = action.targets[0]
        target_index = target.square.row * BOARD_SIZE + target.square.column
        target_piece = board[target_index]
        if target_piece is None:  # Defensive: legality already establishes this.
            raise IllegalActionError("special-action target is empty")
        restored_empire = (
            target_piece.original_empire
            if action.kind is ActionKind.REBRAINWASH
            else moving_piece.current_empire
        )
        board[target_index] = replace(target_piece, current_empire=restored_empire)
        board[source_index] = replace(
            moving_piece, brainwash=BrainwashAvailability.SPENT
        )
    else:
        destination_index = (
            action.destination.row * BOARD_SIZE + action.destination.column
        )
        board[source_index] = None
        board[destination_index] = moving_piece
    next_side = Empire.B if state.side_to_move is Empire.A else Empire.A
    quiet_ply_count = (
        0
        if action.kind
        in (
            ActionKind.CAPTURE,
            ActionKind.GBESELE,
            ActionKind.BRAINWASH,
            ActionKind.REBRAINWASH,
        )
        else state.quiet_ply_count + 1
    )
    successor = GameState(
        board=tuple(board),
        side_to_move=next_side,
        quiet_ply_count=quiet_ply_count,
        ply_number=state.ply_number + 1,
    )
    prior_history = state.position_history or (position_key(state),)
    history = record_position(prior_history, position_key(successor))
    successor = replace(successor, position_history=history)
    return replace(successor, result=adjudicate(successor))
