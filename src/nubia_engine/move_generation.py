"""Deterministic generation of ordinary NUBIA actions."""

from __future__ import annotations

from collections.abc import Iterable

from nubia_engine.actions import Action
from nubia_engine.coordinates import ALL_SQUARES, BOARD_SIZE, Square
from nubia_engine.enums import ActionKind, Empire, PieceType, RowKind
from nubia_engine.models import GameState, Piece

_ORTHOGONAL: tuple[tuple[int, int], ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))
_DIAGONAL: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ALL_DIRECTIONS = _ORTHOGONAL + _DIAGONAL
_WAR_CHIEF_OFFSETS: tuple[tuple[int, int], ...] = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)
_ACTION_KIND_ORDER = {
    ActionKind.MOVE: 0,
    ActionKind.CAPTURE: 1,
    ActionKind.SWITCH: 2,
}


def _square(row: int, column: int) -> Square | None:
    if 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE:
        return Square(row, column)
    return None


def _action_for_destination(
    state: GameState,
    piece: Piece,
    source: Square,
    destination: Square,
    *,
    move_kind: ActionKind = ActionKind.MOVE,
    can_capture: bool = True,
) -> Action | None:
    target = state.piece_at(destination)
    if target is None:
        return Action(piece.id, source, destination, move_kind)
    if (
        can_capture
        and target.current_empire is not piece.current_empire
        and target.piece_type is not PieceType.IMPERION
    ):
        return Action(
            piece.id,
            source,
            destination,
            ActionKind.CAPTURE,
            target.id,
        )
    return None


def _ray_actions(
    state: GameState,
    piece: Piece,
    source: Square,
    directions: Iterable[tuple[int, int]],
) -> list[Action]:
    actions: list[Action] = []
    for row_delta, column_delta in directions:
        distance = 1
        while True:
            destination = _square(
                source.row + row_delta * distance,
                source.column + column_delta * distance,
            )
            if destination is None:
                break
            action = _action_for_destination(state, piece, source, destination)
            if action is not None:
                actions.append(action)
            if state.piece_at(destination) is not None:
                break
            distance += 1
    return actions


def _queen_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    return _ray_actions(state, piece, source, _ALL_DIRECTIONS)


def _advisor_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    return _ray_actions(state, piece, source, _ORTHOGONAL)


def _high_chief_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    actions = _ray_actions(state, piece, source, _DIAGONAL)
    for column_delta in (-1, 1):
        destination = _square(source.row, source.column + column_delta)
        if destination is not None and state.piece_at(destination) is None:
            actions.append(Action(piece.id, source, destination, ActionKind.SWITCH))
    return actions


def _war_chief_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    actions: list[Action] = []
    for row_delta, column_delta in _WAR_CHIEF_OFFSETS:
        destination = _square(source.row + row_delta, source.column + column_delta)
        if destination is not None:
            action = _action_for_destination(state, piece, source, destination)
            if action is not None:
                actions.append(action)
    return actions


def _mystic_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    actions: list[Action] = []
    for row_delta, column_delta in _ALL_DIRECTIONS:
        adjacent = _square(source.row + row_delta, source.column + column_delta)
        if adjacent is None:
            continue
        if state.piece_at(adjacent) is not None:
            continue
        actions.append(Action(piece.id, source, adjacent, ActionKind.MOVE))
        destination = _square(
            source.row + 2 * row_delta,
            source.column + 2 * column_delta,
        )
        if destination is not None:
            action = _action_for_destination(state, piece, source, destination)
            if action is not None:
                actions.append(action)
    return actions


def _peasant_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    actions: list[Action] = []
    forward = -1 if piece.current_empire is Empire.A else 1
    for row_delta, column_delta in ((forward, 0), (0, -1), (0, 1)):
        destination = _square(source.row + row_delta, source.column + column_delta)
        if destination is not None and state.piece_at(destination) is None:
            actions.append(Action(piece.id, source, destination, ActionKind.MOVE))
    for column_delta in (-1, 1):
        destination = _square(source.row + forward, source.column + column_delta)
        if destination is None:
            continue
        target = state.piece_at(destination)
        if (
            target is not None
            and target.current_empire is not piece.current_empire
            and target.piece_type is not PieceType.IMPERION
        ):
            actions.append(
                Action(
                    piece.id,
                    source,
                    destination,
                    ActionKind.CAPTURE,
                    target.id,
                )
            )
    if source.half is piece.current_empire and source.row_kind in (
        RowKind.TOWN_CENTER,
        RowKind.OUTSKIRTS,
    ):
        for column_delta in (-2, 2):
            between = _square(source.row, source.column + column_delta // 2)
            destination = _square(source.row, source.column + column_delta)
            if (
                between is not None
                and destination is not None
                and state.piece_at(between) is None
                and state.piece_at(destination) is None
            ):
                actions.append(Action(piece.id, source, destination, ActionKind.MOVE))
    return actions


def _imperion_actions(state: GameState, piece: Piece, source: Square) -> list[Action]:
    palace_row = 9 if piece.current_empire is Empire.A else 0
    if source.row != palace_row:
        return []
    return [
        Action(piece.id, source, Square(palace_row, column), ActionKind.MOVE)
        for column in range(BOARD_SIZE)
        if column != source.column
        and state.piece_at(Square(palace_row, column)) is None
    ]


def _actions_for_piece(state: GameState, piece: Piece, source: Square) -> list[Action]:
    generators = {
        PieceType.QUEEN: _queen_actions,
        PieceType.SOUTH_AFRICAN_ADVISOR: _advisor_actions,
        PieceType.EAST_AFRICAN_HIGH_CHIEF: _high_chief_actions,
        PieceType.NORTH_CENTRAL_WAR_CHIEF: _war_chief_actions,
        PieceType.WEST_AFRICAN_MYSTIC: _mystic_actions,
        PieceType.PEASANT: _peasant_actions,
        PieceType.IMPERION: _imperion_actions,
    }
    return generators[piece.piece_type](state, piece, source)


def _sort_key(action: Action) -> tuple[int, int, int]:
    return (
        action.source.row * BOARD_SIZE + action.source.column,
        action.destination.row * BOARD_SIZE + action.destination.column,
        _ACTION_KIND_ORDER[action.kind],
    )


def legal_actions_from(state: GameState, source: Square) -> tuple[Action, ...]:
    """Return ordinary actions for the active piece at ``source`` in stable order."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(source, Square):
        raise TypeError("source must be a Square")
    piece = state.piece_at(source)
    if piece is None or piece.current_empire is not state.side_to_move:
        return ()
    return tuple(sorted(_actions_for_piece(state, piece, source), key=_sort_key))


def legal_actions(state: GameState) -> tuple[Action, ...]:
    """Return every ordinary action for the side to move in stable board order."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    actions = [
        action for source in ALL_SQUARES for action in legal_actions_from(state, source)
    ]
    return tuple(actions)
