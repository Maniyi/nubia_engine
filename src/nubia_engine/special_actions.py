"""Deterministic generation of GBESELE and Mystic special actions."""

from __future__ import annotations

from nubia_engine.actions import Action, ActionTarget
from nubia_engine.coordinates import BOARD_SIZE, Square
from nubia_engine.defense import is_defended_for_gbesele
from nubia_engine.enums import ActionKind, PieceType
from nubia_engine.models import GameState, Piece

_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _square(row: int, column: int) -> Square | None:
    if 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE:
        return Square(row, column)
    return None


def _gbesele_action(state: GameState, imperion: Piece, source: Square) -> Action | None:
    targets: list[ActionTarget] = []
    for row_delta, column_delta in _DIRECTIONS:
        distance = 1
        while True:
            square = _square(
                source.row + row_delta * distance,
                source.column + column_delta * distance,
            )
            if square is None:
                break
            target = state.piece_at(square)
            if target is None:
                distance += 1
                continue
            if (
                target.current_empire is not imperion.current_empire
                and square.half is imperion.current_empire
                and target.piece_type is not PieceType.IMPERION
                and not is_defended_for_gbesele(state, square, target.current_empire)
            ):
                targets.append(ActionTarget(square, target.id))
            break

    if not targets:
        return None
    ordered_targets = tuple(
        sorted(
            targets,
            key=lambda target: target.square.row * BOARD_SIZE + target.square.column,
        )
    )
    return Action(
        imperion.id,
        source,
        source,
        ActionKind.GBESELE,
        targets=ordered_targets,
    )


def _brainwash_actions(state: GameState, mystic: Piece, source: Square) -> list[Action]:
    if not mystic.brainwash_available:
        return []
    actions: list[Action] = []
    for row_delta, column_delta in _DIRECTIONS:
        midpoint = _square(source.row + row_delta, source.column + column_delta)
        target_square = _square(
            source.row + 2 * row_delta,
            source.column + 2 * column_delta,
        )
        if (
            midpoint is None
            or target_square is None
            or state.piece_at(midpoint) is not None
        ):
            continue
        target = state.piece_at(target_square)
        if (
            target is None
            or target.current_empire is mystic.current_empire
            or target.piece_type is PieceType.IMPERION
        ):
            continue
        kind = (
            ActionKind.REBRAINWASH
            if target.current_empire is not target.original_empire
            and mystic.current_empire is target.original_empire
            else ActionKind.BRAINWASH
        )
        actions.append(
            Action(
                mystic.id,
                source,
                target_square,
                kind,
                targets=(ActionTarget(target_square, target.id),),
            )
        )
    return actions


def special_actions_for_piece(
    state: GameState, piece: Piece, source: Square
) -> list[Action]:
    """Generate the special actions belonging to one active piece."""

    if piece.piece_type is PieceType.IMPERION:
        action = _gbesele_action(state, piece, source)
        return [] if action is None else [action]
    if piece.piece_type is PieceType.WEST_AFRICAN_MYSTIC:
        return _brainwash_actions(state, piece, source)
    return []
