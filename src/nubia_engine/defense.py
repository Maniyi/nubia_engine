"""Non-recursive ordinary-capture defence used by Imperion GBESELE."""

from __future__ import annotations

from nubia_engine.coordinates import BOARD_SIZE, Square
from nubia_engine.enums import Empire, PieceType
from nubia_engine.models import GameState, Piece


def _is_clear_ray(state: GameState, source: Square, target: Square) -> bool:
    row_distance = target.row - source.row
    column_distance = target.column - source.column
    row_step = (row_distance > 0) - (row_distance < 0)
    column_step = (column_distance > 0) - (column_distance < 0)
    distance = max(abs(row_distance), abs(column_distance))
    return all(
        state.piece_at(
            Square(source.row + row_step * step, source.column + column_step * step)
        )
        is None
        for step in range(1, distance)
    )


def _could_capture_target(
    state: GameState, piece: Piece, source: Square, target: Square
) -> bool:
    row_distance = target.row - source.row
    column_distance = target.column - source.column
    absolute_rows = abs(row_distance)
    absolute_columns = abs(column_distance)
    same_row = row_distance == 0 and column_distance != 0
    same_column = column_distance == 0 and row_distance != 0
    diagonal = absolute_rows == absolute_columns and absolute_rows != 0

    if piece.piece_type is PieceType.QUEEN:
        return (same_row or same_column or diagonal) and _is_clear_ray(
            state, source, target
        )
    if piece.piece_type is PieceType.SOUTH_AFRICAN_ADVISOR:
        return (same_row or same_column) and _is_clear_ray(state, source, target)
    if piece.piece_type is PieceType.EAST_AFRICAN_HIGH_CHIEF:
        return diagonal and _is_clear_ray(state, source, target)
    if piece.piece_type is PieceType.NORTH_CENTRAL_WAR_CHIEF:
        return (absolute_rows, absolute_columns) in ((1, 2), (2, 1))
    if piece.piece_type is PieceType.WEST_AFRICAN_MYSTIC:
        return (
            (same_row or same_column or diagonal)
            and max(absolute_rows, absolute_columns) == 2
            and _is_clear_ray(state, source, target)
        )
    if piece.piece_type is PieceType.PEASANT:
        forward = -1 if piece.current_empire is Empire.A else 1
        return row_distance == forward and absolute_columns == 1
    return False


def is_defended_for_gbesele(
    state: GameState, target: Square, defending_empire: Empire
) -> bool:
    """Whether another ordinary capturer of ``defending_empire`` protects target.

    The target occupant is treated as the protected piece rather than a blocker.
    No complete action generation is used, so this query cannot recurse through
    GBESELE or include non-capture actions.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(target, Square):
        raise TypeError("target must be a Square")
    if not isinstance(defending_empire, Empire):
        raise TypeError("defending_empire must be an Empire")

    for index, piece in enumerate(state.board):
        if piece is None or piece.current_empire is not defending_empire:
            continue
        source = Square(index // BOARD_SIZE, index % BOARD_SIZE)
        if source == target or piece.piece_type is PieceType.IMPERION:
            continue
        if _could_capture_target(state, piece, source, target):
            return True
    return False
