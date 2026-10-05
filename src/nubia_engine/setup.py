"""Deterministic construction of the complete NUBIA starting state."""

from __future__ import annotations

from collections.abc import Iterable

from nubia_engine.coordinates import BOARD_SIZE, Square
from nubia_engine.enums import BrainwashAvailability, Empire, PieceType
from nubia_engine.models import GameState, Piece

_STARTING_PATHS: tuple[tuple[PieceType, str, tuple[int, ...]], ...] = (
    (PieceType.IMPERION, "p", (5,)),
    (PieceType.QUEEN, "p", (6,)),
    (PieceType.SOUTH_AFRICAN_ADVISOR, "p", (2, 9)),
    (PieceType.EAST_AFRICAN_HIGH_CHIEF, "m", (2, 9)),
    (PieceType.NORTH_CENTRAL_WAR_CHIEF, "m", (4, 7)),
    (PieceType.WEST_AFRICAN_MYSTIC, "m", (3, 8)),
    (PieceType.PEASANT, "c", tuple(range(1, 11))),
)


def _starting_pieces(empire: Empire) -> Iterable[tuple[Square, Piece]]:
    for piece_type, row_symbol, paths in _STARTING_PATHS:
        for ordinal, path in enumerate(paths, start=1):
            brainwash = (
                BrainwashAvailability.AVAILABLE
                if piece_type is PieceType.WEST_AFRICAN_MYSTIC
                else None
            )
            piece = Piece(
                id=f"{empire.value}-{piece_type.value}-{ordinal}",
                piece_type=piece_type,
                original_empire=empire,
                current_empire=empire,
                brainwash=brainwash,
            )
            yield Square.from_notation(f"{empire.value}:{row_symbol}{path}"), piece


def create_initial_state(first_player: Empire) -> GameState:
    """Create the complete deterministic initial position."""

    if not isinstance(first_player, Empire):
        raise TypeError("first_player must be an Empire")

    board: list[Piece | None] = [None] * (BOARD_SIZE * BOARD_SIZE)
    for empire in Empire:
        for square, piece in _starting_pieces(empire):
            index = square.row * BOARD_SIZE + square.column
            if board[index] is not None:
                raise RuntimeError(f"starting-piece collision at {square}")
            board[index] = piece

    return GameState(board=tuple(board), side_to_move=first_player)
