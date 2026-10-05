"""Identity-independent repetition position keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nubia_engine.enums import BrainwashAvailability, Empire, PieceType

if TYPE_CHECKING:
    from nubia_engine.models import GameState


@dataclass(frozen=True, slots=True, order=True)
class PositionPiece:
    """Rules-relevant state for one occupied canonical board square."""

    square_index: int
    piece_type: PieceType
    current_empire: Empire
    original_empire: Empire
    brainwash: BrainwashAvailability | None


@dataclass(frozen=True, slots=True)
class PositionKey:
    """A hashable repetition identity, independent of piece identifiers."""

    side_to_move: Empire
    occupied: tuple[PositionPiece, ...]


def position_key(state: GameState) -> PositionKey:
    """Return the canonical rules-relevant repetition key for ``state``."""

    from nubia_engine.models import GameState

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    occupied = tuple(
        PositionPiece(
            index,
            piece.piece_type,
            piece.current_empire,
            piece.original_empire,
            piece.brainwash,
        )
        for index, piece in enumerate(state.board)
        if piece is not None
    )
    return PositionKey(state.side_to_move, occupied)


def record_position(
    history: tuple[PositionKey, ...], key: PositionKey
) -> tuple[PositionKey, ...]:
    """Return a new history with one resulting position appended."""

    if not isinstance(history, tuple) or any(
        not isinstance(item, PositionKey) for item in history
    ):
        raise TypeError("history must be a tuple of PositionKey values")
    if not isinstance(key, PositionKey):
        raise TypeError("key must be a PositionKey")
    return (*history, key)
