"""Immutable domain models for pieces and game state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from nubia_engine.coordinates import ALL_SQUARES, Square
from nubia_engine.enums import BrainwashAvailability, Empire, PieceType


@dataclass(frozen=True, slots=True)
class Piece:
    """A stable, immutable piece identity and its rules-relevant attributes."""

    id: str
    piece_type: PieceType
    original_empire: Empire
    current_empire: Empire
    brainwash: BrainwashAvailability | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("piece id must be a non-empty string")
        if not isinstance(self.piece_type, PieceType):
            raise TypeError("piece_type must be a PieceType")
        if not isinstance(self.original_empire, Empire):
            raise TypeError("original_empire must be an Empire")
        if not isinstance(self.current_empire, Empire):
            raise TypeError("current_empire must be an Empire")
        if self.brainwash is not None and not isinstance(
            self.brainwash, BrainwashAvailability
        ):
            raise TypeError("brainwash must be a BrainwashAvailability or None")
        is_mystic = self.piece_type is PieceType.WEST_AFRICAN_MYSTIC
        if is_mystic and self.brainwash is None:
            raise ValueError("Mystics must have a Brainwash availability state")
        if not is_mystic and self.brainwash is not None:
            raise ValueError("only Mystics may have a Brainwash availability state")

    @property
    def brainwash_available(self) -> bool:
        """Whether this Mystic retains its power; always false for other pieces."""

        return self.brainwash is BrainwashAvailability.AVAILABLE


Board: TypeAlias = tuple[Piece | None, ...]


@dataclass(frozen=True, slots=True)
class GameState:
    """The immutable Milestone 1 state of a NUBIA game."""

    board: Board
    side_to_move: Empire
    quiet_ply_count: int = 0
    ply_number: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.board, tuple):
            raise TypeError("board must be an immutable tuple")
        if len(self.board) != len(ALL_SQUARES):
            raise ValueError("board must contain exactly 100 squares")
        if any(
            piece is not None and not isinstance(piece, Piece) for piece in self.board
        ):
            raise TypeError("board entries must be Piece instances or None")
        if not isinstance(self.side_to_move, Empire):
            raise TypeError("side_to_move must be an Empire")
        piece_ids = [piece.id for piece in self.board if piece is not None]
        if len(piece_ids) != len(set(piece_ids)):
            raise ValueError("piece identifiers must be unique")
        if (
            isinstance(self.quiet_ply_count, bool)
            or not isinstance(self.quiet_ply_count, int)
            or self.quiet_ply_count < 0
        ):
            raise ValueError("quiet_ply_count must be a non-negative integer")
        if (
            isinstance(self.ply_number, bool)
            or not isinstance(self.ply_number, int)
            or self.ply_number < 0
        ):
            raise ValueError("ply_number must be a non-negative integer")

    def piece_at(self, square: Square) -> Piece | None:
        """Return the piece at ``square``, if any."""

        return self.board[square.row * 10 + square.column]

    @property
    def pieces(self) -> tuple[Piece, ...]:
        """All pieces in deterministic fixed-board order."""

        return tuple(piece for piece in self.board if piece is not None)
