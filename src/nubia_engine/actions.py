"""Immutable ordinary actions and action-validation errors."""

from __future__ import annotations

from dataclasses import dataclass

from nubia_engine.coordinates import Square
from nubia_engine.enums import ActionKind


class IllegalActionError(ValueError):
    """Raised when an action cannot legally be applied to a game state."""


@dataclass(frozen=True, slots=True)
class Action:
    """A deterministic description of one ordinary turn action."""

    piece_id: str
    source: Square
    destination: Square
    kind: ActionKind
    captured_piece_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.piece_id, str) or not self.piece_id.strip():
            raise ValueError("piece_id must be a non-empty string")
        if not isinstance(self.source, Square):
            raise TypeError("source must be a Square")
        if not isinstance(self.destination, Square):
            raise TypeError("destination must be a Square")
        if self.source == self.destination:
            raise ValueError("source and destination must differ")
        if not isinstance(self.kind, ActionKind):
            raise TypeError("kind must be an ActionKind")
        if self.kind is ActionKind.CAPTURE:
            if (
                not isinstance(self.captured_piece_id, str)
                or not self.captured_piece_id.strip()
            ):
                raise ValueError("CAPTURE must identify a captured piece")
        elif self.captured_piece_id is not None:
            raise ValueError("MOVE and SWITCH cannot identify a captured piece")
