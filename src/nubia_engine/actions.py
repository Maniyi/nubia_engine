"""Immutable turn actions and action-validation errors."""

from __future__ import annotations

from dataclasses import dataclass

from nubia_engine.coordinates import BOARD_SIZE, Square
from nubia_engine.enums import ActionKind


class IllegalActionError(ValueError):
    """Raised when an action cannot legally be applied to a game state."""


@dataclass(frozen=True, slots=True)
class ActionTarget:
    """Stable square-and-piece identity for a special-action target."""

    square: Square
    piece_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.square, Square):
            raise TypeError("target square must be a Square")
        if not isinstance(self.piece_id, str) or not self.piece_id.strip():
            raise ValueError("target piece_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Action:
    """A deterministic description of one complete turn action.

    The first five fields retain the Milestone 2 positional API. Special actions
    use ``targets`` so both target squares and stable piece identities are part of
    the value. GBESELE uses ``destination == source`` because the Imperion stays
    put; Brainwash actions use their single target square as ``destination``.
    """

    piece_id: str
    source: Square
    destination: Square
    kind: ActionKind
    captured_piece_id: str | None = None
    targets: tuple[ActionTarget, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.piece_id, str) or not self.piece_id.strip():
            raise ValueError("piece_id must be a non-empty string")
        if not isinstance(self.source, Square):
            raise TypeError("source must be a Square")
        if not isinstance(self.destination, Square):
            raise TypeError("destination must be a Square")
        if not isinstance(self.kind, ActionKind):
            raise TypeError("kind must be an ActionKind")
        if not isinstance(self.targets, tuple) or any(
            not isinstance(target, ActionTarget) for target in self.targets
        ):
            raise TypeError("targets must be a tuple of ActionTarget values")

        special_kinds = {
            ActionKind.GBESELE,
            ActionKind.BRAINWASH,
            ActionKind.REBRAINWASH,
        }
        if self.kind not in special_kinds and self.source == self.destination:
            raise ValueError("source and destination must differ")
        if self.kind is ActionKind.CAPTURE:
            if (
                not isinstance(self.captured_piece_id, str)
                or not self.captured_piece_id.strip()
            ):
                raise ValueError("CAPTURE must identify a captured piece")
        elif self.captured_piece_id is not None:
            raise ValueError("only CAPTURE may identify a captured piece")

        if self.kind in (ActionKind.MOVE, ActionKind.CAPTURE, ActionKind.SWITCH):
            if self.targets:
                raise ValueError("ordinary actions cannot identify special targets")
            return

        if not self.targets:
            raise ValueError("special actions must identify their targets")
        squares = tuple(target.square for target in self.targets)
        piece_ids = tuple(target.piece_id for target in self.targets)
        if len(squares) != len(set(squares)) or len(piece_ids) != len(set(piece_ids)):
            raise ValueError("special-action targets must be unique")
        if self.piece_id in piece_ids:
            raise ValueError("an acting piece cannot target itself")

        if self.kind is ActionKind.GBESELE:
            if self.destination != self.source:
                raise ValueError("GBESELE leaves the Imperion on its source square")
            ordered = tuple(
                sorted(
                    self.targets,
                    key=lambda target: (
                        target.square.row * BOARD_SIZE + target.square.column
                    ),
                )
            )
            if self.targets != ordered:
                raise ValueError("GBESELE targets must use canonical square order")
        else:
            if self.source == self.destination:
                raise ValueError("Brainwash source and target must differ")
            if len(self.targets) != 1 or self.targets[0].square != self.destination:
                raise ValueError(
                    "Brainwash actions must identify exactly the destination target"
                )

    @property
    def target_piece_id(self) -> str | None:
        """The target identity for a single-target special action, if present."""

        return self.targets[0].piece_id if len(self.targets) == 1 else None
