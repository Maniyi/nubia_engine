"""Deterministic engine display notation for NUBIA actions."""

from __future__ import annotations

from nubia_engine.actions import Action
from nubia_engine.enums import ActionKind, PieceType
from nubia_engine.models import GameState

_PIECE_SYMBOLS = {
    PieceType.IMPERION: "I",
    PieceType.QUEEN: "Q",
    PieceType.NORTH_CENTRAL_WAR_CHIEF: "W",
    PieceType.EAST_AFRICAN_HIGH_CHIEF: "C",
    PieceType.SOUTH_AFRICAN_ADVISOR: "A",
    PieceType.WEST_AFRICAN_MYSTIC: "M",
    PieceType.PEASANT: "P",
}


def format_action(state: GameState, action: Action) -> str:
    """Return stable human-readable engine display notation for ``action``.

    This is display notation, not an official notation or a parseable wire format.
    The supplied state must be the pre-action state so identities can be checked.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(action, Action):
        raise TypeError("action must be an Action")

    actor = state.piece_at(action.source)
    if actor is None or actor.id != action.piece_id:
        raise ValueError("action actor does not match the supplied pre-action state")
    for target in action.targets:
        target_piece = state.piece_at(target.square)
        if target_piece is None or target_piece.id != target.piece_id:
            raise ValueError(
                "action target does not match the supplied pre-action state"
            )
    if action.kind is ActionKind.CAPTURE:
        captured = state.piece_at(action.destination)
        if captured is None or captured.id != action.captured_piece_id:
            raise ValueError(
                "captured piece does not match the supplied pre-action state"
            )

    symbol = _PIECE_SYMBOLS[actor.piece_type]
    source = action.source.to_notation()
    destination = action.destination.to_notation()
    if action.kind is ActionKind.MOVE:
        return f"{symbol} {source}-{destination}"
    if action.kind is ActionKind.CAPTURE:
        return f"{symbol} {source}x{destination}"
    if action.kind is ActionKind.SWITCH:
        return f"{symbol} {source}-{destination} (switch)"
    if action.kind is ActionKind.GBESELE:
        targets = ", ".join(target.square.to_notation() for target in action.targets)
        return f"{symbol} {source} ⚡ [{targets}]"
    if action.kind is ActionKind.BRAINWASH:
        return f"{symbol} {source} 🌀 {destination} (brainwash)"
    return f"{symbol} {source} 🌀 {destination} (restore)"
