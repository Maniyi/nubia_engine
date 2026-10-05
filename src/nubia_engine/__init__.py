"""Public API for the NUBIA ordinary-movement rules engine."""

from nubia_engine.actions import Action, IllegalActionError
from nubia_engine.coordinates import ALL_SQUARES, RESOURCE_SQUARES, Square
from nubia_engine.enums import (
    ActionKind,
    BrainwashAvailability,
    Empire,
    PieceType,
    RowKind,
    Terrain,
)
from nubia_engine.models import GameState, Piece
from nubia_engine.move_generation import legal_actions, legal_actions_from
from nubia_engine.setup import create_initial_state
from nubia_engine.transitions import apply_action

__all__ = [
    "ALL_SQUARES",
    "RESOURCE_SQUARES",
    "Action",
    "ActionKind",
    "BrainwashAvailability",
    "Empire",
    "GameState",
    "IllegalActionError",
    "Piece",
    "PieceType",
    "RowKind",
    "Square",
    "Terrain",
    "apply_action",
    "create_initial_state",
    "legal_actions",
    "legal_actions_from",
]
