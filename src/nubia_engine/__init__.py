"""Public API for the NUBIA rules engine."""

from nubia_engine.actions import (
    Action,
    ActionTarget,
    GameAlreadyOverError,
    IllegalActionError,
)
from nubia_engine.adjudication import adjudicate, officer_scores
from nubia_engine.coordinates import ALL_SQUARES, RESOURCE_SQUARES, Square
from nubia_engine.defense import is_defended_for_gbesele
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
from nubia_engine.position import PositionKey, position_key
from nubia_engine.results import GameResult, Outcome, ResultReason
from nubia_engine.setup import create_initial_state
from nubia_engine.transitions import apply_action

__all__ = [
    "ALL_SQUARES",
    "RESOURCE_SQUARES",
    "Action",
    "ActionKind",
    "ActionTarget",
    "BrainwashAvailability",
    "Empire",
    "GameAlreadyOverError",
    "GameResult",
    "GameState",
    "IllegalActionError",
    "Outcome",
    "Piece",
    "PieceType",
    "PositionKey",
    "ResultReason",
    "RowKind",
    "Square",
    "Terrain",
    "adjudicate",
    "apply_action",
    "create_initial_state",
    "is_defended_for_gbesele",
    "legal_actions",
    "legal_actions_from",
    "officer_scores",
    "position_key",
]
