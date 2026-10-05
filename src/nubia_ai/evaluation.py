"""Transparent deterministic one-position evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from nubia_engine import (
    ALL_SQUARES,
    RESOURCE_SQUARES,
    BrainwashAvailability,
    Empire,
    GameState,
    Outcome,
    PieceType,
    legal_actions,
)

_OFFICER_VALUES = {
    PieceType.NORTH_CENTRAL_WAR_CHIEF: 3,
    PieceType.EAST_AFRICAN_HIGH_CHIEF: 5,
    PieceType.SOUTH_AFRICAN_ADVISOR: 5,
    PieceType.QUEEN: 9,
}
# Conservative bounds cover any structurally valid 100-square GameState, not only
# a reachable standard game. They keep terminal values dominant under custom
# weights too.
_MAX_OFFICER_DIFFERENCE = 900
_MAX_PEASANT_DIFFERENCE = 100
_MAX_PROGRESS_DIFFERENCE = 1_800
_MAX_MOBILITY = 12_000
_MAX_DISTANCE = 18


@dataclass(frozen=True, slots=True)
class EvaluationWeights:
    """Integer tuning values; these are AI policy, not game rules."""

    officer_material: int = 100
    peasant_presence: int = 75
    peasant_progress: int = 5
    mobility: int = 1
    terminal: int = 1_000_000

    def __post_init__(self) -> None:
        values = (
            self.officer_material,
            self.peasant_presence,
            self.peasant_progress,
            self.mobility,
            self.terminal,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("evaluation weights must be integers")
        non_terminal_bound = (
            _MAX_OFFICER_DIFFERENCE * abs(self.officer_material)
            + _MAX_PEASANT_DIFFERENCE * abs(self.peasant_presence)
            + _MAX_PROGRESS_DIFFERENCE * abs(self.peasant_progress)
            + _MAX_MOBILITY * abs(self.mobility)
        )
        if self.terminal <= non_terminal_bound:
            raise ValueError(
                "terminal weight must exceed the conservative non-terminal bound"
            )


DEFAULT_WEIGHTS = EvaluationWeights()


@dataclass(frozen=True, slots=True)
class EvaluationBreakdown:
    """Individual weighted feature contributions and their exact sum."""

    officer_material: int
    peasant_presence: int
    peasant_progress: int
    mobility: int
    terminal: int
    total: int


def _opponent(empire: Empire) -> Empire:
    return Empire.B if empire is Empire.A else Empire.A


def _officer_value(piece_type: PieceType, power: BrainwashAvailability | None) -> int:
    if piece_type is PieceType.WEST_AFRICAN_MYSTIC:
        return 7 if power is BrainwashAvailability.AVAILABLE else 3
    return _OFFICER_VALUES.get(piece_type, 0)


def _progress_value(state: GameState, empire: Empire) -> int:
    targets = tuple(square for square in RESOURCE_SQUARES if square.half is not empire)
    return sum(
        _MAX_DISTANCE
        - min(
            abs(square.row - target.row) + abs(square.column - target.column)
            for target in targets
        )
        for square, piece in zip(ALL_SQUARES, state.board, strict=True)
        if piece is not None
        and piece.piece_type is PieceType.PEASANT
        and piece.current_empire is empire
    )


def evaluate_state(
    state: GameState,
    perspective: Empire,
    weights: EvaluationWeights = DEFAULT_WEIGHTS,
) -> EvaluationBreakdown:
    """Evaluate ``state`` from ``perspective`` with explainable integer features.

    Mystic power is represented entirely in officer material through its official
    7-point unused and 3-point spent values, so it is not counted twice.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(perspective, Empire):
        raise TypeError("perspective must be an Empire")
    if not isinstance(weights, EvaluationWeights):
        raise TypeError("weights must be EvaluationWeights")

    opponent = _opponent(perspective)
    terminal = 0
    if state.result is not None:
        if state.result.outcome is Outcome.WIN:
            terminal = (
                weights.terminal
                if state.result.winner is perspective
                else -weights.terminal
            )
        return EvaluationBreakdown(0, 0, 0, 0, terminal, terminal)

    officer_totals = {Empire.A: 0, Empire.B: 0}
    peasant_totals = {Empire.A: 0, Empire.B: 0}
    for piece in state.pieces:
        officer_totals[piece.current_empire] += _officer_value(
            piece.piece_type, piece.brainwash
        )
        if piece.piece_type is PieceType.PEASANT:
            peasant_totals[piece.current_empire] += 1

    officer_material = (
        officer_totals[perspective] - officer_totals[opponent]
    ) * weights.officer_material
    peasant_presence = (
        peasant_totals[perspective] - peasant_totals[opponent]
    ) * weights.peasant_presence
    peasant_progress = (
        _progress_value(state, perspective) - _progress_value(state, opponent)
    ) * weights.peasant_progress
    mobility_sign = 1 if state.side_to_move is perspective else -1
    mobility = mobility_sign * len(legal_actions(state)) * weights.mobility
    total = officer_material + peasant_presence + peasant_progress + mobility
    return EvaluationBreakdown(
        officer_material,
        peasant_presence,
        peasant_progress,
        mobility,
        terminal,
        total,
    )
