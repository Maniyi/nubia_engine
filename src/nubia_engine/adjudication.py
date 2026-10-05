"""Pure post-action adjudication in the confirmed rules priority order."""

from __future__ import annotations

from nubia_engine.coordinates import ALL_SQUARES
from nubia_engine.enums import BrainwashAvailability, Empire, PieceType
from nubia_engine.models import GameState
from nubia_engine.position import position_key
from nubia_engine.results import GameResult, Outcome, ResultReason

_OFFICER_VALUES = {
    PieceType.NORTH_CENTRAL_WAR_CHIEF: 3,
    PieceType.EAST_AFRICAN_HIGH_CHIEF: 5,
    PieceType.SOUTH_AFRICAN_ADVISOR: 5,
    PieceType.QUEEN: 9,
}


def officer_scores(state: GameState) -> tuple[int, int]:
    """Return officer totals in canonical ``(Empire A, Empire B)`` order."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    totals = {Empire.A: 0, Empire.B: 0}
    for piece in state.pieces:
        if piece.piece_type is PieceType.WEST_AFRICAN_MYSTIC:
            value = 7 if piece.brainwash is BrainwashAvailability.AVAILABLE else 3
        else:
            value = _OFFICER_VALUES.get(piece.piece_type, 0)
        totals[piece.current_empire] += value
    return totals[Empire.A], totals[Empire.B]


def _mine_winner(state: GameState) -> Empire | None:
    for index, piece in enumerate(state.board):
        if piece is None or piece.piece_type is not PieceType.PEASANT:
            continue
        square = ALL_SQUARES[index]
        if square.is_resource and square.half is not piece.current_empire:
            return piece.current_empire
    return None


def adjudicate(state: GameState) -> GameResult | None:
    """Evaluate a post-action state in the authoritative priority order."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")

    mine_winner = _mine_winner(state)
    if mine_winner is not None:
        return GameResult(Outcome.WIN, mine_winner, (ResultReason.MINE_VICTORY,))

    if all(piece.piece_type is not PieceType.PEASANT for piece in state.pieces):
        scores = officer_scores(state)
        winner = (
            Empire.A
            if scores[0] > scores[1]
            else Empire.B
            if scores[1] > scores[0]
            else None
        )
        outcome = Outcome.WIN if winner is not None else Outcome.DRAW
        return GameResult(
            outcome,
            winner,
            (ResultReason.NO_PEASANTS_SCORING,),
            scores,
        )

    reasons: list[ResultReason] = []
    key = position_key(state)
    if state.position_history.count(key) >= 3:
        reasons.append(ResultReason.THREEFOLD_REPETITION)
    if state.quiet_ply_count >= 40:
        reasons.append(ResultReason.NO_PROGRESS_40_PLIES)
    if reasons:
        return GameResult(Outcome.DRAW, None, tuple(reasons))
    return None
