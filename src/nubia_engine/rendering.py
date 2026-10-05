"""Portable deterministic text rendering for NUBIA game states."""

from __future__ import annotations

from nubia_engine.coordinates import BOARD_SIZE, Square
from nubia_engine.enums import BrainwashAvailability, PieceType, Terrain
from nubia_engine.models import GameState, Piece
from nubia_engine.results import Outcome

_PIECE_SYMBOLS = {
    PieceType.IMPERION: "I",
    PieceType.QUEEN: "Q",
    PieceType.NORTH_CENTRAL_WAR_CHIEF: "W",
    PieceType.EAST_AFRICAN_HIGH_CHIEF: "C",
    PieceType.SOUTH_AFRICAN_ADVISOR: "A",
    PieceType.WEST_AFRICAN_MYSTIC: "M",
    PieceType.PEASANT: "P",
}
_ROW_LABELS = ("B:p", "B:m", "B:c", "B:o", "B:b", "A:b", "A:o", "A:c", "A:m", "A:p")


def _piece_token(piece: Piece) -> str:
    token = f"{piece.current_empire.value}{_PIECE_SYMBOLS[piece.piece_type]}"
    if piece.is_converted:
        token += "*"
    if piece.piece_type is PieceType.WEST_AFRICAN_MYSTIC:
        token += "+" if piece.brainwash is BrainwashAvailability.AVAILABLE else "-"
    return token


def _square_token(state: GameState, square: Square) -> str:
    piece = state.piece_at(square)
    if piece is not None:
        return _piece_token(piece) + ("^" if square.is_resource else "")
    if square.is_resource:
        return "R~"
    return "." if square.terrain is Terrain.LAND else "~"


def render_result(state: GameState) -> str:
    """Return a compact human-readable description of a terminal result."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if state.result is None:
        return "Result: in progress"
    result = state.result
    outcome = (
        f"Empire {result.winner.value} wins"
        if result.outcome is Outcome.WIN and result.winner is not None
        else "Draw"
    )
    reasons = ", ".join(reason.value.replace("_", " ") for reason in result.reasons)
    text = f"Result: {outcome}; reason(s): {reasons}"
    if result.scores is not None:
        text += f"; officer totals A={result.scores[0]}, B={result.scores[1]}"
    return text


def render_board(state: GameState) -> str:
    """Render the canonical fixed view with Empire B above Empire A."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    lines = [
        "NUBIA board (Empire B top, Empire A bottom)",
        "A-view paths " + "".join(f"{path:^5}" for path in range(1, BOARD_SIZE + 1)),
    ]
    for row, label in enumerate(_ROW_LABELS):
        cells = "".join(
            f"{_square_token(state, Square(row, column)):^5}"
            for column in range(BOARD_SIZE)
        )
        lines.append(f"{label:>3}  {cells}")
    lines.extend(
        (
            f"Side to move: Empire {state.side_to_move.value}",
            f"Ply: {state.ply_number} | Quiet plies: {state.quiet_ply_count}",
            render_result(state),
            "Legend: .=LAND ~=SEA R~=resource; <empire><piece> (I/Q/W/C/A/M/P)",
            "        *=converted; ^=on resource; Mystic +=power unused, -=power spent",
        )
    )
    return "\n".join(lines)
