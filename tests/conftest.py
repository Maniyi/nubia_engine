from collections.abc import Iterable

from nubia_engine import (
    BrainwashAvailability,
    Empire,
    GameState,
    Piece,
    PieceType,
    Square,
)


def piece(
    piece_id: str,
    piece_type: PieceType,
    empire: Empire = Empire.A,
    *,
    original_empire: Empire | None = None,
) -> Piece:
    brainwash = (
        BrainwashAvailability.AVAILABLE
        if piece_type is PieceType.WEST_AFRICAN_MYSTIC
        else None
    )
    return Piece(
        piece_id,
        piece_type,
        empire if original_empire is None else original_empire,
        empire,
        brainwash,
    )


def state_with(
    placements: Iterable[tuple[Square, Piece]],
    side: Empire = Empire.A,
    *,
    quiet: int = 0,
    ply: int = 0,
) -> GameState:
    board: list[Piece | None] = [None] * 100
    for square, placed_piece in placements:
        board[square.row * 10 + square.column] = placed_piece
    return GameState(tuple(board), side, quiet, ply)


def destinations(state: GameState, source: Square) -> set[Square]:
    from nubia_engine import legal_actions_from

    return {action.destination for action in legal_actions_from(state, source)}
