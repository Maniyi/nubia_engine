from dataclasses import FrozenInstanceError

import pytest

from nubia_engine import (
    BrainwashAvailability,
    Empire,
    GameState,
    Piece,
    PieceType,
    Square,
)


def make_piece(
    piece_id: str = "A-QUEEN-1",
    piece_type: PieceType = PieceType.QUEEN,
    brainwash: BrainwashAvailability | None = None,
) -> Piece:
    return Piece(piece_id, piece_type, Empire.A, Empire.A, brainwash)


def test_piece_is_immutable_and_has_stable_value_identity() -> None:
    piece = make_piece()
    assert piece.id == "A-QUEEN-1"
    assert piece == make_piece()
    with pytest.raises(FrozenInstanceError):
        piece.current_empire = Empire.B  # type: ignore[misc]


def test_piece_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        make_piece("")


def test_piece_rejects_non_enum_domain_values() -> None:
    with pytest.raises(TypeError):
        Piece("id", "QUEEN", Empire.A, Empire.A)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Piece("id", PieceType.QUEEN, "A", Empire.A)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Piece("id", PieceType.QUEEN, Empire.A, "A")  # type: ignore[arg-type]


def test_mystic_requires_brainwash_state() -> None:
    with pytest.raises(ValueError):
        make_piece(piece_type=PieceType.WEST_AFRICAN_MYSTIC)


def test_non_mystic_forbids_brainwash_state() -> None:
    with pytest.raises(ValueError):
        make_piece(brainwash=BrainwashAvailability.AVAILABLE)


def test_brainwash_availability_accessor() -> None:
    available = make_piece(
        piece_type=PieceType.WEST_AFRICAN_MYSTIC,
        brainwash=BrainwashAvailability.AVAILABLE,
    )
    spent = make_piece(
        "A-WEST_AFRICAN_MYSTIC-2",
        PieceType.WEST_AFRICAN_MYSTIC,
        BrainwashAvailability.SPENT,
    )
    assert available.brainwash_available
    assert not spent.brainwash_available
    assert not make_piece().brainwash_available


def test_game_state_requires_immutable_100_square_board() -> None:
    with pytest.raises(TypeError):
        GameState([None] * 100, Empire.A)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GameState((None,) * 99, Empire.A)
    with pytest.raises(ValueError):
        GameState((None,) * 101, Empire.A)


def test_game_state_rejects_invalid_entries_and_duplicate_ids() -> None:
    with pytest.raises(TypeError):
        GameState(("piece",) + (None,) * 99, Empire.A)  # type: ignore[arg-type]
    piece = make_piece()
    with pytest.raises(ValueError):
        GameState((piece, piece) + (None,) * 98, Empire.A)


def test_game_state_validates_side_and_counters() -> None:
    with pytest.raises(TypeError):
        GameState((None,) * 100, "A")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GameState((None,) * 100, Empire.A, quiet_ply_count=-1)
    with pytest.raises(ValueError):
        GameState((None,) * 100, Empire.A, ply_number=-1)


def test_piece_lookup_uses_fixed_board_coordinates() -> None:
    piece = make_piece()
    board = (piece,) + (None,) * 99
    state = GameState(board, Empire.A)
    assert state.piece_at(Square(0, 0)) is piece
    assert state.piece_at(Square(0, 1)) is None
    assert state.pieces == (piece,)
