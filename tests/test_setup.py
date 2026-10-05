from collections import Counter

import pytest

from nubia_engine import (
    BrainwashAvailability,
    Empire,
    PieceType,
    Square,
    create_initial_state,
)

EXPECTED_COUNTS = {
    PieceType.IMPERION: 1,
    PieceType.QUEEN: 1,
    PieceType.SOUTH_AFRICAN_ADVISOR: 2,
    PieceType.EAST_AFRICAN_HIGH_CHIEF: 2,
    PieceType.NORTH_CENTRAL_WAR_CHIEF: 2,
    PieceType.WEST_AFRICAN_MYSTIC: 2,
    PieceType.PEASANT: 10,
}

OFFICER_STARTS = {
    PieceType.IMPERION: {"p5"},
    PieceType.QUEEN: {"p6"},
    PieceType.SOUTH_AFRICAN_ADVISOR: {"p2", "p9"},
    PieceType.EAST_AFRICAN_HIGH_CHIEF: {"m2", "m9"},
    PieceType.NORTH_CENTRAL_WAR_CHIEF: {"m4", "m7"},
    PieceType.WEST_AFRICAN_MYSTIC: {"m3", "m8"},
}


@pytest.mark.parametrize("first_player", list(Empire))
def test_initial_counts_and_counters(first_player: Empire) -> None:
    state = create_initial_state(first_player)
    assert len(state.board) == 100
    assert len(state.pieces) == 40
    assert state.side_to_move is first_player
    assert state.quiet_ply_count == 0
    assert state.ply_number == 0
    for empire in Empire:
        pieces = [piece for piece in state.pieces if piece.current_empire is empire]
        assert len(pieces) == 20
        assert Counter(piece.piece_type for piece in pieces) == EXPECTED_COUNTS


def test_every_piece_starts_on_the_exact_required_square() -> None:
    state = create_initial_state(Empire.A)
    for empire in Empire:
        for piece_type, relative_squares in OFFICER_STARTS.items():
            actual = {
                square.to_notation()[2:]
                for square in (
                    Square.from_notation(f"{empire.value}:{relative}")
                    for relative in relative_squares
                )
                if state.piece_at(square) is not None
                and state.piece_at(square).piece_type is piece_type  # type: ignore[union-attr]
            }
            assert actual == relative_squares
        for path in range(1, 11):
            piece = state.piece_at(Square.from_notation(f"{empire.value}:c{path}"))
            assert piece is not None
            assert piece.piece_type is PieceType.PEASANT
            assert piece.current_empire is empire


def test_initial_allegiances_brainwash_and_identifiers() -> None:
    state = create_initial_state(Empire.A)
    assert all(piece.original_empire is piece.current_empire for piece in state.pieces)
    mystics = [
        piece
        for piece in state.pieces
        if piece.piece_type is PieceType.WEST_AFRICAN_MYSTIC
    ]
    non_mystics = [piece for piece in state.pieces if piece not in mystics]
    assert len(mystics) == 4
    assert all(piece.brainwash is BrainwashAvailability.AVAILABLE for piece in mystics)
    assert all(piece.brainwash is None for piece in non_mystics)
    ids = [piece.id for piece in state.pieces]
    assert len(ids) == len(set(ids)) == 40


def test_initial_state_is_collision_free_and_deterministic() -> None:
    first = create_initial_state(Empire.A)
    second = create_initial_state(Empire.A)
    assert first == second
    assert sum(piece is not None for piece in first.board) == 40


def test_first_player_is_the_only_difference_between_initial_states() -> None:
    state_a = create_initial_state(Empire.A)
    state_b = create_initial_state(Empire.B)
    assert state_a.board == state_b.board
    assert state_a.quiet_ply_count == state_b.quiet_ply_count
    assert state_a.ply_number == state_b.ply_number
    assert state_a.side_to_move is Empire.A
    assert state_b.side_to_move is Empire.B


def test_first_player_must_be_an_empire() -> None:
    with pytest.raises(TypeError):
        create_initial_state("A")  # type: ignore[arg-type]
