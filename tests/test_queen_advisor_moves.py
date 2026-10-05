import pytest
from conftest import destinations, piece, state_with

from nubia_engine import ActionKind, Empire, PieceType, Square, legal_actions_from


@pytest.mark.parametrize(
    ("piece_type", "expected_count"),
    [(PieceType.QUEEN, 35), (PieceType.SOUTH_AFRICAN_ADVISOR, 18)],
)
def test_sliders_on_empty_board(piece_type: PieceType, expected_count: int) -> None:
    source = Square(4, 4)
    state = state_with([(source, piece("slider", piece_type))])
    assert len(legal_actions_from(state, source)) == expected_count


def test_queen_has_rows_columns_and_diagonals_but_advisor_has_no_diagonals() -> None:
    source = Square(4, 4)
    queen_state = state_with([(source, piece("q", PieceType.QUEEN))])
    advisor_state = state_with([(source, piece("a", PieceType.SOUTH_AFRICAN_ADVISOR))])
    assert {Square(4, 9), Square(0, 4), Square(0, 0), Square(8, 8)} <= destinations(
        queen_state, source
    )
    assert Square(0, 0) not in destinations(advisor_state, source)


@pytest.mark.parametrize(
    "piece_type", [PieceType.QUEEN, PieceType.SOUTH_AFRICAN_ADVISOR]
)
def test_slider_blockers_capture_first_opponent_and_stop(piece_type: PieceType) -> None:
    source = Square(4, 4)
    own = piece("slider", piece_type)
    friendly = piece("friend", PieceType.PEASANT)
    enemy = piece("enemy", PieceType.PEASANT, Empire.B)
    state = state_with([(source, own), (Square(4, 2), friendly), (Square(4, 7), enemy)])
    actions = legal_actions_from(state, source)
    by_destination = {action.destination: action for action in actions}
    assert Square(4, 2) not in by_destination
    assert Square(4, 1) not in by_destination
    assert by_destination[Square(4, 7)].kind is ActionKind.CAPTURE
    assert by_destination[Square(4, 7)].captured_piece_id == "enemy"
    assert Square(4, 8) not in by_destination


def test_queen_corner_and_diagonal_blocker() -> None:
    source = Square(0, 0)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(2, 2), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    squares = destinations(state, source)
    assert Square(1, 1) in squares
    assert Square(2, 2) in squares
    assert Square(3, 3) not in squares
    assert all(square.row >= 0 and square.column >= 0 for square in squares)
