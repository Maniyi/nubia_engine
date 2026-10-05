import pytest
from conftest import destinations, piece, state_with

from nubia_engine import ActionKind, Empire, PieceType, Square, legal_actions_from


@pytest.mark.parametrize(
    ("empire", "source", "forward", "backward"),
    [
        (Empire.A, Square(5, 4), Square(4, 4), Square(6, 4)),
        (Empire.B, Square(4, 4), Square(5, 4), Square(3, 4)),
    ],
)
def test_peasant_direction_sideways_and_no_backward(
    empire: Empire, source: Square, forward: Square, backward: Square
) -> None:
    state = state_with([(source, piece("p", PieceType.PEASANT, empire))], empire)
    squares = destinations(state, source)
    assert {forward, Square(source.row, 3), Square(source.row, 5)} <= squares
    assert backward not in squares


@pytest.mark.parametrize(
    ("empire", "source", "capture_rows"),
    [
        (Empire.A, Square(5, 4), (4, 6)),
        (Empire.B, Square(4, 4), (5, 3)),
    ],
)
def test_peasant_only_captures_diagonally_forward(
    empire: Empire, source: Square, capture_rows: tuple[int, int]
) -> None:
    opposing = Empire.B if empire is Empire.A else Empire.A
    targets = [
        (Square(capture_rows[0], 3), piece("forward-left", PieceType.QUEEN, opposing)),
        (Square(capture_rows[0], 5), piece("forward-right", PieceType.QUEEN, opposing)),
        (Square(capture_rows[1], 3), piece("back", PieceType.QUEEN, opposing)),
        (Square(source.row, 5), piece("side", PieceType.QUEEN, opposing)),
        (Square(capture_rows[0], 4), piece("straight", PieceType.QUEEN, opposing)),
    ]
    state = state_with(
        [(source, piece("p", PieceType.PEASANT, empire)), *targets], empire
    )
    captures = {
        action.destination
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.CAPTURE
    }
    assert captures == {Square(capture_rows[0], 3), Square(capture_rows[0], 5)}


@pytest.mark.parametrize(
    ("empire", "row"),
    [(Empire.A, 7), (Empire.A, 6), (Empire.B, 2), (Empire.B, 3)],
)
def test_two_square_horizontal_move_on_own_eligible_rows(
    empire: Empire, row: int
) -> None:
    source = Square(row, 4)
    state = state_with([(source, piece("p", PieceType.PEASANT, empire))], empire)
    assert {Square(row, 2), Square(row, 6)} <= destinations(state, source)


@pytest.mark.parametrize(
    ("empire", "row"),
    [
        (Empire.A, 2),
        (Empire.A, 3),
        (Empire.A, 5),
        (Empire.B, 6),
        (Empire.B, 7),
        (Empire.B, 4),
    ],
)
def test_no_two_square_move_on_other_or_opposing_rows(empire: Empire, row: int) -> None:
    source = Square(row, 4)
    state = state_with([(source, piece("p", PieceType.PEASANT, empire))], empire)
    assert Square(row, 2) not in destinations(state, source)
    assert Square(row, 6) not in destinations(state, source)


def test_two_square_horizontal_requires_clear_intermediate_and_destination() -> None:
    source = Square(7, 4)
    state = state_with(
        [
            (source, piece("p", PieceType.PEASANT)),
            (Square(7, 3), piece("middle", PieceType.QUEEN)),
            (Square(7, 6), piece("destination", PieceType.QUEEN, Empire.B)),
        ]
    )
    squares = destinations(state, source)
    assert Square(7, 2) not in squares
    assert Square(7, 6) not in squares


def test_changed_allegiance_controls_orientation_and_eligibility() -> None:
    converted = piece(
        "converted", PieceType.PEASANT, Empire.B, original_empire=Empire.A
    )
    source = Square(2, 4)
    state = state_with([(source, converted)], Empire.B)
    squares = destinations(state, source)
    assert Square(3, 4) in squares
    assert Square(1, 4) not in squares
    assert {Square(2, 2), Square(2, 6)} <= squares


def test_peasant_edge_behavior_and_imperion_immunity() -> None:
    source = Square(1, 0)
    state = state_with(
        [
            (source, piece("p", PieceType.PEASANT)),
            (Square(0, 1), piece("i", PieceType.IMPERION, Empire.B)),
        ]
    )
    assert Square(0, 1) not in destinations(state, source)
    assert all(square.column >= 0 for square in destinations(state, source))
