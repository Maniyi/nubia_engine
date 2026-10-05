from conftest import destinations, piece, state_with

from nubia_engine import ActionKind, Empire, PieceType, Square, legal_actions_from


def test_mystic_moves_one_and_two_squares_in_all_directions() -> None:
    source = Square(4, 4)
    state = state_with([(source, piece("m", PieceType.WEST_AFRICAN_MYSTIC))])
    assert len(legal_actions_from(state, source)) == 16
    assert {Square(3, 4), Square(2, 4), Square(5, 5), Square(6, 6)} <= destinations(
        state, source
    )


def test_mystic_capture_is_exactly_two_squares_with_clear_middle() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("far", PieceType.PEASANT, Empire.B)),
            (Square(4, 5), piece("adjacent", PieceType.PEASANT, Empire.B)),
        ]
    )
    actions = {
        action.destination: action for action in legal_actions_from(state, source)
    }
    assert actions[Square(2, 4)].kind is ActionKind.CAPTURE
    assert Square(4, 5) not in actions
    assert Square(4, 6) not in actions


def test_mystic_cannot_capture_friendly_or_imperion_at_distance_two() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 2), piece("friend", PieceType.PEASANT)),
            (Square(6, 6), piece("i", PieceType.IMPERION, Empire.B)),
        ]
    )
    squares = destinations(state, source)
    assert Square(2, 2) not in squares
    assert Square(6, 6) not in squares


def test_mystic_actions_only_use_ordinary_kinds() -> None:
    source = Square(4, 4)
    state = state_with([(source, piece("m", PieceType.WEST_AFRICAN_MYSTIC))])
    assert {action.kind for action in legal_actions_from(state, source)} == {
        ActionKind.MOVE
    }
