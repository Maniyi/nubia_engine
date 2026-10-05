from conftest import destinations, piece, state_with

from nubia_engine import ActionKind, Empire, PieceType, Square, legal_actions_from


def test_imperion_moves_to_every_empty_square_in_own_palace_row_and_jumps() -> None:
    source = Square(9, 4)
    state = state_with(
        [
            (source, piece("i", PieceType.IMPERION)),
            (Square(9, 5), piece("block", PieceType.PEASANT)),
        ]
    )
    squares = destinations(state, source)
    assert len(squares) == 8
    assert Square(9, 6) in squares
    assert Square(9, 5) not in squares
    assert all(square.row == 9 for square in squares)


def test_imperion_excludes_opposing_occupied_destination_without_capture() -> None:
    source = Square(0, 5)
    state = state_with(
        [
            (source, piece("i", PieceType.IMPERION, Empire.B)),
            (Square(0, 2), piece("enemy", PieceType.QUEEN, Empire.A)),
        ],
        Empire.B,
    )
    actions = legal_actions_from(state, source)

    assert Square(0, 2) not in {action.destination for action in actions}

    action_kinds = {action.kind for action in actions}
    assert ActionKind.MOVE in action_kinds
    assert action_kinds <= {ActionKind.MOVE, ActionKind.GBESELE}


def test_imperion_outside_own_palace_row_has_no_ordinary_action() -> None:
    source = Square(8, 4)
    state = state_with([(source, piece("i", PieceType.IMPERION))])
    assert legal_actions_from(state, source) == ()
