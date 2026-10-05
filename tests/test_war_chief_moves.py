from conftest import destinations, piece, state_with

from nubia_engine import ActionKind, Empire, PieceType, Square, legal_actions_from


def test_war_chief_has_all_eight_l_destinations_from_center() -> None:
    source = Square(4, 4)
    state = state_with([(source, piece("war", PieceType.NORTH_CENTRAL_WAR_CHIEF))])
    assert destinations(state, source) == {
        Square(2, 3),
        Square(2, 5),
        Square(3, 2),
        Square(3, 6),
        Square(5, 2),
        Square(5, 6),
        Square(6, 3),
        Square(6, 5),
    }


def test_war_chief_corner_destinations_and_jumping() -> None:
    source = Square(0, 0)
    state = state_with(
        [
            (source, piece("war", PieceType.NORTH_CENTRAL_WAR_CHIEF)),
            (Square(0, 1), piece("block", PieceType.PEASANT)),
            (Square(1, 0), piece("block2", PieceType.PEASANT)),
        ]
    )
    assert destinations(state, source) == {Square(1, 2), Square(2, 1)}


def test_war_chief_destination_rules() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("war", PieceType.NORTH_CENTRAL_WAR_CHIEF)),
            (Square(2, 3), piece("friend", PieceType.PEASANT)),
            (Square(2, 5), piece("enemy", PieceType.PEASANT, Empire.B)),
            (Square(3, 2), piece("imperion", PieceType.IMPERION, Empire.B)),
        ]
    )
    actions = {
        action.destination: action for action in legal_actions_from(state, source)
    }
    assert Square(2, 3) not in actions
    assert actions[Square(2, 5)].kind is ActionKind.CAPTURE
    assert Square(3, 2) not in actions
