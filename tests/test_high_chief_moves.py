from conftest import destinations, piece, state_with

from nubia_engine import (
    ActionKind,
    Empire,
    PieceType,
    Square,
    apply_action,
    legal_actions_from,
)


def test_high_chief_diagonals_and_exact_horizontal_switches() -> None:
    source = Square(4, 4)
    chief = piece("chief", PieceType.EAST_AFRICAN_HIGH_CHIEF)
    state = state_with([(source, chief)])
    actions = legal_actions_from(state, source)
    switches = {
        action.destination for action in actions if action.kind is ActionKind.SWITCH
    }
    assert switches == {Square(4, 3), Square(4, 5)}
    assert {Square(0, 0), Square(0, 8), Square(8, 0), Square(9, 9)} <= destinations(
        state, source
    )
    assert Square(3, 4) not in destinations(state, source)
    assert Square(4, 6) not in destinations(state, source)


def test_high_chief_diagonal_capture_blocking_and_occupied_switch() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("chief", PieceType.EAST_AFRICAN_HIGH_CHIEF)),
            (Square(2, 2), piece("enemy", PieceType.PEASANT, Empire.B)),
            (Square(4, 3), piece("side", PieceType.PEASANT, Empire.B)),
        ]
    )
    actions = legal_actions_from(state, source)
    captured = next(action for action in actions if action.destination == Square(2, 2))
    assert captured.kind is ActionKind.CAPTURE
    assert Square(1, 1) not in destinations(state, source)
    assert Square(4, 3) not in destinations(state, source)


def test_switch_application_preserves_piece_and_allegiance() -> None:
    source = Square(4, 4)
    chief = piece("chief", PieceType.EAST_AFRICAN_HIGH_CHIEF)
    state = state_with([(source, chief)], quiet=3)
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.SWITCH
    )
    result = apply_action(state, action)
    moved = result.piece_at(action.destination)
    assert moved is chief
    assert moved.piece_type is PieceType.EAST_AFRICAN_HIGH_CHIEF
    assert moved.current_empire is Empire.A
    assert result.quiet_ply_count == 4
