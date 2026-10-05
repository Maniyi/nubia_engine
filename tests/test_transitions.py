import pytest
from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    Empire,
    IllegalActionError,
    PieceType,
    Square,
    apply_action,
    legal_actions_from,
)


def test_legal_move_returns_new_state_without_mutating_original() -> None:
    source = Square(4, 4)
    destination = Square(4, 5)
    queen = piece("q", PieceType.QUEEN)
    state = state_with([(source, queen)], quiet=2, ply=7)
    original_board = state.board
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.destination == destination
    )
    result = apply_action(state, action)
    assert result is not state
    assert state.board is original_board
    assert state.piece_at(source) is queen
    assert result.piece_at(source) is None
    assert result.piece_at(destination) is queen
    assert len(result.pieces) == len(state.pieces)
    assert result.side_to_move is Empire.B
    assert result.ply_number == 8
    assert result.quiet_ply_count == 3
    assert len(result.board) == 100
    assert apply_action(state, action) == result


def test_capture_removes_exactly_one_piece_and_resets_quiet_count() -> None:
    source = Square(4, 4)
    destination = Square(4, 7)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (destination, piece("enemy", PieceType.PEASANT, Empire.B)),
            (Square(0, 0), piece("other", PieceType.PEASANT, Empire.B)),
        ],
        quiet=9,
    )
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.destination == destination
    )
    result = apply_action(state, action)
    assert len(result.pieces) == len(state.pieces) - 1
    assert {placed.id for placed in result.pieces} == {"q", "other"}
    assert result.quiet_ply_count == 0


@pytest.mark.parametrize(
    "action",
    [
        Action("forged", Square(4, 4), Square(4, 5), ActionKind.MOVE),
        Action("q", Square(4, 4), Square(2, 3), ActionKind.MOVE),
        Action("q", Square(4, 4), Square(4, 6), ActionKind.CAPTURE, "wrong"),
    ],
)
def test_forged_actions_are_rejected(action: Action) -> None:
    state = state_with(
        [
            (Square(4, 4), piece("q", PieceType.QUEEN)),
            (Square(4, 6), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    with pytest.raises(IllegalActionError, match="not legal"):
        apply_action(state, action)


def test_stale_and_wrong_side_actions_are_rejected() -> None:
    source = Square(4, 4)
    state = state_with([(source, piece("q", PieceType.QUEEN))])
    action = legal_actions_from(state, source)[0]
    next_state = apply_action(state, action)
    with pytest.raises(IllegalActionError):
        apply_action(next_state, action)

    wrong_side_state = state_with([(source, piece("q", PieceType.QUEEN))], Empire.B)
    with pytest.raises(IllegalActionError):
        apply_action(wrong_side_state, action)


def test_capture_of_imperion_and_non_action_values_are_rejected() -> None:
    source = Square(4, 4)
    destination = Square(4, 6)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (destination, piece("i", PieceType.IMPERION, Empire.B)),
        ]
    )
    forged = Action("q", source, destination, ActionKind.CAPTURE, "i")
    with pytest.raises(IllegalActionError):
        apply_action(state, forged)
    with pytest.raises(IllegalActionError, match="must be an Action"):
        apply_action(state, "move")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        apply_action("state", forged)  # type: ignore[arg-type]
