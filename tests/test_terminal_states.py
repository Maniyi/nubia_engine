import pytest
from conftest import piece, state_with

from nubia_engine import (
    GameAlreadyOverError,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    apply_action,
    legal_actions,
    legal_actions_from,
)


def test_terminal_state_generates_nothing_and_rejects_transition_immutably() -> None:
    source = Square(4, 4)
    live = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    action = legal_actions_from(live, source)[0]
    terminal = type(live)(
        live.board,
        live.side_to_move,
        live.quiet_ply_count,
        live.ply_number,
        GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
        live.position_history,
    )
    before = terminal
    assert legal_actions(terminal) == ()
    assert legal_actions_from(terminal, source) == ()
    with pytest.raises(GameAlreadyOverError, match="already over"):
        apply_action(terminal, action)
    assert terminal == before
