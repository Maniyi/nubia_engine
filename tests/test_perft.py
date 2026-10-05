from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_engine import (
    ActionKind,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    create_initial_state,
    legal_actions,
    perft,
    perft_divide,
)


def test_depth_validation_zero_one_and_input_immutability() -> None:
    state = create_initial_state(Empire.A)
    before = state
    with pytest.raises(ValueError):
        perft(state, -1)
    with pytest.raises(TypeError):
        perft(state, True)
    with pytest.raises(TypeError):
        perft("state", 1)  # type: ignore[arg-type]
    assert perft(state, 0) == 1
    assert perft(state, 1) == len(legal_actions(state))
    assert perft_divide(state, 0) == ()
    assert state == before


def test_terminal_positive_depth_is_zero() -> None:
    live = state_with([])
    terminal = replace(
        live,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
    assert perft(terminal, 1) == 0
    assert perft_divide(terminal, 2) == ()


def test_manually_enumerable_peasant_position_has_three_moves() -> None:
    source = Square(5, 4)
    state = state_with([(source, piece("p", PieceType.PEASANT))])
    assert perft(state, 1) == 3  # forward, left, right; no eligible two-square move
    capture_state = state_with(
        [
            (source, piece("p", PieceType.PEASANT)),
            (Square(4, 3), piece("target", PieceType.QUEEN, Empire.B)),
        ]
    )
    assert perft(capture_state, 1) == 4


def test_divide_is_legal_order_deterministic_and_sums_to_perft() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("target", PieceType.PEASANT, Empire.B)),
            (Square(7, 0), piece("survivor", PieceType.PEASANT)),
        ]
    )
    divide = perft_divide(state, 1)
    assert tuple(entry.action for entry in divide) == legal_actions(state)
    assert all(entry.count == 1 and entry.label for entry in divide)
    assert all(entry.descendant_count == entry.count for entry in divide)
    assert any(entry.action.kind is ActionKind.BRAINWASH for entry in divide)
    assert sum(entry.count for entry in divide) == perft(state, 1)
    assert divide == perft_divide(state, 1)


def test_initial_position_shallow_smoke() -> None:
    state = create_initial_state(Empire.A)
    # 30 Peasant moves plus 10 currently unblocked officer moves.
    assert perft(state, 1) == 40
