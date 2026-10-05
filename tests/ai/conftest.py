from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_engine import (
    Empire,
    GameResult,
    GameState,
    Outcome,
    PieceType,
    ResultReason,
    Square,
)


@pytest.fixture
def small_state() -> GameState:
    return state_with(
        [
            (Square(4, 4), piece("a-queen", PieceType.QUEEN)),
            (Square(7, 0), piece("a-peasant", PieceType.PEASANT)),
            (Square(5, 7), piece("b-queen", PieceType.QUEEN, Empire.B)),
            (Square(2, 9), piece("b-peasant", PieceType.PEASANT, Empire.B)),
        ]
    )


@pytest.fixture
def terminal_draw(small_state: GameState) -> GameState:
    return replace(
        small_state,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
