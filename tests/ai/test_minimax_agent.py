from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_ai import (
    DEFAULT_WEIGHTS,
    Agent,
    EvaluationWeights,
    HeuristicAgent,
    MinimaxAgent,
    NoActionAvailableError,
    SearchConfig,
    WrongTurnError,
)
from nubia_engine import (
    ActionKind,
    Empire,
    GameResult,
    GameState,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    create_initial_state,
    legal_actions,
)


def _positions() -> tuple[tuple[GameState, EvaluationWeights], ...]:
    mine = Square.from_notation("B:o2")
    immediate_win = state_with(
        [
            (Square(4, mine.column), piece("p", PieceType.PEASANT)),
            (Square(7, 8), piece("other", PieceType.PEASANT)),
            (Square(2, 8), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    special = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    tied = state_with(
        [
            (Square(9, 4), piece("i", PieceType.IMPERION)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    zero_features = replace(
        DEFAULT_WEIGHTS,
        officer_material=0,
        peasant_presence=0,
        peasant_progress=0,
        mobility=0,
    )
    return (
        (create_initial_state(Empire.A), DEFAULT_WEIGHTS),
        (immediate_win, DEFAULT_WEIGHTS),
        (special, DEFAULT_WEIGHTS),
        (tied, zero_features),
    )


@pytest.mark.parametrize(("state", "weights"), _positions())
def test_depth_one_matches_heuristic_agent(
    state: GameState, weights: EvaluationWeights
) -> None:
    heuristic = HeuristicAgent(Empire.A, weights=weights)
    minimax = MinimaxAgent(Empire.A, SearchConfig(1), weights)
    assert minimax.choose_action(state) == heuristic.choose_action(state)


def test_agent_protocol_name_validation_and_custom_weights(
    small_state: GameState,
) -> None:
    weights = replace(DEFAULT_WEIGHTS, mobility=0)
    agent = MinimaxAgent(Empire.A, SearchConfig(1), weights)
    assert isinstance(agent, Agent)
    assert "depth=1" in agent.name
    assert agent.weights is weights
    assert agent.choose_action(small_state) in legal_actions(small_state)
    with pytest.raises(TypeError):
        MinimaxAgent("A", SearchConfig(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MinimaxAgent(Empire.A, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MinimaxAgent(Empire.A, SearchConfig(1), name=" ")
    with pytest.raises(ValueError):
        MinimaxAgent(Empire.A, SearchConfig(1), name="")


def test_agent_rejects_wrong_turn_and_terminal(
    small_state: GameState,
    terminal_draw: GameState,
) -> None:
    with pytest.raises(WrongTurnError):
        MinimaxAgent(Empire.B, SearchConfig(1)).choose_action(small_state)
    with pytest.raises(NoActionAvailableError):
        MinimaxAgent(Empire.A, SearchConfig(1)).choose_action(terminal_draw)


def test_immediate_win_and_special_action_are_selected() -> None:
    immediate_win = _positions()[1][0]
    special = _positions()[2][0]
    assert MinimaxAgent(Empire.A, SearchConfig(2)).choose_action(
        immediate_win
    ).destination == Square.from_notation("B:o2")
    assert (
        MinimaxAgent(Empire.A, SearchConfig(1)).choose_action(special).kind
        is ActionKind.BRAINWASH
    )


def test_terminal_fixture_is_actually_terminal(small_state: GameState) -> None:
    terminal = replace(
        small_state,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
    assert terminal.result is not None
