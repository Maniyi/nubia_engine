from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_ai import (
    AgentConfigurationError,
    HeuristicAgent,
    IllegalAgentActionError,
    MatchLimitExceededError,
    RandomAgent,
    run_match,
)
from nubia_engine import (
    Action,
    ActionKind,
    Empire,
    GameResult,
    GameState,
    Outcome,
    PieceType,
    ResultReason,
    Square,
)


def _quick_state() -> GameState:
    return state_with(
        [
            (Square.from_notation("B:b2"), piece("a", PieceType.PEASANT)),
            (
                Square.from_notation("A:b2"),
                piece("b", PieceType.PEASANT, Empire.B),
            ),
        ]
    )


def test_random_match_records_consistent_immutable_history() -> None:
    initial = _quick_state()
    result = run_match(
        initial,
        {
            Empire.A: RandomAgent(Empire.A, seed=2, name="random-a"),
            Empire.B: RandomAgent(Empire.B, seed=3, name="random-b"),
        },
        max_plies=100,
    )
    assert result.initial_state is initial
    assert result.states[0] is initial
    assert result.states[-1] == result.final_state
    assert result.result == result.final_state.result
    assert result.total_plies == len(result.actions) == len(result.states) - 1
    assert initial.result is None
    assert result.names_by_empire == {
        Empire.A: "random-a",
        Empire.B: "random-b",
    }


def test_heuristic_vs_random_finishes() -> None:
    result = run_match(
        _quick_state(),
        {
            Empire.A: HeuristicAgent(Empire.A),
            Empire.B: RandomAgent(Empire.B, seed=1),
        },
        max_plies=100,
    )
    assert result.result.outcome in (Outcome.WIN, Outcome.DRAW)


def test_initial_terminal_state_never_requests_action(terminal_draw: GameState) -> None:
    class Never:
        def __init__(self, empire: Empire) -> None:
            self.empire = empire
            self.name = "never"

        def choose_action(self, state: GameState) -> Action:
            raise AssertionError("must not be called")

    result = run_match(
        terminal_draw,
        {Empire.A: Never(Empire.A), Empire.B: Never(Empire.B)},
        max_plies=1,
    )
    assert result.total_plies == 0 and result.states == (terminal_draw,)


def test_wrong_mapping_and_non_agent_are_rejected() -> None:
    with pytest.raises(AgentConfigurationError, match="exactly"):
        run_match(_quick_state(), {Empire.A: RandomAgent(Empire.A)}, max_plies=2)
    with pytest.raises(AgentConfigurationError, match="assigned"):
        run_match(
            _quick_state(),
            {
                Empire.A: RandomAgent(Empire.B),
                Empire.B: RandomAgent(Empire.B),
            },
            max_plies=2,
        )
    with pytest.raises(AgentConfigurationError, match="satisfy"):
        run_match(
            _quick_state(),
            {Empire.A: object(), Empire.B: RandomAgent(Empire.B)},  # type: ignore[dict-item]
            max_plies=2,
        )


def test_illegal_action_from_broken_agent_is_rejected() -> None:
    class Broken:
        empire = Empire.A
        name = "broken"

        def choose_action(self, state: GameState) -> Action:
            return Action("fake", Square(0, 0), Square(0, 1), ActionKind.MOVE)

    with pytest.raises(IllegalAgentActionError, match="illegal"):
        run_match(
            _quick_state(),
            {Empire.A: Broken(), Empire.B: RandomAgent(Empire.B)},
            max_plies=2,
        )


def test_limit_is_positive_and_never_fabricates_draw(small_state: GameState) -> None:
    players = {
        Empire.A: RandomAgent(Empire.A, seed=1),
        Empire.B: RandomAgent(Empire.B, seed=2),
    }
    with pytest.raises(ValueError, match="positive"):
        run_match(small_state, players, max_plies=0)
    with pytest.raises(TypeError):
        run_match(small_state, players, max_plies=True)
    with pytest.raises(MatchLimitExceededError, match="1 plies"):
        run_match(small_state, players, max_plies=1)
    assert small_state.result is None


def test_equal_seeded_matches_are_equal() -> None:
    def play() -> object:
        return run_match(
            _quick_state(),
            {
                Empire.A: RandomAgent(Empire.A, seed=11),
                Empire.B: RandomAgent(Empire.B, seed=12),
            },
            max_plies=100,
        )

    assert play() == play()


def test_run_match_validates_state_and_mapping() -> None:
    with pytest.raises(TypeError):
        run_match("state", {}, max_plies=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_match(_quick_state(), [], max_plies=1)  # type: ignore[arg-type]


def test_terminal_fixture_has_engine_result(terminal_draw: GameState) -> None:
    assert terminal_draw == replace(
        terminal_draw,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
