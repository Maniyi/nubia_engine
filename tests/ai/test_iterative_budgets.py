from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_ai import (
    IterativeSearchConfig,
    NoActionAvailableError,
    SearchConfig,
    SearchStopReason,
    WrongTurnError,
    iterative_search_state,
    search_state,
)
from nubia_engine import (
    Empire,
    GameResult,
    GameState,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    legal_actions,
)


def _state() -> GameState:
    return state_with(
        [
            (Square(6, 0), piece("advisor", PieceType.SOUTH_AFRICAN_ADVISOR)),
            (Square(5, 4), piece("a-peasant", PieceType.PEASANT)),
            (Square(5, 1), piece("threat", PieceType.PEASANT, Empire.B)),
            (Square(2, 9), piece("b-peasant", PieceType.PEASANT, Empire.B)),
        ]
    )


class TriggerClock:
    def __init__(self, zero_calls: int) -> None:
        self.zero_calls = zero_calls
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 0.0 if self.calls <= self.zero_calls else 2.0


def test_node_limit_before_depth_one_uses_first_legal_fallback_exactly() -> None:
    state = _state()
    result = iterative_search_state(
        state,
        Empire.A,
        IterativeSearchConfig(3, node_limit=1),
        clock=lambda: 0.0,
    )
    assert result.action == legal_actions(state)[0]
    assert result.completed_depth == 0
    assert result.score is None
    assert result.principal_variation == ()
    assert result.iterations == ()
    assert result.used_fallback
    assert result.stop_reason is SearchStopReason.NODE_LIMIT
    assert result.cumulative_stats.nodes_visited == 1


def test_node_limit_keeps_depth_one_and_counts_partial_depth_two() -> None:
    state = _state()
    depth_one = search_state(state, Empire.A, SearchConfig(1))
    limit = depth_one.stats.nodes_visited + 1
    result = iterative_search_state(
        state,
        Empire.A,
        IterativeSearchConfig(3, node_limit=limit),
        clock=lambda: 0.0,
    )
    assert result.stop_reason is SearchStopReason.NODE_LIMIT
    assert result.completed_depth == 1
    assert result.iterations[0].result == depth_one
    assert (result.action, result.score, result.principal_variation) == (
        depth_one.action,
        depth_one.score,
        depth_one.principal_variation,
    )
    assert result.cumulative_stats.nodes_visited == limit
    assert result.cumulative_stats.nodes_visited > depth_one.stats.nodes_visited
    assert not result.used_fallback


def test_node_limited_decision_and_telemetry_are_repeatable() -> None:
    config = IterativeSearchConfig(3, node_limit=40)
    first = iterative_search_state(_state(), Empire.A, config, clock=lambda: 5.0)
    second = iterative_search_state(_state(), Empire.A, config, clock=lambda: 5.0)
    assert first == second


def test_time_limit_before_depth_one_has_zero_search_work() -> None:
    result = iterative_search_state(
        _state(),
        Empire.A,
        IterativeSearchConfig(2, time_limit_seconds=1.0),
        clock=TriggerClock(1),
    )
    assert result.used_fallback
    assert result.stop_reason is SearchStopReason.TIME_LIMIT
    assert result.cumulative_stats.nodes_visited == 0
    assert result.elapsed_seconds >= 0


def test_time_limit_between_iterations_retains_depth_one() -> None:
    state = _state()
    depth_one = search_state(state, Empire.A, SearchConfig(1))
    # start, precheck, iteration start, one check per node, iteration end
    clock = TriggerClock(depth_one.stats.nodes_visited + 4)
    result = iterative_search_state(
        state,
        Empire.A,
        IterativeSearchConfig(3, time_limit_seconds=1.0),
        clock=clock,
    )
    assert result.completed_depth == 1
    assert result.stop_reason is SearchStopReason.TIME_LIMIT
    assert result.iterations[0].result == depth_one
    assert result.cumulative_stats.nodes_visited == depth_one.stats.nodes_visited


def test_time_limit_during_depth_two_includes_partial_work() -> None:
    state = _state()
    depth_one = search_state(state, Empire.A, SearchConfig(1))
    completed_calls = depth_one.stats.nodes_visited + 4
    # Permit depth-two precheck, start, root, and one successor check.
    clock = TriggerClock(completed_calls + 4)
    result = iterative_search_state(
        state,
        Empire.A,
        IterativeSearchConfig(3, time_limit_seconds=1.0),
        clock=clock,
    )
    assert result.completed_depth == 1
    assert result.stop_reason is SearchStopReason.TIME_LIMIT
    assert len(result.iterations) == 1
    assert result.cumulative_stats.nodes_visited > depth_one.stats.nodes_visited
    assert result.elapsed_seconds >= 0


def test_combined_limits_report_first_and_node_wins_simultaneous_check() -> None:
    node_first = iterative_search_state(
        _state(),
        Empire.A,
        IterativeSearchConfig(2, time_limit_seconds=1.0, node_limit=1),
        clock=TriggerClock(4),
    )
    assert node_first.stop_reason is SearchStopReason.NODE_LIMIT

    time_first = iterative_search_state(
        _state(),
        Empire.A,
        IterativeSearchConfig(2, time_limit_seconds=1.0, node_limit=1000),
        clock=TriggerClock(1),
    )
    assert time_first.stop_reason is SearchStopReason.TIME_LIMIT


def test_terminal_and_wrong_turn_are_rejected_before_fallback() -> None:
    state = _state()
    terminal = replace(
        state,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
    config = IterativeSearchConfig(1, node_limit=1)
    with pytest.raises(NoActionAvailableError):
        iterative_search_state(terminal, Empire.A, config)
    with pytest.raises(WrongTurnError):
        iterative_search_state(state, Empire.B, config)
