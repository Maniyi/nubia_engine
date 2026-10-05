from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from conftest import piece, state_with

from nubia_ai import (
    CumulativeSearchStats,
    IterationRecord,
    IterativeSearchConfig,
    SearchConfig,
    SearchStopReason,
    iterative_search_state,
    search_state,
)
from nubia_engine import Empire, GameState, PieceType, Square


def _branching_state() -> GameState:
    return state_with(
        [
            (Square(6, 0), piece("advisor", PieceType.SOUTH_AFRICAN_ADVISOR)),
            (Square(5, 4), piece("a-peasant", PieceType.PEASANT)),
            (Square(5, 1), piece("threat", PieceType.PEASANT, Empire.B)),
            (Square(2, 9), piece("b-peasant", PieceType.PEASANT, Empire.B)),
        ]
    )


@pytest.mark.parametrize("use_alpha_beta", [False, True])
def test_unlimited_iterations_equal_each_fixed_search(
    use_alpha_beta: bool,
) -> None:
    state = _branching_state()
    result = iterative_search_state(
        state,
        Empire.A,
        IterativeSearchConfig(3, use_alpha_beta),
        clock=lambda: 4.0,
    )
    fixed = tuple(
        search_state(state, Empire.A, SearchConfig(depth, use_alpha_beta))
        for depth in range(1, 4)
    )

    assert result.stop_reason is SearchStopReason.MAX_DEPTH_COMPLETED
    assert not result.used_fallback
    assert result.completed_depth == 3
    assert tuple(record.depth for record in result.iterations) == (1, 2, 3)
    assert tuple(record.result for record in result.iterations) == fixed
    assert (result.action, result.score, result.principal_variation) == (
        fixed[-1].action,
        fixed[-1].score,
        fixed[-1].principal_variation,
    )
    assert result.cumulative_stats.nodes_visited == sum(
        search.stats.nodes_visited for search in fixed
    )
    assert result.cumulative_stats.leaves_evaluated == sum(
        search.stats.leaves_evaluated for search in fixed
    )
    assert result.cumulative_stats.alpha_beta_cutoffs == sum(
        search.stats.alpha_beta_cutoffs for search in fixed
    )
    assert result.cumulative_stats.max_depth_reached == 3
    assert state == _branching_state()


def test_configuration_validation_immutability_and_value_comparison() -> None:
    config = IterativeSearchConfig(2, False, 0.5, 20)
    assert config == IterativeSearchConfig(2, False, 0.5, 20)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.max_depth = 3  # type: ignore[misc]
    for depth in (0, -1):
        with pytest.raises(ValueError, match="at least one"):
            IterativeSearchConfig(depth)
    with pytest.raises(TypeError):
        IterativeSearchConfig(True)
    with pytest.raises(TypeError):
        IterativeSearchConfig(1, 1)  # type: ignore[arg-type]
    for limit in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            IterativeSearchConfig(1, time_limit_seconds=limit)
    with pytest.raises(TypeError):
        IterativeSearchConfig(1, time_limit_seconds=True)
    for limit in (0, -1):
        with pytest.raises(ValueError, match="strictly positive"):
            IterativeSearchConfig(1, node_limit=limit)
    with pytest.raises(TypeError):
        IterativeSearchConfig(1, node_limit=True)


def test_completed_result_collections_and_records_are_immutable() -> None:
    result = iterative_search_state(
        _branching_state(),
        Empire.A,
        IterativeSearchConfig(1),
        clock=lambda: 0.0,
    )
    assert isinstance(result.iterations, tuple)
    assert isinstance(result.principal_variation, tuple)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.completed_depth = 2  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.iterations[0].depth = 2  # type: ignore[misc]


def test_iteration_and_cumulative_telemetry_validation() -> None:
    result = iterative_search_state(
        _branching_state(), Empire.A, IterativeSearchConfig(1), clock=lambda: 0.0
    )
    record = result.iterations[0]
    for value, error in ((True, TypeError), (0, ValueError)):
        with pytest.raises(error):
            replace(record, depth=value)
    with pytest.raises(TypeError):
        replace(record, result=object())  # type: ignore[arg-type]
    for elapsed in (-1.0, float("inf")):
        with pytest.raises(ValueError):
            replace(record, elapsed_seconds=elapsed)

    assert CumulativeSearchStats(0, 0, 0, 0).nodes_visited == 0
    with pytest.raises(TypeError):
        CumulativeSearchStats(True, 0, 0, 0)
    with pytest.raises(ValueError):
        CumulativeSearchStats(-1, 0, 0, 0)
    with pytest.raises(ValueError):
        CumulativeSearchStats(1, 2, 0, 0)


def test_iterative_result_validation_rejects_inconsistent_values() -> None:
    state = _branching_state()
    completed = iterative_search_state(
        state, Empire.A, IterativeSearchConfig(1), clock=lambda: 0.0
    )
    fallback = iterative_search_state(
        state,
        Empire.A,
        IterativeSearchConfig(1, node_limit=1),
        clock=lambda: 0.0,
    )
    alternative = next(
        action
        for action in search_state(state, Empire.A, SearchConfig(1)).principal_variation
        if action == completed.action
    )
    invalid_completed: tuple[dict[str, object], ...] = (
        {"action": object()},
        {"score": True},
        {"principal_variation": []},
        {"completed_depth": True},
        {"stop_reason": "done"},
        {"iterations": []},
        {"cumulative_stats": object()},
        {"elapsed_seconds": -1.0},
        {"used_fallback": 1},
        {"iterations": (IterationRecord(2, completed.iterations[0].result, 0.0),)},
        {"completed_depth": 0},
        {"iterations": (), "completed_depth": 0},
        {"action": replace(alternative, piece_id="different")},
    )
    for changes in invalid_completed:
        with pytest.raises((TypeError, ValueError)):
            replace(completed, **changes)  # type: ignore[arg-type]

    invalid_fallback: tuple[dict[str, object], ...] = (
        {"score": 0},
        {"principal_variation": (fallback.action,)},
        {"stop_reason": SearchStopReason.MAX_DEPTH_COMPLETED},
    )
    for changes in invalid_fallback:
        with pytest.raises(ValueError):
            replace(fallback, **changes)  # type: ignore[arg-type]


def test_iterative_search_validates_api_and_clock() -> None:
    state = _branching_state()
    with pytest.raises(TypeError):
        iterative_search_state(state, Empire.A, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        iterative_search_state(
            state,
            Empire.A,
            IterativeSearchConfig(1),
            clock=object(),  # type: ignore[arg-type]
        )
    for clock, error in (
        (lambda: True, TypeError),
        (lambda: float("inf"), ValueError),
    ):
        with pytest.raises(error):
            iterative_search_state(
                state, Empire.A, IterativeSearchConfig(1), clock=clock
            )

    values = iter((0.0, "bad"))
    with pytest.raises(TypeError):
        iterative_search_state(
            state,
            Empire.A,
            IterativeSearchConfig(1, time_limit_seconds=1.0),
            clock=lambda: next(values),  # type: ignore[arg-type,return-value]
        )
    non_finite = iter((0.0, float("nan")))
    with pytest.raises(ValueError):
        iterative_search_state(
            state,
            Empire.A,
            IterativeSearchConfig(1, time_limit_seconds=1.0),
            clock=lambda: next(non_finite),
        )
