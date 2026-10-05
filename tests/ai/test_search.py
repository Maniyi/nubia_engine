from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from conftest import piece, state_with

from nubia_ai import (
    DEFAULT_WEIGHTS,
    NoActionAvailableError,
    SearchConfig,
    SearchInvariantError,
    SearchResult,
    SearchStats,
    WrongTurnError,
    search_state,
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
    apply_action,
    legal_actions,
    position_key,
)


def _branching_state() -> GameState:
    return state_with(
        [
            (Square(6, 0), piece("advisor", PieceType.SOUTH_AFRICAN_ADVISOR)),
            (Square(5, 4), piece("a-peasant", PieceType.PEASANT)),
            (Square(5, 1), piece("threat", PieceType.PEASANT, Empire.B)),
            (Square(2, 9), piece("b-peasant", PieceType.PEASANT, Empire.B)),
        ]
    )


def test_configuration_statistics_and_result_are_validated_and_immutable() -> None:
    assert SearchConfig(2) == SearchConfig(2, True)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        SearchConfig(1).depth = 2  # type: ignore[misc]
    for depth in (0, -1):
        with pytest.raises(ValueError, match="at least one"):
            SearchConfig(depth)
    with pytest.raises(TypeError):
        SearchConfig(True)
    with pytest.raises(TypeError):
        SearchConfig(1, 1)  # type: ignore[arg-type]

    stats = SearchStats(2, 1, 0, 1)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        stats.nodes_visited = 3  # type: ignore[misc]
    with pytest.raises(ValueError):
        SearchStats(0, 0, 0, 0)
    with pytest.raises(ValueError):
        SearchStats(1, 2, 0, 0)
    with pytest.raises(TypeError):
        SearchStats(1, 1, False, 1)

    action = legal_actions(_branching_state())[0]
    result = SearchResult(action, 4, (action,), stats)
    assert result.action == result.principal_variation[0]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.score = 5  # type: ignore[misc]
    with pytest.raises(ValueError, match="empty"):
        SearchResult(action, 0, (), stats)
    with pytest.raises(ValueError, match="begin"):
        SearchResult(action, 0, (legal_actions(_branching_state())[1],), stats)


def test_plain_minimax_and_alpha_beta_have_identical_public_decision() -> None:
    state = _branching_state()
    plain = search_state(state, Empire.A, SearchConfig(2, False))
    pruned = search_state(state, Empire.A, SearchConfig(2, True))
    assert (plain.action, plain.score, plain.principal_variation) == (
        pruned.action,
        pruned.score,
        pruned.principal_variation,
    )
    assert plain.stats.alpha_beta_cutoffs == 0
    assert pruned.stats.alpha_beta_cutoffs > 0
    assert pruned.stats.nodes_visited <= plain.stats.nodes_visited
    assert plain.stats.max_depth_reached == pruned.stats.max_depth_reached == 2


def test_statistics_count_root_leaves_and_are_scoped_per_call() -> None:
    state = _branching_state()
    first = search_state(state, Empire.A, SearchConfig(1))
    second = search_state(state, Empire.A, SearchConfig(1))
    assert first == second
    assert first.stats.nodes_visited == len(legal_actions(state)) + 1
    assert first.stats.leaves_evaluated == len(legal_actions(state))
    assert first.stats.max_depth_reached == 1


def test_principal_variation_replays_legally_without_mutating_root() -> None:
    state = _branching_state()
    before = state
    result = search_state(state, Empire.A, SearchConfig(3))
    replay = state
    for action in result.principal_variation:
        assert action in legal_actions(replay)
        replay = apply_action(replay, action)
    assert result.action in legal_actions(state)
    assert len(result.principal_variation) <= 3
    assert replay.result is not None or len(result.principal_variation) == 3
    assert state is before


def test_root_validation_and_nonterminal_no_action_invariant() -> None:
    state = _branching_state()
    terminal = replace(
        state,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
    with pytest.raises(NoActionAvailableError, match="terminal"):
        search_state(terminal, Empire.A, SearchConfig(1))
    with pytest.raises(WrongTurnError, match="Empire B"):
        search_state(state, Empire.B, SearchConfig(1))
    with pytest.raises(SearchInvariantError, match="no legal actions"):
        search_state(state_with([]), Empire.A, SearchConfig(1))
    with pytest.raises(TypeError):
        search_state(state, Empire.A, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        search_state(state, Empire.A, SearchConfig(1), object())  # type: ignore[arg-type]


def test_terminal_states_are_evaluated_before_remaining_horizon() -> None:
    mine = Square.from_notation("B:o2")
    state = state_with(
        [
            (Square(4, mine.column), piece("winner", PieceType.PEASANT)),
            (Square(7, 8), piece("a-other", PieceType.PEASANT)),
            (Square(2, 8), piece("b-other", PieceType.PEASANT, Empire.B)),
        ]
    )
    result = search_state(state, Empire.A, SearchConfig(3))
    assert result.action.destination == mine
    assert result.score == DEFAULT_WEIGHTS.terminal
    assert len(result.principal_variation) == 1


def test_forced_opponent_mine_win_has_terminal_loss_score() -> None:
    state = state_with(
        [
            (Square(9, 4), piece("imperion", PieceType.IMPERION)),
            (Square(5, 1), piece("threat", PieceType.PEASANT, Empire.B)),
        ]
    )
    result = search_state(state, Empire.A, SearchConfig(2))
    assert result.score == -DEFAULT_WEIGHTS.terminal
    assert len(result.principal_variation) == 2
    assert (
        apply_action(
            apply_action(state, result.principal_variation[0]),
            result.principal_variation[1],
        ).result
        is not None
    )


def test_no_progress_and_repetition_draws_are_inherited_from_engine() -> None:
    base = state_with(
        [
            (Square(9, 4), piece("a-i", PieceType.IMPERION)),
            (Square(0, 5), piece("b-i", PieceType.IMPERION, Empire.B)),
            (Square(7, 0), piece("a-p", PieceType.PEASANT)),
            (Square(2, 9), piece("b-p", PieceType.PEASANT, Empire.B)),
        ],
        quiet=39,
    )
    no_progress = search_state(base, Empire.A, SearchConfig(2))
    assert no_progress.score == 0
    assert apply_action(base, no_progress.action).result is not None

    successor_keys = tuple(
        position_key(apply_action(replace(base, quiet_ply_count=0), action))
        for action in legal_actions(base)
    )
    repeated = replace(
        base,
        quiet_ply_count=0,
        position_history=tuple(key for key in successor_keys for _ in range(2)),
    )
    repetition = search_state(repeated, Empire.A, SearchConfig(2))
    replayed = apply_action(repeated, repetition.action)
    assert repetition.score == 0
    assert replayed.result is not None
    assert ResultReason.THREEFOLD_REPETITION in replayed.result.reasons


def test_special_action_can_begin_principal_variation() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    result = search_state(state, Empire.A, SearchConfig(1))
    assert result.action.kind is ActionKind.BRAINWASH
    assert result.principal_variation[0].kind is ActionKind.BRAINWASH
