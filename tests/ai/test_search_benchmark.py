from __future__ import annotations

import pytest
from conftest import piece, state_with

from nubia_ai import (
    HeuristicAgent,
    MatchResult,
    MinimaxAgent,
    RandomAgent,
    SearchConfig,
    run_match,
    run_matchup,
)
from nubia_ai.benchmark import _factory, main
from nubia_engine import Empire, GameState, Outcome, PieceType, Square


def _quick_state() -> GameState:
    return state_with(
        [
            (Square.from_notation("B:b2"), piece("a", PieceType.PEASANT)),
            (Square.from_notation("A:b2"), piece("b", PieceType.PEASANT, Empire.B)),
        ]
    )


@pytest.mark.parametrize("opponent", ["random", "heuristic"])
def test_minimax_match_integration_is_deterministic(opponent: str) -> None:
    def play() -> MatchResult:
        other = (
            RandomAgent(Empire.B, seed=8)
            if opponent == "random"
            else HeuristicAgent(Empire.B)
        )
        return run_match(
            _quick_state(),
            {
                Empire.A: MinimaxAgent(Empire.A, SearchConfig(2)),
                Empire.B: other,
            },
            max_plies=100,
        )

    first = play()
    assert first == play()
    assert first.result.outcome in (Outcome.WIN, Outcome.DRAW)
    assert first.total_plies == len(first.actions)


def test_factory_builds_requested_minimax_mode() -> None:
    agent = _factory("minimax", depth=3, use_alpha_beta=False)(Empire.A, 42)
    assert isinstance(agent, MinimaxAgent)
    assert agent.config == SearchConfig(3, False)


def test_minimax_matchup_supports_side_swapping() -> None:
    summary = run_matchup(
        _factory("minimax", depth=1),
        _factory("random"),
        games=2,
        seed=9,
        max_plies=100,
        swap_sides=True,
        initial_state_factory=_quick_state,
        agent_labels=("minimax", "random"),
    )
    assert summary.games == 2
    assert summary.swap_sides
    assert summary.game_seeds == ((9, 10), (11, 12))


def test_cli_accepts_depths_reports_configuration_and_validates(capsys) -> None:  # type: ignore[no-untyped-def]
    args = [
        "--agent-a",
        "minimax",
        "--agent-b",
        "heuristic",
        "--depth-a",
        "1",
        "--games",
        "1",
        "--seed",
        "2",
        "--max-plies",
        "100",
        "--no-alpha-beta",
    ]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "agent-a:minimax(depth=1,alpha_beta=False)" in output
    for flag in ("--depth-a", "--depth-b"):
        with pytest.raises(SystemExit) as invalid:
            main([flag, "0"])
        assert invalid.value.code == 2
        capsys.readouterr()
