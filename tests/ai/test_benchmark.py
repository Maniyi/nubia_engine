from __future__ import annotations

import pytest
from conftest import piece, state_with

from nubia_ai import HeuristicAgent, RandomAgent, run_matchup
from nubia_ai.benchmark import _factory, format_summary, main, summarize_matches
from nubia_engine import Empire, PieceType, Square


def _factory_state():  # type: ignore[no-untyped-def]
    return state_with(
        [
            (Square.from_notation("B:b2"), piece("a", PieceType.PEASANT)),
            (
                Square.from_notation("A:b2"),
                piece("b", PieceType.PEASANT, Empire.B),
            ),
        ]
    )


def _random(empire: Empire, seed: int) -> RandomAgent:
    return RandomAgent(empire, seed=seed, name="random")


def _heuristic(empire: Empire, _seed: int) -> HeuristicAgent:
    return HeuristicAgent(empire, name="heuristic")


def test_matchup_counts_seeds_stats_and_side_swapping() -> None:
    summary = run_matchup(
        _heuristic,
        _random,
        games=4,
        seed=20,
        max_plies=100,
        swap_sides=True,
        initial_state_factory=_factory_state,
        agent_labels=("heuristic", "random"),
    )
    assert summary.games == summary.a_wins + summary.b_wins + summary.draws == 4
    assert summary.game_seeds == ((20, 21), (22, 23), (24, 25), (26, 27))
    assert summary.min_plies <= summary.average_plies <= summary.max_plies
    assert sum(count for _, count in summary.results_by_reason) >= summary.games
    assert summary.swap_sides
    assert "games=4 seed=20" in format_summary(summary)


def test_matchup_is_reproducible() -> None:
    first = run_matchup(
        _random,
        _random,
        games=2,
        seed=4,
        max_plies=100,
        initial_state_factory=_factory_state,
    )
    second = run_matchup(
        _random,
        _random,
        games=2,
        seed=4,
        max_plies=100,
        initial_state_factory=_factory_state,
    )
    assert first == second


def test_matchup_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_matchup(_random, _random, games=0, seed=1, max_plies=2)
    with pytest.raises(TypeError):
        run_matchup(_random, _random, games=True, seed=1, max_plies=2)
    with pytest.raises(ValueError, match="positive"):
        run_matchup(_random, _random, games=1, seed=1, max_plies=0)
    with pytest.raises(TypeError):
        run_matchup(_random, _random, games=1, seed=True, max_plies=2)
    with pytest.raises(ValueError, match="names"):
        run_matchup(
            _random,
            _random,
            games=1,
            seed=1,
            max_plies=2,
            agent_labels=("", "b"),
        )
    with pytest.raises(ValueError, match="one match"):
        summarize_matches(
            [],
            base_seed=1,
            game_seeds=[],
            swap_sides=False,
            agent_labels=("a", "b"),
        )


def test_cli_help_invalid_values_and_seeded_output(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "Benchmark baseline NUBIA agents" in capsys.readouterr().out

    for arguments in (
        ["--agent-a", "unknown"],
        ["--games", "0"],
        ["--max-plies", "-1"],
    ):
        with pytest.raises(SystemExit) as invalid_exit:
            main(arguments)
        assert invalid_exit.value.code == 2
        capsys.readouterr()


def test_cli_random_and_heuristic_configs_are_deterministic(capsys) -> None:  # type: ignore[no-untyped-def]
    args = [
        "--agent-a",
        "heuristic",
        "--agent-b",
        "random",
        "--games",
        "1",
        "--seed",
        "42",
        "--max-plies",
        "1000",
    ]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    assert capsys.readouterr().out == first
    assert "agent-a:heuristic" in first


def test_cli_reports_safety_limit_without_fabricating_a_result(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as limit_exit:
        main(["--games", "1", "--max-plies", "1"])
    assert limit_exit.value.code == 1
    assert "without an engine result" in capsys.readouterr().err


def test_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _factory("other")
