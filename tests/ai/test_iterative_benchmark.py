from __future__ import annotations

import pytest

from nubia_ai import IterativeMinimaxAgent, IterativeSearchConfig
from nubia_ai.benchmark import _factory, main
from nubia_engine import Empire


def test_factory_builds_iterative_agent_with_both_budgets() -> None:
    agent = _factory(
        "iterative",
        max_depth=4,
        use_alpha_beta=False,
        time_limit_seconds=0.025,
        node_limit=50,
    )(Empire.A, 1)
    assert isinstance(agent, IterativeMinimaxAgent)
    assert agent.config == IterativeSearchConfig(4, False, 0.025, 50)


def test_cli_accepts_and_displays_iterative_configuration(capsys) -> None:  # type: ignore[no-untyped-def]
    arguments = [
        "--agent-a",
        "iterative",
        "--agent-b",
        "heuristic",
        "--max-depth-a",
        "2",
        "--node-limit-a",
        "50",
        "--games",
        "1",
        "--max-plies",
        "100",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert "agent-a:iterative(max_depth=2,alpha_beta=True,node_limit=50)" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-depth-a", "0"],
        ["--max-depth-b", "-1"],
        ["--node-limit-a", "0"],
        ["--node-limit-b", "-1"],
        ["--time-ms-a", "0"],
        ["--time-ms-b", "nan"],
        ["--time-ms-a", "inf"],
    ],
)
def test_cli_rejects_invalid_iterative_limits(arguments: list[str], capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    capsys.readouterr()


def test_time_limited_cli_smoke_exits_cleanly(capsys) -> None:  # type: ignore[no-untyped-def]
    assert (
        main(
            [
                "--agent-a",
                "iterative",
                "--agent-b",
                "heuristic",
                "--max-depth-a",
                "2",
                "--time-ms-a",
                "10",
                "--games",
                "1",
                "--max-plies",
                "100",
            ]
        )
        == 0
    )
    assert "time_ms=10" in capsys.readouterr().out
