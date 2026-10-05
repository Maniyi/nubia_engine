from __future__ import annotations

import pytest

from nubia_ai import (
    Agent,
    HeuristicAgent,
    NoActionAvailableError,
    RandomAgent,
    WrongTurnError,
)
from nubia_engine import Empire, GameState


def test_agents_structurally_satisfy_protocol() -> None:
    assert isinstance(RandomAgent(Empire.A, seed=1), Agent)
    assert isinstance(HeuristicAgent(Empire.B), Agent)


@pytest.mark.parametrize("agent_type", [RandomAgent, HeuristicAgent])
def test_wrong_turn_and_terminal_are_ai_errors(
    agent_type: type[RandomAgent] | type[HeuristicAgent],
    small_state: GameState,
    terminal_draw: GameState,
) -> None:
    with pytest.raises(WrongTurnError, match="Empire B"):
        agent_type(Empire.B).choose_action(small_state)
    with pytest.raises(NoActionAvailableError, match="terminal"):
        agent_type(Empire.A).choose_action(terminal_draw)


def test_agent_constructors_validate_configuration() -> None:
    with pytest.raises(TypeError):
        RandomAgent("A")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RandomAgent(Empire.A, name="")
    with pytest.raises(TypeError):
        HeuristicAgent(Empire.A, weights=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HeuristicAgent(Empire.A, name=" ")
