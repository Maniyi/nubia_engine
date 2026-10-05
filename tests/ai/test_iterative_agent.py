from __future__ import annotations

import pytest
from conftest import piece, state_with

from nubia_ai import (
    Agent,
    IterativeMinimaxAgent,
    IterativeSearchConfig,
    MinimaxAgent,
    NoActionAvailableError,
    SearchConfig,
    WrongTurnError,
)
from nubia_engine import ActionKind, Empire, GameState, PieceType, Square, legal_actions


def _state() -> GameState:
    return state_with(
        [
            (Square(6, 0), piece("advisor", PieceType.SOUTH_AFRICAN_ADVISOR)),
            (Square(5, 4), piece("a-peasant", PieceType.PEASANT)),
            (Square(5, 1), piece("threat", PieceType.PEASANT, Empire.B)),
            (Square(2, 9), piece("b-peasant", PieceType.PEASANT, Empire.B)),
        ]
    )


def test_agent_protocol_name_and_fixed_depth_compatibility() -> None:
    state = _state()
    agent = IterativeMinimaxAgent(
        Empire.A,
        IterativeSearchConfig(2),
        clock=lambda: 0.0,
    )
    fixed = MinimaxAgent(Empire.A, SearchConfig(2)).search(state)
    result = agent.search(state)
    assert isinstance(agent, Agent)
    assert "max_depth=2" in agent.name
    assert agent.choose_action(state) == result.action == fixed.action
    assert (result.score, result.principal_variation) == (
        fixed.score,
        fixed.principal_variation,
    )


def test_agent_validation_wrong_turn_terminal_and_immutability(
    terminal_draw: GameState,
) -> None:
    state = _state()
    before = state
    agent = IterativeMinimaxAgent(
        Empire.A,
        IterativeSearchConfig(2, node_limit=1),
        clock=lambda: 0.0,
    )
    assert agent.choose_action(state) in legal_actions(state)
    assert state is before
    with pytest.raises(WrongTurnError):
        IterativeMinimaxAgent(Empire.B, agent.config).choose_action(state)
    with pytest.raises(NoActionAvailableError):
        agent.choose_action(terminal_draw)
    with pytest.raises(TypeError):
        IterativeMinimaxAgent("A", agent.config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IterativeMinimaxAgent(Empire.A, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IterativeMinimaxAgent(Empire.A, agent.config, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IterativeMinimaxAgent(
            Empire.A,
            agent.config,
            clock=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        IterativeMinimaxAgent(Empire.A, agent.config, name=" ")

    named = IterativeMinimaxAgent(
        Empire.A,
        IterativeSearchConfig(2, time_limit_seconds=0.1, node_limit=20),
        clock=lambda: 0.0,
    )
    assert "time=0.1s" in named.name
    assert "nodes=20" in named.name


def test_special_action_can_be_selected() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    action = IterativeMinimaxAgent(
        Empire.A,
        IterativeSearchConfig(1),
        clock=lambda: 0.0,
    ).choose_action(state)
    assert action.kind is ActionKind.BRAINWASH
