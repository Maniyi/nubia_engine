from __future__ import annotations

import pytest

from nubia_ai import Agent, WrongTurnError
from nubia_engine import Empire, GameState, create_initial_state, legal_actions
from nubia_training import legal_action_indices
from nubia_training.neural import (
    MCTSAgent,
    MCTSConfig,
    NeuralPolicyAgent,
    PositionEvaluation,
)


class OrderedEvaluator:
    def evaluate(self, state: GameState) -> PositionEvaluation:
        indices = legal_action_indices(state)
        priors = [0.0] * len(indices)
        priors[-1] = 1.0
        return PositionEvaluation(
            indices,
            tuple(priors),
            0.0,
            state.side_to_move,
        )


def test_direct_and_mcts_agents_satisfy_protocol_and_return_exact_actions() -> None:
    state = create_initial_state(Empire.A)
    direct = NeuralPolicyAgent(Empire.A, OrderedEvaluator())
    mcts = MCTSAgent(
        Empire.A, OrderedEvaluator(), MCTSConfig(simulations=3, max_depth=2)
    )
    assert isinstance(direct, Agent)
    assert isinstance(mcts, Agent)
    assert direct.choose_action(state) == legal_actions(state)[-1]
    chosen = mcts.choose_action(state)
    assert chosen in legal_actions(state)
    assert mcts.last_search_result is not None
    assert mcts.last_search_result.action == chosen


def test_agent_wrong_side_and_temperature_validation() -> None:
    agent = NeuralPolicyAgent(Empire.B, OrderedEvaluator())
    with pytest.raises(WrongTurnError):
        agent.choose_action(create_initial_state(Empire.A))
    with pytest.raises(ValueError, match="seed"):
        NeuralPolicyAgent(Empire.A, OrderedEvaluator(), temperature=1.0)


def test_direct_positive_temperature_is_reproducible() -> None:
    state = create_initial_state(Empire.A)
    first = NeuralPolicyAgent(Empire.A, OrderedEvaluator(), temperature=1.0, seed=5)
    second = NeuralPolicyAgent(Empire.A, OrderedEvaluator(), temperature=1.0, seed=5)
    assert first.choose_action(state) == second.choose_action(state)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"temperature": -1.0}, ValueError),
        ({"temperature": True}, ValueError),
        ({"seed": True}, TypeError),
        ({"name": ""}, ValueError),
    ],
)
def test_direct_agent_configuration_validation(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        NeuralPolicyAgent(Empire.A, OrderedEvaluator(), **kwargs)  # type: ignore[arg-type]


def test_mcts_agent_configuration_and_empty_last_result() -> None:
    agent = MCTSAgent(Empire.A, OrderedEvaluator(), MCTSConfig(simulations=1))
    assert agent.last_search_result is None
    assert "simulations=1" in agent.name
    with pytest.raises(TypeError, match="Empire"):
        MCTSAgent("A", OrderedEvaluator())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        MCTSAgent(Empire.A, OrderedEvaluator(), name="")
