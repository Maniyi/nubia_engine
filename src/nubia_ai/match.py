"""Immutable agent-versus-agent match orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nubia_ai.agent import (
    Agent,
    AgentConfigurationError,
    IllegalAgentActionError,
    MatchLimitExceededError,
)
from nubia_engine import (
    Action,
    Empire,
    GameResult,
    GameState,
    apply_action,
    legal_actions,
)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """A complete immutable record of a finished engine-decided match."""

    initial_state: GameState
    actions: tuple[Action, ...]
    states: tuple[GameState, ...]
    final_state: GameState
    result: GameResult
    total_plies: int
    agent_names: tuple[tuple[Empire, str], ...]

    @property
    def names_by_empire(self) -> dict[Empire, str]:
        """Return a fresh convenient lookup for the immutable stored pairs."""

        return dict(self.agent_names)

    @property
    def agent_names_by_empire(self) -> dict[Empire, str]:
        """Explicit alias for the agent-name lookup."""

        return self.names_by_empire


def _validated_agents(agents: Mapping[Empire, Agent]) -> dict[Empire, Agent]:
    if not isinstance(agents, Mapping):
        raise TypeError("agents must be a mapping")
    if set(agents) != set(Empire):
        raise AgentConfigurationError("agents must contain exactly Empire A and B")
    validated: dict[Empire, Agent] = {}
    for empire in Empire:
        agent = agents[empire]
        if not isinstance(agent, Agent):
            raise AgentConfigurationError(
                f"agent for Empire {empire.value} does not satisfy Agent"
            )
        if agent.empire is not empire:
            raise AgentConfigurationError(
                f"agent mapped to Empire {empire.value} is assigned to "
                f"Empire {agent.empire.value}"
            )
        validated[empire] = agent
    return validated


def run_match(
    initial_state: GameState,
    agents: Mapping[Empire, Agent],
    *,
    max_plies: int,
) -> MatchResult:
    """Run until the engine records a result or the safety limit is reached."""

    if not isinstance(initial_state, GameState):
        raise TypeError("initial_state must be a GameState")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    players = _validated_agents(agents)
    state = initial_state
    actions: list[Action] = []
    states = [initial_state]

    while state.result is None:
        if len(actions) >= max_plies:
            raise MatchLimitExceededError(
                f"match reached {max_plies} plies without an engine result"
            )
        legal = legal_actions(state)
        chosen = players[state.side_to_move].choose_action(state)
        if not isinstance(chosen, Action) or chosen not in legal:
            raise IllegalAgentActionError(
                f"{players[state.side_to_move].name} returned an illegal action"
            )
        state = apply_action(state, chosen)
        actions.append(chosen)
        states.append(state)

    return MatchResult(
        initial_state,
        tuple(actions),
        tuple(states),
        state,
        state.result,
        len(actions),
        tuple((empire, players[empire].name) for empire in Empire),
    )
