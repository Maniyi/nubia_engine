"""Shared structural agent contract and AI-layer errors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nubia_engine import Action, Empire, GameState, legal_actions


class AgentError(ValueError):
    """Base class for failures owned by the AI layer."""


class WrongTurnError(AgentError):
    """Raised when an agent is asked to act for the other empire."""


class NoActionAvailableError(AgentError):
    """Raised when action selection is requested from a terminal state."""


class IllegalAgentActionError(AgentError):
    """Raised when an agent returns an action that is not currently legal."""


class AgentConfigurationError(AgentError):
    """Raised when a match does not receive one valid agent per empire."""


class MatchLimitExceededError(AgentError):
    """Raised when a non-terminal match reaches its configured safety limit."""


@runtime_checkable
class Agent(Protocol):
    """A policy that selects, but never applies, one engine action."""

    empire: Empire
    name: str

    def choose_action(self, state: GameState) -> Action:
        """Choose one currently legal action without mutating ``state``."""
        ...


def available_actions_for(agent: Agent, state: GameState) -> tuple[Action, ...]:
    """Validate a selection request and return engine-owned legal actions."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if state.result is not None:
        raise NoActionAvailableError("cannot choose an action from a terminal state")
    if state.side_to_move is not agent.empire:
        raise WrongTurnError(
            f"{agent.name} controls Empire {agent.empire.value}, "
            f"but Empire {state.side_to_move.value} is to move"
        )
    actions = legal_actions(state)
    if not actions:
        raise NoActionAvailableError(
            "the engine returned no legal actions for a non-terminal state"
        )
    return actions
