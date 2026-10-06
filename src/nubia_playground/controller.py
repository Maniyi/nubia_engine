"""Pygame-independent gameplay and setup controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nubia_ai import (
    Agent,
    HeuristicAgent,
    IterativeMinimaxAgent,
    IterativeSearchConfig,
    MinimaxAgent,
    RandomAgent,
    SearchConfig,
)
from nubia_engine import (
    Action,
    Empire,
    GameState,
    Square,
    apply_action,
    create_initial_state,
    format_action,
    legal_actions,
    legal_actions_from,
)


class GameMode(StrEnum):
    """Supported local player combinations."""

    HUMAN_VS_HUMAN = "Human vs Human"
    HUMAN_VS_AGENT = "Human vs Agent"
    AGENT_VS_AGENT = "Agent vs Agent"


class AgentKind(StrEnum):
    """Existing AI policies exposed by the playground."""

    RANDOM = "Random"
    HEURISTIC = "Heuristic"
    MINIMAX = "Minimax"
    ITERATIVE = "Iterative Minimax"


class SetupValidationError(ValueError):
    """Raised for locally invalid setup values."""


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Safe configuration for one optional agent."""

    kind: AgentKind = AgentKind.RANDOM
    seed: int = 0
    depth: int = 1
    max_depth: int = 2
    node_limit: int | None = 2_000
    time_limit_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AgentKind):
            raise SetupValidationError("agent type is invalid")
        self._positive_int("search depth", self.depth)
        self._positive_int("maximum depth", self.max_depth)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SetupValidationError("random seed must be an integer")
        if self.node_limit is not None:
            self._positive_int("node budget", self.node_limit)
        if self.time_limit_ms is not None:
            self._positive_int("time budget", self.time_limit_ms)
        if self.kind is AgentKind.ITERATIVE:
            supplied = sum(
                value is not None for value in (self.node_limit, self.time_limit_ms)
            )
            if supplied != 1:
                raise SetupValidationError(
                    "iterative search needs exactly one node or time budget"
                )

    @staticmethod
    def _positive_int(label: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SetupValidationError(f"{label} must be a positive integer")

    def description(self) -> str:
        """Return a compact display description."""

        if self.kind is AgentKind.RANDOM:
            return f"Random (seed {self.seed})"
        if self.kind is AgentKind.HEURISTIC:
            return "Heuristic (1 ply)"
        if self.kind is AgentKind.MINIMAX:
            return f"Minimax (depth {self.depth})"
        budget = (
            f"{self.node_limit} nodes"
            if self.node_limit is not None
            else f"{self.time_limit_ms} ms"
        )
        return f"Iterative (max {self.max_depth}, {budget})"


@dataclass(frozen=True, slots=True)
class SetupConfig:
    """Validated settings used to create or restart a game."""

    mode: GameMode = GameMode.HUMAN_VS_AGENT
    first_player: Empire = Empire.A
    human_empire: Empire = Empire.A
    agent_a: AgentSettings = AgentSettings()
    agent_b: AgentSettings = AgentSettings()
    move_delay_ms: int = 500

    def __post_init__(self) -> None:
        if not isinstance(self.mode, GameMode):
            raise SetupValidationError("game mode is invalid")
        if not isinstance(self.first_player, Empire):
            raise SetupValidationError("first empire is invalid")
        if not isinstance(self.human_empire, Empire):
            raise SetupValidationError("human empire is invalid")
        if not isinstance(self.agent_a, AgentSettings) or not isinstance(
            self.agent_b, AgentSettings
        ):
            raise SetupValidationError("agent settings are invalid")
        if (
            isinstance(self.move_delay_ms, bool)
            or not isinstance(self.move_delay_ms, int)
            or not 50 <= self.move_delay_ms <= 10_000
        ):
            raise SetupValidationError("move delay must be 50-10000 ms")


def create_agent(empire: Empire, settings: AgentSettings) -> Agent:
    """Create one existing public AI agent from validated settings."""

    if settings.kind is AgentKind.RANDOM:
        return RandomAgent(empire, seed=settings.seed)
    if settings.kind is AgentKind.HEURISTIC:
        return HeuristicAgent(empire)
    if settings.kind is AgentKind.MINIMAX:
        return MinimaxAgent(empire, SearchConfig(settings.depth))
    return IterativeMinimaxAgent(
        empire,
        IterativeSearchConfig(
            settings.max_depth,
            node_limit=settings.node_limit,
            time_limit_seconds=(
                None
                if settings.time_limit_ms is None
                else settings.time_limit_ms / 1000
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """One action plus its engine display notation."""

    action: Action
    notation: str


class GameController:
    """Own one immutable engine state and UI-only interaction state."""

    def __init__(self, config: SetupConfig, *, state: GameState | None = None) -> None:
        if not isinstance(config, SetupConfig):
            raise TypeError("config must be a SetupConfig")
        self.config = config
        self.state = (
            state if state is not None else create_initial_state(config.first_player)
        )
        self.agents = self._create_agents()
        self.history: list[MoveRecord] = []
        self.selected_source: Square | None = None
        self.selected_actions: tuple[Action, ...] = ()
        self.action_choices: tuple[Action, ...] = ()
        self.paused = config.mode is GameMode.AGENT_VS_AGENT
        self.thinking = False
        self.error: str | None = None

    def _create_agents(self) -> dict[Empire, Agent]:
        controlled: set[Empire]
        if self.config.mode is GameMode.HUMAN_VS_HUMAN:
            controlled = set()
        elif self.config.mode is GameMode.HUMAN_VS_AGENT:
            controlled = {self.other_empire(self.config.human_empire)}
        else:
            controlled = set(Empire)
        settings = {Empire.A: self.config.agent_a, Empire.B: self.config.agent_b}
        return {empire: create_agent(empire, settings[empire]) for empire in controlled}

    @staticmethod
    def other_empire(empire: Empire) -> Empire:
        return Empire.B if empire is Empire.A else Empire.A

    @property
    def is_human_turn(self) -> bool:
        return self.state.result is None and self.state.side_to_move not in self.agents

    @property
    def current_agent(self) -> Agent | None:
        return self.agents.get(self.state.side_to_move)

    @property
    def latest_action(self) -> str:
        return self.history[-1].notation if self.history else "None"

    def clear_selection(self) -> None:
        self.selected_source = None
        self.selected_actions = ()
        self.action_choices = ()

    def select_square(self, square: Square) -> bool:
        """Select a source or apply its unambiguous destination action."""

        if not isinstance(square, Square):
            raise TypeError("square must be a Square")
        if not self.is_human_turn:
            self.error = "It is not a human turn."
            return False
        if self.selected_source is None:
            actions = legal_actions_from(self.state, square)
            if not actions:
                return False
            self.selected_source = square
            self.selected_actions = actions
            self.action_choices = tuple(
                action for action in actions if action.destination == square
            )
            self.error = None
            return True
        if square == self.selected_source:
            self.clear_selection()
            return False
        candidates = tuple(
            action for action in self.selected_actions if action.destination == square
        )
        if len(candidates) == 1:
            return self.submit_human_action(candidates[0])
        if candidates:
            self.action_choices = candidates
            return True
        replacement = legal_actions_from(self.state, square)
        if replacement:
            self.selected_source = square
            self.selected_actions = replacement
            self.action_choices = tuple(
                action for action in replacement if action.destination == square
            )
            return True
        return False

    def submit_human_action(self, action: Action) -> bool:
        """Apply the exact selected engine action if it remains current."""

        if not self.is_human_turn:
            self.error = "It is not a human turn."
            return False
        current = legal_actions_from(self.state, action.source)
        if action not in self.selected_actions or action not in current:
            self.error = "That selection is stale; choose again."
            self.clear_selection()
            return False
        self._apply(action)
        return True

    def agent_snapshot(self) -> tuple[GameState, Agent] | None:
        """Return the immutable state and active agent for background work."""

        agent = self.current_agent
        if self.state.result is not None or agent is None:
            return None
        return self.state, agent

    def apply_agent_choice(self, snapshot: GameState, action: Action) -> bool:
        """Apply an agent choice only if its immutable snapshot is still current."""

        if snapshot is not self.state:
            return False
        if action not in legal_actions(self.state):
            name = self.current_agent.name if self.current_agent else "Agent"
            self.error = f"{name} returned an illegal action."
            self.paused = True
            return False
        self._apply(action)
        return True

    def step_agent(self) -> bool:
        """Synchronously choose and apply exactly one agent action."""

        snapshot = self.agent_snapshot()
        if snapshot is None:
            return False
        state, agent = snapshot
        try:
            action = agent.choose_action(state)
        except Exception as error:
            self.error = f"Agent error: {error}"
            self.paused = True
            return False
        return self.apply_agent_choice(state, action)

    def _apply(self, action: Action) -> None:
        before = self.state
        notation = format_action(before, action)
        self.state = apply_action(before, action)
        self.history.append(MoveRecord(action, notation))
        self.clear_selection()
        self.error = None
        if self.state.result is not None:
            self.paused = True

    def toggle_pause(self) -> None:
        if self.config.mode is GameMode.AGENT_VS_AGENT and self.state.result is None:
            self.paused = not self.paused

    def restart(self) -> None:
        """Start fresh using exactly the same validated configuration."""

        self.state = create_initial_state(self.config.first_player)
        self.agents = self._create_agents()
        self.history.clear()
        self.clear_selection()
        self.paused = self.config.mode is GameMode.AGENT_VS_AGENT
        self.thinking = False
        self.error = None

    def controller_name(self, empire: Empire) -> str:
        agent = self.agents.get(empire)
        return "Human" if agent is None else agent.name

    def settings_for(self, empire: Empire) -> AgentSettings:
        return self.config.agent_a if empire is Empire.A else self.config.agent_b
