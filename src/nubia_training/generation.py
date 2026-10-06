"""Deterministic agent construction and raw-game generation."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from nubia_ai import (
    DEFAULT_WEIGHTS,
    EvaluationWeights,
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
    apply_action,
    create_initial_state,
    format_action,
    legal_actions,
)
from nubia_training.action_space import ACTION_SPACE_VERSION, action_to_index
from nubia_training.encoding import encode_state
from nubia_training.errors import AgentSpecError
from nubia_training.records import (
    AgentSpec,
    PlyRecord,
    RawGameRecord,
    create_raw_game_record,
)


class GenerationAgent(Protocol):
    empire: Empire
    name: str

    def choose_action(self, state: GameState) -> Action: ...


_WEIGHT_NAMES = (
    "officer_material",
    "peasant_presence",
    "peasant_progress",
    "mobility",
    "terminal",
)


def agent_spec(
    agent_type: str,
    *,
    name: str | None = None,
    seed: int | None = None,
    depth: int = 1,
    max_depth: int = 2,
    node_limit: int | None = 1000,
    use_alpha_beta: bool = True,
    weights: EvaluationWeights = DEFAULT_WEIGHTS,
) -> AgentSpec:
    """Build a validated specification from public behavior settings."""

    if not isinstance(weights, EvaluationWeights):
        raise AgentSpecError("weights must be EvaluationWeights")
    config: dict[str, int | bool | None] = {}
    if agent_type in {"heuristic", "minimax", "iterative"}:
        config.update({key: getattr(weights, key) for key in _WEIGHT_NAMES})
    if agent_type == "minimax":
        config.update(depth=depth, use_alpha_beta=use_alpha_beta)
    elif agent_type == "iterative":
        config.update(
            max_depth=max_depth,
            node_limit=node_limit,
            use_alpha_beta=use_alpha_beta,
            time_limit_seconds=None,
        )
    default_names = {
        "random": "RandomAgent",
        "heuristic": "HeuristicAgent",
        "minimax": f"MinimaxAgent(depth={depth})",
        "iterative": f"IterativeMinimaxAgent(max_depth={max_depth},nodes={node_limit})",
    }
    if agent_type not in default_names:
        raise AgentSpecError(f"unknown agent type: {agent_type!r}")
    return AgentSpec(
        agent_type=agent_type,
        name=default_names[agent_type] if name is None else name,
        configuration=tuple(sorted(config.items())),
        seed=seed,
    )


def _weights(config: dict[str, object]) -> EvaluationWeights:
    values = {name: config.pop(name) for name in _WEIGHT_NAMES}
    return EvaluationWeights(**values)  # type: ignore[arg-type]


def create_agent(spec: AgentSpec, empire: Empire) -> GenerationAgent:
    """Construct one existing AI agent from a stable public specification."""

    config: dict[str, object] = dict(spec.configuration)
    if spec.agent_type == "random":
        return RandomAgent(empire, seed=spec.seed, name=spec.name)
    weights = _weights(config)
    if spec.agent_type == "heuristic":
        if config:
            raise AgentSpecError(f"unknown heuristic configuration: {sorted(config)}")
        return HeuristicAgent(empire, weights=weights, name=spec.name)
    use_alpha_beta = config.pop("use_alpha_beta")
    if spec.agent_type == "minimax":
        depth = config.pop("depth")
        if config:
            raise AgentSpecError(f"unknown minimax configuration: {sorted(config)}")
        return MinimaxAgent(
            empire,
            SearchConfig(depth, use_alpha_beta),  # type: ignore[arg-type]
            weights,
            name=spec.name,
        )
    max_depth = config.pop("max_depth")
    node_limit = config.pop("node_limit")
    time_limit = config.pop("time_limit_seconds")
    if time_limit is not None:
        raise AgentSpecError("wall-clock iterative generation is not reproducible")
    if config:
        raise AgentSpecError(f"unknown iterative configuration: {sorted(config)}")
    return IterativeMinimaxAgent(
        empire,
        IterativeSearchConfig(
            max_depth,  # type: ignore[arg-type]
            use_alpha_beta,  # type: ignore[arg-type]
            node_limit=node_limit,  # type: ignore[arg-type]
        ),
        weights,
        name=spec.name,
    )


def generate_game(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    *,
    first_player: Empire,
    max_plies: int,
    generator_config: tuple[tuple[str, str | int | float | bool | None], ...] = (),
) -> RawGameRecord:
    """Play one game, preserving genuine terminal versus stable abort status."""

    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    state = create_initial_state(first_player)
    players = {
        Empire.A: create_agent(agent_a, Empire.A),
        Empire.B: create_agent(agent_b, Empire.B),
    }
    plies: list[PlyRecord] = []
    abort_reason: str | None = None
    while state.result is None:
        if len(plies) >= max_plies:
            abort_reason = "maximum_plies_reached"
            break
        actor = state.side_to_move
        try:
            legal = legal_actions(state)
            action = players[actor].choose_action(state)
            if action not in legal:
                abort_reason = "illegal_agent_action"
                break
            action_index = action_to_index(state, action, perspective=actor)
            notation = format_action(state, action)
            pre = encode_state(state, perspective=actor).fingerprint()
            successor = apply_action(state, action)
            post = encode_state(successor, perspective=actor).fingerprint()
        except Exception as error:  # Agent failures become inspectable aborted records.
            abort_reason = f"agent_exception:{type(error).__name__}"
            break
        plies.append(
            PlyRecord(
                len(plies),
                actor,
                action_index,
                ACTION_SPACE_VERSION,
                notation,
                pre,
                post,
                action.kind,
            )
        )
        state = successor
    completed = state.result is not None
    return create_raw_game_record(
        first_player=first_player,
        agent_a=agent_a,
        agent_b=agent_b,
        plies=tuple(plies),
        status="completed" if completed else "aborted",
        final_result=state.result,
        total_plies=len(plies),
        abort_reason=None if completed else abort_reason or "generation_interrupted",
        final_state_fingerprint=encode_state(state).fingerprint(),
        generator_config=generator_config,
    )


def _scheduled_spec(spec: AgentSpec, seed: int) -> AgentSpec:
    return replace(spec, seed=seed) if spec.agent_type == "random" else spec


def generate_series(
    first_spec: AgentSpec,
    second_spec: AgentSpec,
    *,
    game_count: int,
    base_seed: int,
    first_player: Empire = Empire.A,
    alternate_first_player: bool = False,
    swap_sides: bool = False,
    max_plies: int = 200,
) -> tuple[RawGameRecord, ...]:
    """Generate games using a documented index/side/seed schedule."""

    if isinstance(game_count, bool) or not isinstance(game_count, int):
        raise TypeError("game_count must be an integer")
    if game_count <= 0:
        raise ValueError("game_count must be positive")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    records: list[RawGameRecord] = []
    for index in range(game_count):
        scheduled_first = _scheduled_spec(first_spec, base_seed + index * 2)
        scheduled_second = _scheduled_spec(second_spec, base_seed + index * 2 + 1)
        swapped = swap_sides and index % 2 == 1
        a_spec, b_spec = (
            (scheduled_second, scheduled_first)
            if swapped
            else (scheduled_first, scheduled_second)
        )
        opening = (
            (Empire.B if first_player is Empire.A else Empire.A)
            if alternate_first_player and index % 2 == 1
            else first_player
        )
        records.append(
            generate_game(
                a_spec,
                b_spec,
                first_player=opening,
                max_plies=max_plies,
                generator_config=(
                    ("base_seed", base_seed),
                    ("game_index", index),
                    ("seed_first_identity", base_seed + index * 2),
                    ("seed_second_identity", base_seed + index * 2 + 1),
                    ("sides_swapped", swapped),
                ),
            )
        )
    return tuple(records)
