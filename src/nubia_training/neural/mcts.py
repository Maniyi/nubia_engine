"""Deterministic PUCT Monte Carlo tree search over authoritative engine states."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nubia_engine import Action, GameState, Outcome, apply_action, legal_actions
from nubia_training import (
    ACTION_SPACE_VERSION,
    STATE_ENCODING_VERSION,
    TrainingExample,
    action_to_index,
    encode_state,
)
from nubia_training.neural.checkpoints import CHECKPOINT_VERSION
from nubia_training.neural.errors import MCTSConfigurationError, MCTSSearchError
from nubia_training.neural.evaluator import PositionEvaluation, PositionEvaluator
from nubia_training.neural.model import MODEL_ARCHITECTURE_VERSION

MCTS_VERSION = 1


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MCTSConfigurationError(f"{name} must be a positive integer")
    return value


def _finite(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MCTSConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise MCTSConfigurationError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 32
    c_puct: float = 1.5
    temperature: float = 0.0
    max_depth: int = 128
    root_dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.25
    random_seed: int | None = None
    root_noise_enabled: bool = False
    version: int = MCTS_VERSION
    state_encoding_version: int = STATE_ENCODING_VERSION
    action_space_version: int = ACTION_SPACE_VERSION
    model_architecture_version: int = MODEL_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        _positive_int("simulations", self.simulations)
        _positive_int("max_depth", self.max_depth)
        _finite("c_puct", self.c_puct, positive=True)
        _finite("temperature", self.temperature)
        _finite("root_noise_fraction", self.root_noise_fraction)
        if not 0.0 <= self.root_noise_fraction <= 1.0:
            raise MCTSConfigurationError("root_noise_fraction must be in [0, 1]")
        if not isinstance(self.root_noise_enabled, bool):
            raise MCTSConfigurationError("root_noise_enabled must be Boolean")
        if self.root_noise_enabled:
            _finite("root_dirichlet_alpha", self.root_dirichlet_alpha, positive=True)
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise MCTSConfigurationError("random_seed must be an integer or None")
        if (
            self.root_noise_enabled or self.temperature > 0.0
        ) and self.random_seed is None:
            raise MCTSConfigurationError("stochastic search requires an explicit seed")
        expected = {
            "version": MCTS_VERSION,
            "state_encoding_version": STATE_ENCODING_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        }
        for name, value in expected.items():
            if isinstance(getattr(self, name), bool) or getattr(self, name) != value:
                raise MCTSConfigurationError(f"{name} must be {value}")


@dataclass(frozen=True, slots=True)
class RootChildStatistics:
    action: Action
    action_index: int
    prior: float
    visits: int
    mean_value: float
    engine_order: int

    @property
    def index(self) -> int:
        return self.action_index

    @property
    def mean_action_value(self) -> float:
        return self.mean_value


@dataclass(frozen=True, slots=True, eq=False)
class RootVisitPolicy:
    action_indices: NDArray[np.uint16]
    visit_counts: NDArray[np.int64]
    probabilities: NDArray[np.float32]

    def __post_init__(self) -> None:
        indices = np.frombuffer(
            np.asarray(self.action_indices, dtype=np.uint16).tobytes(), dtype=np.uint16
        )
        counts = np.frombuffer(
            np.asarray(self.visit_counts, dtype=np.int64).tobytes(), dtype=np.int64
        )
        probabilities = np.frombuffer(
            np.asarray(self.probabilities, dtype=np.float32).tobytes(), dtype=np.float32
        )
        if (
            not indices.size
            or indices.shape != counts.shape
            or indices.shape != probabilities.shape
        ):
            raise MCTSSearchError("root policy arrays must be non-empty and aligned")
        if np.unique(indices).size != indices.size or (counts < 0).any():
            raise MCTSSearchError(
                "root policy indices must be unique and counts non-negative"
            )
        if int(counts.sum()) <= 0 or not np.isclose(
            float(probabilities.sum(dtype=np.float64)), 1.0, atol=1e-6, rtol=0.0
        ):
            raise MCTSSearchError(
                "root visit policy must contain normalized positive mass"
            )
        object.__setattr__(self, "action_indices", indices)
        object.__setattr__(self, "visit_counts", counts)
        object.__setattr__(self, "probabilities", probabilities)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RootVisitPolicy):
            return NotImplemented
        return (
            np.array_equal(self.action_indices, other.action_indices)
            and np.array_equal(self.visit_counts, other.visit_counts)
            and np.array_equal(self.probabilities, other.probabilities)
        )

    @property
    def legal_action_indices(self) -> NDArray[np.uint16]:
        return self.action_indices

    @property
    def visit_probabilities(self) -> NDArray[np.float32]:
        return self.probabilities

    def to_training_example(
        self, state: GameState, *, value_target: float
    ) -> TrainingExample:
        return TrainingExample(
            encode_state(state, perspective=state.side_to_move),
            self.action_indices,
            self.probabilities,
            value_target,
        )


@dataclass(frozen=True, slots=True)
class MCTSSearchResult:
    action: Action
    action_index: int
    policy: RootVisitPolicy
    root_value: float
    simulations: int
    root_visits: int
    total_nodes: int
    expanded_nodes: int
    evaluator_calls: int
    terminal_leaves: int
    depth_guard_leaves: int
    maximum_depth: int
    root_children: tuple[RootChildStatistics, ...]
    principal_variation: tuple[Action, ...]
    mcts_version: int = MCTS_VERSION
    elapsed_seconds: float = field(default=0.0, compare=False)

    @property
    def visit_policy(self) -> RootVisitPolicy:
        return self.policy

    @property
    def selected_action(self) -> Action:
        return self.action

    @property
    def selected_action_index(self) -> int:
        return self.action_index

    @property
    def simulation_count(self) -> int:
        return self.simulations

    @property
    def total_nodes_created(self) -> int:
        return self.total_nodes

    @property
    def model_calls(self) -> int:
        return self.evaluator_calls

    @property
    def maximum_depth_reached(self) -> int:
        return self.maximum_depth

    @property
    def root_child_statistics(self) -> tuple[RootChildStatistics, ...]:
        return self.root_children


@dataclass(slots=True)
class _Edge:
    action: Action
    action_index: int
    prior: float
    engine_order: int
    visits: int = 0
    value_sum: float = 0.0
    child: _Node | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class _Node:
    state: GameState
    edges: list[_Edge] = field(default_factory=list)
    expanded: bool = False
    visits: int = 0


def puct_score(
    *,
    mean_value: float,
    prior: float,
    parent_visits: int,
    child_visits: int,
    c_puct: float,
) -> float:
    """Return Q + c_puct P sqrt(N)/(1+n), with unvisited Q supplied as zero."""
    return mean_value + c_puct * prior * math.sqrt(parent_visits) / (1 + child_visits)


def _select_edge(node: _Node, c_puct: float) -> _Edge:
    return max(
        node.edges,
        key=lambda edge: (
            puct_score(
                mean_value=edge.mean_value,
                prior=edge.prior,
                parent_visits=node.visits,
                child_visits=edge.visits,
                c_puct=c_puct,
            ),
            -edge.engine_order,
        ),
    )


def _validated_evaluation(
    node: _Node, evaluator: PositionEvaluator
) -> PositionEvaluation:
    evaluation = evaluator.evaluate(node.state)
    if not isinstance(evaluation, PositionEvaluation):
        raise MCTSSearchError("evaluator must return PositionEvaluation")
    if evaluation.perspective is not node.state.side_to_move:
        raise MCTSSearchError("evaluation perspective must equal the side to move")
    if evaluation.model_architecture_version not in (
        None,
        MODEL_ARCHITECTURE_VERSION,
    ):
        raise MCTSSearchError("evaluation model architecture version mismatch")
    if evaluation.checkpoint_version not in (None, CHECKPOINT_VERSION):
        raise MCTSSearchError("evaluation checkpoint version mismatch")
    actions = legal_actions(node.state)
    expected = tuple(action_to_index(node.state, action) for action in actions)
    if evaluation.action_indices != expected:
        raise MCTSSearchError("evaluation indices must match legal engine action order")
    return evaluation


def _expand(node: _Node, evaluation: PositionEvaluation) -> None:
    actions = legal_actions(node.state)
    node.edges = [
        _Edge(action, index, prior, order)
        for order, (action, index, prior) in enumerate(
            zip(actions, evaluation.action_indices, evaluation.priors, strict=True)
        )
    ]
    node.expanded = True


def _terminal_value(state: GameState) -> float:
    if state.result is None:
        raise MCTSSearchError("terminal value requested for a non-terminal state")
    if state.result.outcome is Outcome.DRAW:
        return 0.0
    return 1.0 if state.result.winner is state.side_to_move else -1.0


def _select_action(
    edges: list[_Edge], temperature: float, rng: np.random.Generator
) -> _Edge:
    if temperature == 0.0:
        return max(edges, key=lambda edge: (edge.visits, -edge.engine_order))
    weights = np.asarray([edge.visits for edge in edges], dtype=np.float64) ** (
        1.0 / temperature
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise MCTSSearchError("temperature transform produced invalid visit weights")
    return edges[int(rng.choice(len(edges), p=weights / weights.sum()))]


def search_state(
    state: GameState, evaluator: PositionEvaluator, config: MCTSConfig | None = None
) -> MCTSSearchResult:
    """Run a fresh tree search. Root expansion is not counted as a simulation."""
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if state.result is not None:
        raise MCTSSearchError("cannot search a terminal root")
    active = config or MCTSConfig()
    rng = np.random.default_rng(active.random_seed)
    started = time.perf_counter()
    root = _Node(state)
    root_evaluation = _validated_evaluation(root, evaluator)
    evaluator_calls = 1
    _expand(root, root_evaluation)
    expanded_nodes = 1
    total_nodes = 1
    if active.root_noise_enabled:
        noise = rng.dirichlet(np.full(len(root.edges), active.root_dirichlet_alpha))
        fraction = active.root_noise_fraction
        for edge, item in zip(root.edges, noise, strict=True):
            edge.prior = (1.0 - fraction) * edge.prior + fraction * float(item)
        normalization = sum(edge.prior for edge in root.edges)
        for edge in root.edges:
            edge.prior /= normalization

    terminal_leaves = 0
    depth_guard_leaves = 0
    maximum_depth = 0
    for _simulation in range(active.simulations):
        node = root
        path: list[_Edge] = []
        depth = 0
        while node.state.result is None and node.expanded and depth < active.max_depth:
            edge = _select_edge(node, active.c_puct)
            path.append(edge)
            if edge.child is None:
                edge.child = _Node(apply_action(node.state, edge.action))
                total_nodes += 1
            node = edge.child
            depth += 1
        maximum_depth = max(maximum_depth, depth)
        if node.state.result is not None:
            value = _terminal_value(node.state)
            terminal_leaves += 1
        else:
            evaluation = _validated_evaluation(node, evaluator)
            evaluator_calls += 1
            value = evaluation.value
            if depth >= active.max_depth:
                depth_guard_leaves += 1
            else:
                _expand(node, evaluation)
                expanded_nodes += 1
        for edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.value_sum += value
            if edge.child is not None:
                edge.child.visits += 1
        root.visits += 1

    counts = np.asarray([edge.visits for edge in root.edges], dtype=np.int64)
    probabilities = counts.astype(np.float64) / int(counts.sum())
    policy = RootVisitPolicy(
        np.asarray([edge.action_index for edge in root.edges], dtype=np.uint16),
        counts,
        probabilities.astype(np.float32),
    )
    selected = _select_action(root.edges, active.temperature, rng)
    root_children = tuple(
        RootChildStatistics(
            edge.action,
            edge.action_index,
            edge.prior,
            edge.visits,
            edge.mean_value,
            edge.engine_order,
        )
        for edge in root.edges
    )
    pv: list[Action] = []
    pv_node = root
    while pv_node.expanded and pv_node.edges:
        pv_edge = max(pv_node.edges, key=lambda edge: (edge.visits, -edge.engine_order))
        if pv_edge.visits == 0:
            break
        pv.append(pv_edge.action)
        if pv_edge.child is None:
            break
        pv_node = pv_edge.child
    root_value = sum(edge.value_sum for edge in root.edges) / active.simulations
    return MCTSSearchResult(
        selected.action,
        selected.action_index,
        policy,
        root_value,
        active.simulations,
        root.visits,
        total_nodes,
        expanded_nodes,
        evaluator_calls,
        terminal_leaves,
        depth_guard_leaves,
        maximum_depth,
        root_children,
        tuple(pv),
        elapsed_seconds=time.perf_counter() - started,
    )


run_mcts = search_state
