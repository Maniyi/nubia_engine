"""Deterministic fixed-depth minimax search with optional alpha-beta pruning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nubia_ai.agent import AgentError, NoActionAvailableError, WrongTurnError
from nubia_ai.evaluation import DEFAULT_WEIGHTS, EvaluationWeights, evaluate_state
from nubia_engine import Action, Empire, GameState, apply_action, legal_actions


class SearchInvariantError(AgentError):
    """Raised when a non-terminal engine state has no legal actions."""


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Immutable fixed-depth search settings; depth counts actions from the root."""

    depth: int
    use_alpha_beta: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError("search depth must be an integer")
        if self.depth < 1:
            raise ValueError("search depth must be at least one")
        if not isinstance(self.use_alpha_beta, bool):
            raise TypeError("use_alpha_beta must be a bool")


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Counters for one search call.

    The root and every visited successor count as nodes. A leaf is evaluated
    because it is terminal or reaches the horizon. A cutoff is one node where
    alpha-beta skips remaining siblings. ``max_depth_reached`` counts actions
    from the root.
    """

    nodes_visited: int
    leaves_evaluated: int
    alpha_beta_cutoffs: int
    max_depth_reached: int

    def __post_init__(self) -> None:
        values = (
            self.nodes_visited,
            self.leaves_evaluated,
            self.alpha_beta_cutoffs,
            self.max_depth_reached,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("search statistics must be integers")
        if any(value < 0 for value in values):
            raise ValueError("search statistics cannot be negative")
        if self.nodes_visited < 1:
            raise ValueError("a search must visit its root node")
        if self.leaves_evaluated > self.nodes_visited:
            raise ValueError("leaves evaluated cannot exceed nodes visited")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The chosen action, exact root score, principal variation, and counters."""

    action: Action
    score: int
    principal_variation: tuple[Action, ...]
    stats: SearchStats

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError("action must be an Action")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise TypeError("score must be an integer")
        if not isinstance(self.principal_variation, tuple) or any(
            not isinstance(action, Action) for action in self.principal_variation
        ):
            raise TypeError("principal_variation must be a tuple of Action values")
        if not self.principal_variation:
            raise ValueError("principal_variation cannot be empty")
        if self.principal_variation[0] != self.action:
            raise ValueError("selected action must begin the principal variation")
        if not isinstance(self.stats, SearchStats):
            raise TypeError("stats must be SearchStats")


@dataclass(slots=True)
class _Counters:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0
    max_depth: int = 0


def _validate_search_request(
    state: GameState,
    perspective: Empire,
    config: SearchConfig,
    weights: EvaluationWeights,
) -> tuple[Action, ...]:
    """Validate one search request and return its stable root action order."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(perspective, Empire):
        raise TypeError("perspective must be an Empire")
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if not isinstance(weights, EvaluationWeights):
        raise TypeError("weights must be EvaluationWeights")
    if state.result is not None:
        raise NoActionAvailableError("cannot search a terminal root state")
    if state.side_to_move is not perspective:
        raise WrongTurnError(
            f"search perspective is Empire {perspective.value}, "
            f"but Empire {state.side_to_move.value} is to move"
        )
    root_actions = legal_actions(state)
    if not root_actions:
        raise SearchInvariantError(
            "the engine returned no legal actions for a non-terminal root state"
        )
    return root_actions


def _search_state(
    state: GameState,
    perspective: Empire,
    config: SearchConfig,
    weights: EvaluationWeights,
    *,
    counters: _Counters | None = None,
    before_node: Callable[[], None] | None = None,
) -> SearchResult:
    """Internal fixed search with an optional cooperative node-entry hook."""

    root_actions = _validate_search_request(state, perspective, config, weights)
    active_counters = counters if counters is not None else _Counters()
    if before_node is not None:
        before_node()
    active_counters.nodes += 1
    return _run_search(
        state,
        perspective,
        config,
        weights,
        root_actions,
        active_counters,
        before_node,
    )


def search_state(
    state: GameState,
    perspective: Empire,
    config: SearchConfig,
    weights: EvaluationWeights = DEFAULT_WEIGHTS,
) -> SearchResult:
    """Search ``state`` from one fixed empire's perspective."""
    return _search_state(state, perspective, config, weights)


def _run_search(
    state: GameState,
    perspective: Empire,
    config: SearchConfig,
    weights: EvaluationWeights,
    root_actions: tuple[Action, ...],
    counters: _Counters,
    before_node: Callable[[], None] | None,
) -> SearchResult:
    """Implementation body split out to keep public validation unchanged."""

    def visit(
        current: GameState,
        remaining_depth: int,
        depth_from_root: int,
        alpha: int | None,
        beta: int | None,
    ) -> tuple[int, tuple[Action, ...]]:
        if before_node is not None:
            before_node()
        counters.nodes += 1
        counters.max_depth = max(counters.max_depth, depth_from_root)
        if current.result is not None or remaining_depth == 0:
            counters.leaves += 1
            return evaluate_state(current, perspective, weights).total, ()

        actions = legal_actions(current)
        if not actions:
            raise SearchInvariantError(
                "the engine returned no legal actions for a non-terminal state"
            )
        maximizing = current.side_to_move is perspective
        best_score: int | None = None
        best_pv: tuple[Action, ...] = ()
        for index, action in enumerate(actions):
            child_score, child_pv = visit(
                apply_action(current, action),
                remaining_depth - 1,
                depth_from_root + 1,
                alpha,
                beta,
            )
            if (
                best_score is None
                or (maximizing and child_score > best_score)
                or (not maximizing and child_score < best_score)
            ):
                best_score = child_score
                best_pv = (action, *child_pv)
            if config.use_alpha_beta:
                if maximizing and (alpha is None or best_score > alpha):
                    alpha = best_score
                if not maximizing and (beta is None or best_score < beta):
                    beta = best_score
                if alpha is not None and beta is not None and alpha >= beta:
                    if index + 1 < len(actions):
                        counters.cutoffs += 1
                    break
        if best_score is None:  # Defensive: actions was checked above.
            raise SearchInvariantError("search node unexpectedly had no actions")
        return best_score, best_pv

    best_score: int | None = None
    best_pv: tuple[Action, ...] = ()
    alpha: int | None = None
    for action in root_actions:
        child_score, child_pv = visit(
            apply_action(state, action), config.depth - 1, 1, alpha, None
        )
        if best_score is None or child_score > best_score:
            best_score = child_score
            best_pv = (action, *child_pv)
        if config.use_alpha_beta and (alpha is None or best_score > alpha):
            alpha = best_score

    if best_score is None or not best_pv:  # Defensive: root actions was non-empty.
        raise SearchInvariantError("root search unexpectedly produced no result")
    return SearchResult(
        best_pv[0],
        best_score,
        best_pv,
        SearchStats(
            counters.nodes,
            counters.leaves,
            counters.cutoffs,
            counters.max_depth,
        ),
    )
