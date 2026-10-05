"""Bounded iterative-deepening minimax with immutable telemetry."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from nubia_ai.evaluation import DEFAULT_WEIGHTS, EvaluationWeights
from nubia_ai.search import (
    SearchConfig,
    SearchResult,
    _Counters,
    _search_state,
    _validate_search_request,
)
from nubia_engine import Action, Empire, GameState

Clock = Callable[[], float]


class SearchStopReason(Enum):
    """The closed set of reasons an iterative search stops."""

    MAX_DEPTH_COMPLETED = "max_depth_completed"
    TIME_LIMIT = "time_limit"
    NODE_LIMIT = "node_limit"


@dataclass(frozen=True, slots=True)
class IterativeSearchConfig:
    """Immutable depth and optional whole-search budgets."""

    max_depth: int
    use_alpha_beta: bool = True
    time_limit_seconds: float | None = None
    node_limit: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            raise TypeError("maximum search depth must be an integer")
        if self.max_depth < 1:
            raise ValueError("maximum search depth must be at least one")
        if not isinstance(self.use_alpha_beta, bool):
            raise TypeError("use_alpha_beta must be a bool")
        if self.time_limit_seconds is not None:
            if isinstance(self.time_limit_seconds, bool) or not isinstance(
                self.time_limit_seconds, (int, float)
            ):
                raise TypeError("time limit must be a number")
            if (
                not math.isfinite(self.time_limit_seconds)
                or self.time_limit_seconds <= 0
            ):
                raise ValueError("time limit must be finite and strictly positive")
        if self.node_limit is not None:
            if isinstance(self.node_limit, bool) or not isinstance(
                self.node_limit, int
            ):
                raise TypeError("node limit must be an integer")
            if self.node_limit <= 0:
                raise ValueError("node limit must be strictly positive")


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One fully completed fixed-depth iteration."""

    depth: int
    result: SearchResult
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError("iteration depth must be an integer")
        if self.depth < 1:
            raise ValueError("iteration depth must be positive")
        if not isinstance(self.result, SearchResult):
            raise TypeError("iteration result must be a SearchResult")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("iteration elapsed time must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CumulativeSearchStats:
    """Counters across completed and interrupted iterations.

    Unlike fixed-search ``SearchStats``, zero nodes is valid when a time limit
    expires before the first iteration begins.
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
            raise TypeError("cumulative search statistics must be integers")
        if any(value < 0 for value in values):
            raise ValueError("cumulative search statistics cannot be negative")
        if self.leaves_evaluated > self.nodes_visited:
            raise ValueError("leaves evaluated cannot exceed nodes visited")


@dataclass(frozen=True, slots=True)
class IterativeSearchResult:
    """Decision from the deepest completed iteration, or a legal fallback."""

    action: Action
    score: int | None
    principal_variation: tuple[Action, ...]
    completed_depth: int
    stop_reason: SearchStopReason
    iterations: tuple[IterationRecord, ...]
    cumulative_stats: CumulativeSearchStats
    elapsed_seconds: float
    used_fallback: bool

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError("action must be an Action")
        if self.score is not None and (
            isinstance(self.score, bool) or not isinstance(self.score, int)
        ):
            raise TypeError("score must be an integer or None")
        if not isinstance(self.principal_variation, tuple) or any(
            not isinstance(action, Action) for action in self.principal_variation
        ):
            raise TypeError("principal_variation must be a tuple of Action values")
        if isinstance(self.completed_depth, bool) or not isinstance(
            self.completed_depth, int
        ):
            raise TypeError("completed_depth must be an integer")
        if not isinstance(self.stop_reason, SearchStopReason):
            raise TypeError("stop_reason must be a SearchStopReason")
        if not isinstance(self.iterations, tuple) or any(
            not isinstance(record, IterationRecord) for record in self.iterations
        ):
            raise TypeError("iterations must be a tuple of IterationRecord values")
        if not isinstance(self.cumulative_stats, CumulativeSearchStats):
            raise TypeError("cumulative_stats must be CumulativeSearchStats")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed time must be finite and non-negative")
        if not isinstance(self.used_fallback, bool):
            raise TypeError("used_fallback must be a bool")

        expected_depths = tuple(range(1, len(self.iterations) + 1))
        if tuple(record.depth for record in self.iterations) != expected_depths:
            raise ValueError("completed iteration depths must be consecutive from one")
        if self.completed_depth != len(self.iterations):
            raise ValueError("completed_depth must equal the final iteration depth")
        if self.used_fallback:
            if self.completed_depth != 0 or self.score is not None:
                raise ValueError("fallback results have depth zero and no score")
            if self.principal_variation or self.iterations:
                raise ValueError("fallback results have no PV or completed iterations")
            if self.stop_reason is SearchStopReason.MAX_DEPTH_COMPLETED:
                raise ValueError("fallback requires an exhausted budget")
        else:
            if not self.iterations or self.score is None:
                raise ValueError("searched results require a completed iteration")
            deepest = self.iterations[-1].result
            if (
                self.action,
                self.score,
                self.principal_variation,
            ) != (deepest.action, deepest.score, deepest.principal_variation):
                raise ValueError("final decision must come from the deepest iteration")


class _SearchInterrupted(Exception):
    """Private cooperative unwind carrying the first exhausted budget."""

    def __init__(self, reason: SearchStopReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(slots=True)
class _BudgetController:
    node_limit: int | None
    deadline: float | None
    clock: Clock
    nodes_visited: int = 0

    def now(self) -> float:
        value = self.clock()
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("clock must return a number")
        if not math.isfinite(value):
            raise ValueError("clock must return a finite value")
        return float(value)

    def check(self) -> None:
        """Check node before time so simultaneous exhaustion is deterministic."""

        if self.node_limit is not None and self.nodes_visited >= self.node_limit:
            raise _SearchInterrupted(SearchStopReason.NODE_LIMIT)
        if self.deadline is not None and self.now() >= self.deadline:
            raise _SearchInterrupted(SearchStopReason.TIME_LIMIT)

    def before_node(self) -> None:
        self.check()
        self.nodes_visited += 1


def _cumulative_stats(counters: tuple[_Counters, ...]) -> CumulativeSearchStats:
    return CumulativeSearchStats(
        sum(counter.nodes for counter in counters),
        sum(counter.leaves for counter in counters),
        sum(counter.cutoffs for counter in counters),
        max((counter.max_depth for counter in counters), default=0),
    )


def iterative_search_state(
    state: GameState,
    perspective: Empire,
    config: IterativeSearchConfig,
    weights: EvaluationWeights = DEFAULT_WEIGHTS,
    *,
    clock: Clock = time.monotonic,
) -> IterativeSearchResult:
    """Search depths 1..max_depth under cumulative cooperative budgets.

    Node exhaustion takes precedence when node and time limits are unavailable
    at the same check. A wall-clock-limited completed depth may vary by machine
    and load.
    """

    if not isinstance(config, IterativeSearchConfig):
        raise TypeError("config must be an IterativeSearchConfig")
    if not callable(clock):
        raise TypeError("clock must be callable")

    start = clock()
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise TypeError("clock must return a number")
    if not math.isfinite(start):
        raise ValueError("clock must return a finite value")
    start = float(start)
    deadline = (
        start + config.time_limit_seconds
        if config.time_limit_seconds is not None
        else None
    )
    controller = _BudgetController(config.node_limit, deadline, clock)

    # Validate terminal/wrong-turn roots and capture fallback order before any
    # exhausted budget can produce a result.
    root_actions = _validate_search_request(
        state,
        perspective,
        SearchConfig(config.max_depth, config.use_alpha_beta),
        weights,
    )
    records: list[IterationRecord] = []
    all_counters: list[_Counters] = []
    stop_reason = SearchStopReason.MAX_DEPTH_COMPLETED

    for depth in range(1, config.max_depth + 1):
        try:
            controller.check()
            iteration_start = controller.now()
            counters = _Counters()
            all_counters.append(counters)
            result = _search_state(
                state,
                perspective,
                SearchConfig(depth, config.use_alpha_beta),
                weights,
                counters=counters,
                before_node=controller.before_node,
            )
        except _SearchInterrupted as interrupted:
            stop_reason = interrupted.reason
            break
        iteration_end = controller.now()
        records.append(
            IterationRecord(depth, result, max(0.0, iteration_end - iteration_start))
        )

    elapsed = max(0.0, controller.now() - start)
    cumulative = _cumulative_stats(tuple(all_counters))
    if records:
        deepest = records[-1].result
        return IterativeSearchResult(
            deepest.action,
            deepest.score,
            deepest.principal_variation,
            records[-1].depth,
            stop_reason,
            tuple(records),
            cumulative,
            elapsed,
            False,
        )
    return IterativeSearchResult(
        root_actions[0],
        None,
        (),
        0,
        stop_reason,
        (),
        cumulative,
        elapsed,
        True,
    )
