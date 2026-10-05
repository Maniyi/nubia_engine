"""Public API for NUBIA baseline computer players."""

from nubia_ai.agent import (
    Agent,
    AgentConfigurationError,
    AgentError,
    IllegalAgentActionError,
    MatchLimitExceededError,
    NoActionAvailableError,
    WrongTurnError,
)
from nubia_ai.benchmark import MatchupSummary, run_matchup, summarize_matches
from nubia_ai.evaluation import (
    DEFAULT_WEIGHTS,
    EvaluationBreakdown,
    EvaluationWeights,
    evaluate_state,
)
from nubia_ai.heuristic_agent import HeuristicAgent, ScoredAction
from nubia_ai.match import MatchResult, run_match
from nubia_ai.random_agent import RandomAgent

__all__ = [
    "DEFAULT_WEIGHTS",
    "Agent",
    "AgentConfigurationError",
    "AgentError",
    "EvaluationBreakdown",
    "EvaluationWeights",
    "HeuristicAgent",
    "IllegalAgentActionError",
    "MatchLimitExceededError",
    "MatchResult",
    "MatchupSummary",
    "NoActionAvailableError",
    "RandomAgent",
    "ScoredAction",
    "WrongTurnError",
    "evaluate_state",
    "run_match",
    "run_matchup",
    "summarize_matches",
]
