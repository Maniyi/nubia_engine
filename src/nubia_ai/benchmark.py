"""Deterministic baseline matchup execution and CLI."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nubia_ai.agent import Agent, MatchLimitExceededError
from nubia_ai.heuristic_agent import HeuristicAgent
from nubia_ai.match import MatchResult, run_match
from nubia_ai.random_agent import RandomAgent
from nubia_engine import Empire, GameState, Outcome, create_initial_state

AgentFactory = Callable[[Empire, int], Agent]
StateFactory = Callable[[], GameState]


@dataclass(frozen=True, slots=True)
class MatchupSummary:
    """Immutable aggregate counts for a deterministic series of matches."""

    games: int
    a_wins: int
    b_wins: int
    draws: int
    wins_by_agent: tuple[tuple[str, int], ...]
    results_by_reason: tuple[tuple[str, int], ...]
    min_plies: int
    max_plies: int
    average_plies: float
    base_seed: int
    game_seeds: tuple[tuple[int, int], ...]
    swap_sides: bool
    agent_labels: tuple[str, str]


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def summarize_matches(
    matches: Sequence[MatchResult],
    *,
    base_seed: int,
    game_seeds: Sequence[tuple[int, int]],
    swap_sides: bool,
    agent_labels: tuple[str, str],
) -> MatchupSummary:
    """Summarize a non-empty sequence without making statistical claims."""

    if not matches:
        raise ValueError("at least one match is required")
    if len(matches) != len(game_seeds):
        raise ValueError("each match must have one seed pair")
    empire_wins: Counter[Empire] = Counter()
    agent_wins: Counter[str] = Counter()
    agent_names: set[str] = set()
    reasons: Counter[str] = Counter()
    draws = 0
    plies: list[int] = []
    for match in matches:
        agent_names.update(match.names_by_empire.values())
        plies.append(match.total_plies)
        for reason in match.result.reasons:
            reasons[reason.value] += 1
        if match.result.outcome is Outcome.DRAW:
            draws += 1
        else:
            winner = match.result.winner
            if winner is None:  # Defensive: GameResult validates this invariant.
                raise ValueError("winning match has no winner")
            empire_wins[winner] += 1
            agent_wins[match.names_by_empire[winner]] += 1
    return MatchupSummary(
        len(matches),
        empire_wins[Empire.A],
        empire_wins[Empire.B],
        draws,
        tuple((name, agent_wins[name]) for name in sorted(agent_names)),
        tuple(sorted(reasons.items())),
        min(plies),
        max(plies),
        sum(plies) / len(plies),
        base_seed,
        tuple(game_seeds),
        swap_sides,
        agent_labels,
    )


def run_matchup(
    first_factory: AgentFactory,
    second_factory: AgentFactory,
    *,
    games: int,
    seed: int,
    max_plies: int,
    swap_sides: bool = False,
    initial_state_factory: StateFactory | None = None,
    agent_labels: tuple[str, str] = ("agent-1", "agent-2"),
) -> MatchupSummary:
    """Run fresh-agent games with deterministic per-competitor seed schedules."""

    if isinstance(games, bool) or not isinstance(games, int):
        raise TypeError("games must be an integer")
    if games <= 0:
        raise ValueError("games must be positive")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if len(agent_labels) != 2 or any(not label for label in agent_labels):
        raise ValueError("agent_labels must contain two non-empty names")

    make_state = initial_state_factory or (lambda: create_initial_state(Empire.A))
    matches: list[MatchResult] = []
    seeds: list[tuple[int, int]] = []
    for game_index in range(games):
        first_seed = seed + game_index * 2
        second_seed = first_seed + 1
        seeds.append((first_seed, second_seed))
        swapped = swap_sides and game_index % 2 == 1
        first_empire = Empire.B if swapped else Empire.A
        second_empire = Empire.A if swapped else Empire.B
        players = {
            first_empire: first_factory(first_empire, first_seed),
            second_empire: second_factory(second_empire, second_seed),
        }
        matches.append(run_match(make_state(), players, max_plies=max_plies))
    return summarize_matches(
        matches,
        base_seed=seed,
        game_seeds=seeds,
        swap_sides=swap_sides,
        agent_labels=agent_labels,
    )


def _factory(kind: str, name: str | None = None) -> AgentFactory:
    if kind == "random":
        return lambda empire, seed: RandomAgent(
            empire, seed=seed, name=name or "RandomAgent"
        )
    if kind == "heuristic":
        return lambda empire, _seed: HeuristicAgent(
            empire, name=name or "HeuristicAgent"
        )
    raise ValueError(f"unknown agent kind: {kind}")


def format_summary(summary: MatchupSummary) -> str:
    """Render stable concise text suitable for experiments and snapshots."""

    wins = ", ".join(f"{name}={count}" for name, count in summary.wins_by_agent)
    reasons = ", ".join(
        f"{reason}={count}" for reason, count in summary.results_by_reason
    )
    return "\n".join(
        (
            f"agents: {summary.agent_labels[0]} vs {summary.agent_labels[1]}",
            f"games={summary.games} seed={summary.base_seed} "
            f"swap_sides={summary.swap_sides}",
            f"empire wins: A={summary.a_wins} B={summary.b_wins} draws={summary.draws}",
            f"agent wins: {wins or 'none'}",
            f"termination reasons: {reasons or 'none'}",
            f"plies: min={summary.min_plies} max={summary.max_plies} "
            f"average={summary.average_plies:.2f}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the noninteractive baseline benchmark CLI."""

    parser = argparse.ArgumentParser(description="Benchmark baseline NUBIA agents")
    parser.add_argument("--agent-a", choices=("random", "heuristic"), default="random")
    parser.add_argument(
        "--agent-b", choices=("random", "heuristic"), default="heuristic"
    )
    parser.add_argument("--games", type=_positive, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plies", type=_positive, default=1000)
    parser.add_argument("--swap-sides", action="store_true")
    args = parser.parse_args(argv)
    labels = (f"agent-a:{args.agent_a}", f"agent-b:{args.agent_b}")
    try:
        summary = run_matchup(
            _factory(args.agent_a, labels[0]),
            _factory(args.agent_b, labels[1]),
            games=args.games,
            seed=args.seed,
            max_plies=args.max_plies,
            swap_sides=args.swap_sides,
            agent_labels=labels,
        )
    except MatchLimitExceededError as error:
        parser.exit(1, f"benchmark failed: {error}\n")
    print(format_summary(summary))
    return 0
