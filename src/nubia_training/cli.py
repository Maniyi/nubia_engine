"""Testable command-line workflow for persistent training data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nubia_engine import Empire
from nubia_training.datasets import (
    build_dataset,
    inspect_dataset,
    records_from_directory,
    validate_dataset,
)
from nubia_training.errors import TrainingDataError
from nubia_training.generation import agent_spec, generate_series
from nubia_training.records import AgentSpec, write_raw_game
from nubia_training.replay import replay_game


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _agent(args: argparse.Namespace, side: str) -> AgentSpec:
    kind = getattr(args, f"agent_{side}")
    return agent_spec(
        kind,
        seed=args.base_seed if kind == "random" else None,
        depth=getattr(args, f"depth_{side}"),
        max_depth=getattr(args, f"max_depth_{side}"),
        node_limit=getattr(args, f"node_limit_{side}"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NUBIA deterministic training-data pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate-games", help="run agents and persist canonical raw games"
    )
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--games", type=_positive, default=2)
    generate.add_argument("--base-seed", type=int, default=42)
    generate.add_argument(
        "--first-player", choices=("A", "B", "alternate"), default="A"
    )
    generate.add_argument("--swap-sides", action="store_true")
    generate.add_argument("--max-plies", type=_positive, default=200)
    for side in ("a", "b"):
        generate.add_argument(
            f"--agent-{side}",
            choices=("random", "heuristic", "minimax", "iterative"),
            default="random",
        )
        generate.add_argument(f"--depth-{side}", type=_positive, default=1)
        generate.add_argument(f"--max-depth-{side}", type=_positive, default=2)
        generate.add_argument(f"--node-limit-{side}", type=_positive, default=1000)
    validate_games = commands.add_parser(
        "validate-games", help="read and replay every raw game"
    )
    validate_games.add_argument("--games-dir", type=Path, required=True)
    build = commands.add_parser(
        "build-dataset", help="convert validated games into compressed shards"
    )
    build.add_argument("--games-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--shard-size", type=_positive, default=256)
    validate = commands.add_parser("validate-dataset", help="verify a complete dataset")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--games-dir", type=Path)
    inspect = commands.add_parser(
        "inspect-dataset", help="print deterministic manifest summary"
    )
    inspect.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-games":
            alternate = args.first_player == "alternate"
            first = Empire.A if alternate else Empire(args.first_player)
            records = generate_series(
                _agent(args, "a"),
                _agent(args, "b"),
                game_count=args.games,
                base_seed=args.base_seed,
                first_player=first,
                alternate_first_player=alternate,
                swap_sides=args.swap_sides,
                max_plies=args.max_plies,
            )
            for record in records:
                write_raw_game(record, args.output_dir)
            completed = sum(record.status == "completed" for record in records)
            aborted = len(records) - completed
            print(
                f"generated {len(records)} games: "
                f"completed={completed} aborted={aborted}"
            )
        elif args.command == "validate-games":
            records = records_from_directory(args.games_dir)
            for record in records:
                replay_game(record)
            print(f"validated {len(records)} raw games")
        elif args.command == "build-dataset":
            records = records_from_directory(args.games_dir)
            manifest = build_dataset(
                records, args.output_dir, shard_size=args.shard_size
            )
            print(
                f"built dataset {manifest.dataset_id}: "
                f"examples={manifest.total_examples}"
            )
        elif args.command == "validate-dataset":
            manifest = validate_dataset(args.dataset_dir, args.games_dir)
            print(
                f"validated dataset {manifest.dataset_id}: "
                f"examples={manifest.total_examples}"
            )
        else:
            print(inspect_dataset(args.dataset_dir))
    except (TrainingDataError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
