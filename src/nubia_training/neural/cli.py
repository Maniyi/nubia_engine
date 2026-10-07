"""Command-line entry point for bounded neural training and inspection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nubia_ai import (
    Agent,
    HeuristicAgent,
    MatchLimitExceededError,
    RandomAgent,
    run_match,
)
from nubia_engine import Empire, apply_action, create_initial_state, format_action
from nubia_training.errors import TrainingDataError
from nubia_training.neural.agents import MCTSAgent, NeuralPolicyAgent
from nubia_training.neural.checkpoints import load_checkpoint, save_checkpoint
from nubia_training.neural.data import ShardDataset
from nubia_training.neural.errors import NeuralError
from nubia_training.neural.evaluator import NeuralPositionEvaluator
from nubia_training.neural.mcts import MCTSConfig, search_state
from nubia_training.neural.model import (
    ModelConfig,
    PolicyValueNetwork,
    model_memory_report,
)
from nubia_training.neural.training import TrainingConfig, seed_training, train_model


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trunk-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--policy-embedding-dim", type=int, default=16)
    parser.add_argument("--value-hidden-dim", type=int, default=128)
    parser.add_argument("--normalization-groups", type=int, default=8)


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        trunk_channels=args.trunk_channels,
        residual_blocks=args.residual_blocks,
        policy_embedding_dim=args.policy_embedding_dim,
        value_hidden_dim=args.value_hidden_dim,
        normalization_groups=args.normalization_groups,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nubia-neural")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_model = subparsers.add_parser(
        "inspect-model", help="print deterministic architecture and memory metadata"
    )
    _model_arguments(inspect_model)

    inspect_checkpoint = subparsers.add_parser(
        "inspect-checkpoint", help="verify and summarize a checkpoint"
    )
    inspect_checkpoint.add_argument("checkpoint", type=Path)

    train = subparsers.add_parser("train", help="run a bounded training job")
    train.add_argument("dataset", type=Path)
    train.add_argument("output_dir", type=Path)
    train.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--gradient-accumulation", type=int, default=1)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--no-shuffle", action="store_true")
    _model_arguments(train)

    search = subparsers.add_parser("search", help="search the standard initial state")
    search.add_argument("checkpoint", type=Path)
    _search_arguments(search)
    search.add_argument("--first-player", choices=("A", "B"), default="A")

    benchmark = subparsers.add_parser(
        "benchmark", help="run a tiny checkpoint-backed agent comparison"
    )
    benchmark.add_argument("checkpoint", type=Path)
    _search_arguments(benchmark)
    benchmark.add_argument(
        "--opponent", choices=("random", "heuristic", "neural"), default="random"
    )
    benchmark.add_argument("--games", type=int, default=1)
    benchmark.add_argument("--max-plies", type=int, default=40)
    benchmark.add_argument(
        "--first-player", choices=("A", "B", "alternate"), default="alternate"
    )
    benchmark.add_argument("--swap-sides", action="store_true")
    return parser


def _search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--root-noise", action="store_true")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None)


def _search_config(args: argparse.Namespace) -> MCTSConfig:
    return MCTSConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        max_depth=args.max_depth,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        random_seed=args.seed,
        root_noise_enabled=args.root_noise,
    )


def _search_output(args: argparse.Namespace) -> dict[str, object]:
    evaluator = NeuralPositionEvaluator.from_checkpoint(
        args.checkpoint, device=args.device
    )
    state = create_initial_state(Empire(args.first_player))
    result = search_state(state, evaluator, _search_config(args))
    pv_state = state
    principal_variation: list[str] = []
    for action in result.principal_variation:
        principal_variation.append(format_action(pv_state, action))
        pv_state = apply_action(pv_state, action)
    return {
        "action": format_action(state, result.action),
        "action_index": result.action_index,
        "root_value": result.root_value,
        "simulations": result.simulations,
        "evaluator_calls": result.evaluator_calls,
        "maximum_depth": result.maximum_depth,
        "principal_variation": principal_variation,
        "root_children": [
            {
                "action": format_action(state, child.action),
                "action_index": child.action_index,
                "engine_order": child.engine_order,
                "mean_value": child.mean_value,
                "prior": child.prior,
                "visits": child.visits,
            }
            for child in result.root_children
        ],
    }


def _benchmark_output(args: argparse.Namespace) -> dict[str, object]:
    if args.games <= 0 or args.max_plies <= 0:
        raise ValueError("games and max-plies must be positive")
    evaluator = NeuralPositionEvaluator.from_checkpoint(
        args.checkpoint, device=args.device
    )
    config = _search_config(args)
    games: list[dict[str, object]] = []
    for game_index in range(args.games):
        if args.first_player == "alternate":
            first = Empire.A if game_index % 2 == 0 else Empire.B
        else:
            first = Empire(args.first_player)
        mcts_empire = Empire.B if args.swap_sides and game_index % 2 else Empire.A
        opponent_empire = Empire.B if mcts_empire is Empire.A else Empire.A
        mcts = MCTSAgent(mcts_empire, evaluator, config)
        opponent: Agent
        if args.opponent == "random":
            opponent = RandomAgent(opponent_empire, seed=(args.seed or 0) + game_index)
        elif args.opponent == "heuristic":
            opponent = HeuristicAgent(opponent_empire)
        else:
            opponent = NeuralPolicyAgent(opponent_empire, evaluator)
        match = run_match(
            create_initial_state(first),
            {mcts_empire: mcts, opponent_empire: opponent},
            max_plies=args.max_plies,
        )
        games.append(
            {
                "first_player": first.value,
                "mcts_empire": mcts_empire.value,
                "outcome": match.result.outcome.value,
                "winner": None
                if match.result.winner is None
                else match.result.winner.value,
                "plies": match.total_plies,
            }
        )
    return {"games": games, "game_count": len(games)}


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-model":
            config = _model_config(args)
            _print_json(
                {
                    "model_config": config.to_dict(),
                    "memory": model_memory_report(config),
                }
            )
            return 0
        if args.command == "inspect-checkpoint":
            loaded = load_checkpoint(args.checkpoint)
            _print_json(
                {
                    "checkpoint_bytes": loaded.path.stat().st_size,
                    "metadata": loaded.metadata,
                    "sha256": loaded.sha256,
                }
            )
            return 0
        if args.command == "train":
            model_config = _model_config(args)
            training_config = TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip,
                gradient_accumulation_steps=args.gradient_accumulation,
                shuffle=not args.no_shuffle,
                random_seed=args.seed,
                device=args.device,
                data_loader_workers=args.workers,
            )
            dataset = ShardDataset(args.dataset)
            seed_training(training_config.random_seed)
            model = PolicyValueNetwork(model_config)
            result = train_model(model, dataset, training_config)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = args.output_dir / "checkpoint.pt"
            checksum = save_checkpoint(
                checkpoint,
                model,
                training_config,
                dataset_id=dataset.dataset_id,
                dataset_fingerprint=dataset.dataset_fingerprint,
                completed_epoch=result.summary.completed_epochs,
                optimizer_step=result.summary.optimizer_steps,
                examples_seen=result.summary.examples_seen,
                latest_metrics=result.summary.latest_metrics,
                optimizer=result.optimizer,
            )
            _print_json(
                {
                    "checkpoint": str(checkpoint),
                    "sha256": checksum,
                    "summary": {
                        "completed_epochs": result.summary.completed_epochs,
                        "examples_seen": result.summary.examples_seen,
                        "optimizer_steps": result.summary.optimizer_steps,
                        "selected_device": result.summary.selected_device,
                        "latest_metrics": result.summary.latest_metrics,
                    },
                }
            )
            return 0
        if args.command == "search":
            _print_json(_search_output(args))
            return 0
        if args.command == "benchmark":
            _print_json(_benchmark_output(args))
            return 0
        parser.error("unknown command")
    except (
        MatchLimitExceededError,
        NeuralError,
        TrainingDataError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(2, f"nubia-neural: error: {error}\n")
    return 2
