"""Command-line entry point for bounded neural training and inspection."""

from __future__ import annotations

import argparse
import json
import sys
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
from nubia_training.neural.checkpoints import (
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from nubia_training.neural.data import ShardDataset
from nubia_training.neural.errors import NeuralError
from nubia_training.neural.evaluator import NeuralPositionEvaluator
from nubia_training.neural.iteration import (
    IterationConfig,
    inspect_iteration,
    run_iteration,
)
from nubia_training.neural.mcts import MCTSConfig, search_state
from nubia_training.neural.model import (
    ModelConfig,
    PolicyValueNetwork,
    model_memory_report,
)
from nubia_training.neural.self_play import (
    SelfPlayConfig,
    generate_self_play_corpus,
    inspect_self_play,
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
    train.add_argument("--amp", action="store_true")
    train.add_argument("--source-checkpoint", type=Path)
    train.add_argument("--resume", action="store_true")
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

    self_play = subparsers.add_parser(
        "self-play", help="generate a bounded neural-MCTS self-play corpus"
    )
    self_play.add_argument("checkpoint", type=Path)
    self_play.add_argument("output_dir", type=Path)
    _self_play_arguments(self_play)

    inspect_self_play_parser = subparsers.add_parser(
        "inspect-self-play", help="verify and inspect a self-play corpus"
    )
    inspect_self_play_parser.add_argument("corpus", type=Path)

    iteration = subparsers.add_parser(
        "run-iteration", help="run exactly one bounded learning iteration"
    )
    iteration.add_argument("source_checkpoint", type=Path)
    iteration.add_argument("output_dir", type=Path)
    _self_play_arguments(iteration)
    iteration.add_argument("--batch-size", type=int, default=4)
    iteration.add_argument("--epochs", type=int, default=1)
    iteration.add_argument("--max-steps", type=int, default=10)
    iteration.add_argument("--learning-rate", type=float, default=1e-3)
    iteration.add_argument("--gradient-accumulation", type=int, default=1)
    iteration.add_argument("--gradient-clip", type=float, default=1.0)
    iteration.add_argument("--amp", action="store_true")
    iteration.add_argument("--iteration-id")
    iteration.add_argument("--resume-checkpoint", type=Path)

    inspect_iteration_parser = subparsers.add_parser(
        "inspect-iteration", help="verify and inspect an iteration summary"
    )
    inspect_iteration_parser.add_argument("iteration", type=Path)
    return parser


def _self_play_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--late-temperature", type=float, default=0.0)
    parser.add_argument("--temperature-cutoff", type=int, default=20)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--no-root-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-game-plies", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument(
        "--first-player", choices=("A", "B", "alternate"), default="alternate"
    )
    parser.add_argument("--shard-size", type=int, default=256)


def _self_play_config(
    args: argparse.Namespace, checkpoint: Path, output_dir: Path
) -> SelfPlayConfig:
    return SelfPlayConfig(
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        game_count=args.games,
        simulations=args.simulations,
        c_puct=args.c_puct,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        root_noise_enabled=not args.no_root_noise,
        temperature_cutoff_plies=args.temperature_cutoff,
        early_temperature=args.temperature,
        late_temperature=args.late_temperature,
        max_game_plies=args.max_game_plies,
        random_seed=args.seed,
        device=args.device,
        first_player_schedule=args.first_player,
        shard_size=args.shard_size,
        max_search_depth=args.max_depth,
    )


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
                amp_enabled=args.amp,
            )
            dataset = ShardDataset(args.dataset)
            seed_training(training_config.random_seed)
            training_checkpoint: LoadedCheckpoint | None = None
            if args.resume and args.source_checkpoint is None:
                raise ValueError("--resume requires --source-checkpoint")
            if args.source_checkpoint is None:
                model = PolicyValueNetwork(model_config)
                optimizer = None
            else:
                training_checkpoint = load_checkpoint(
                    args.source_checkpoint,
                    expected_model_config=model_config,
                    expected_dataset_id=dataset.dataset_id if args.resume else None,
                    expected_dataset_fingerprint=(
                        dataset.dataset_fingerprint if args.resume else None
                    ),
                )
                model = training_checkpoint.model
                optimizer = (
                    training_checkpoint.restore_optimizer() if args.resume else None
                )
            result = train_model(
                model,
                dataset,
                training_config,
                optimizer=optimizer,
                initial_completed_epochs=(
                    int(training_checkpoint.metadata["completed_epoch"])
                    if training_checkpoint is not None and args.resume
                    else 0
                ),
                initial_optimizer_steps=(
                    int(training_checkpoint.metadata["optimizer_step"])
                    if training_checkpoint is not None and args.resume
                    else 0
                ),
                initial_examples_seen=(
                    int(training_checkpoint.metadata["examples_seen"])
                    if training_checkpoint is not None and args.resume
                    else 0
                ),
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = args.output_dir / (
                "candidate.pt"
                if args.source_checkpoint is not None
                else "checkpoint.pt"
            )
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
                        "initial_metrics": result.summary.initial_metrics,
                        "amp_enabled": result.summary.amp_enabled,
                        "peak_cuda_memory_allocated": (
                            result.summary.peak_cuda_memory_allocated
                        ),
                        "peak_cuda_memory_reserved": (
                            result.summary.peak_cuda_memory_reserved
                        ),
                    },
                }
            )
            return 0
        if args.command == "self-play":
            self_play_config = _self_play_config(args, args.checkpoint, args.output_dir)

            def progress(index: int, game: object) -> None:
                del game
                print(
                    "completed self-play game "
                    f"{index + 1}/{self_play_config.game_count}",
                    file=sys.stderr,
                )

            corpus = generate_self_play_corpus(self_play_config, progress=progress)
            _print_json(corpus.manifest)
            return 0
        if args.command == "inspect-self-play":
            _print_json(inspect_self_play(args.corpus))
            return 0
        if args.command == "run-iteration":
            self_play = _self_play_config(
                args, args.source_checkpoint, args.output_dir / "self_play"
            )
            training = TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                gradient_accumulation_steps=args.gradient_accumulation,
                gradient_clip_norm=args.gradient_clip,
                random_seed=args.seed,
                device=args.device,
                amp_enabled=args.amp,
            )
            iteration_result = run_iteration(
                IterationConfig(
                    args.source_checkpoint,
                    args.output_dir,
                    self_play,
                    training,
                    iteration_id=args.iteration_id,
                    resume_checkpoint=args.resume_checkpoint,
                )
            )
            _print_json(iteration_result.summary)
            return 0
        if args.command == "inspect-iteration":
            _print_json(inspect_iteration(args.iteration))
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
