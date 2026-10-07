from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nubia_engine import Empire, GameState, create_initial_state
from nubia_training import legal_action_indices, load_dataset, replay_game
from nubia_training.neural import (
    IterationConfig,
    ModelConfig,
    NeuralPositionEvaluator,
    PolicyValueNetwork,
    PositionEvaluation,
    SelfPlayConfig,
    ShardDataset,
    TrainingConfig,
    build_self_play_dataset,
    generate_self_play_corpus,
    inspect_iteration,
    load_checkpoint,
    load_self_play_corpus,
    play_self_play_game,
    run_iteration,
    save_checkpoint,
    self_play_examples,
    train_model,
)
from nubia_training.neural.errors import SelfPlayError


class UniformEvaluator:
    def evaluate(self, state: GameState) -> PositionEvaluation:
        indices = tuple(int(item) for item in legal_action_indices(state))
        return PositionEvaluation(
            indices,
            tuple(1.0 / len(indices) for _ in indices),
            0.75,
            state.side_to_move,
        )


def _model_config() -> ModelConfig:
    return ModelConfig(
        trunk_channels=8,
        residual_blocks=1,
        policy_embedding_dim=2,
        value_hidden_dim=8,
        normalization_groups=2,
    )


def _checkpoint(path: Path) -> Path:
    torch.manual_seed(44)
    save_checkpoint(
        path,
        PolicyValueNetwork(_model_config()),
        TrainingConfig(),
        dataset_id="source",
        dataset_fingerprint="0" * 64,
        completed_epoch=0,
        optimizer_step=0,
        examples_seen=0,
        latest_metrics={},
    )
    return path


def _config(tmp_path: Path, **changes: object) -> SelfPlayConfig:
    values: dict[str, object] = {
        "checkpoint_path": tmp_path / "source.pt",
        "output_dir": tmp_path / "corpus",
        "game_count": 1,
        "simulations": 1,
        "root_noise_enabled": False,
        "temperature_cutoff_plies": 0,
        "max_game_plies": 80,
        "random_seed": 19,
    }
    values.update(changes)
    return SelfPlayConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("game_count", 0),
        ("game_count", True),
        ("simulations", 0),
        ("early_temperature", -1.0),
        ("temperature_cutoff_plies", -1),
        ("max_game_plies", 0),
        ("root_noise_fraction", 1.1),
        ("root_noise_enabled", 1),
        ("version", 2),
    ],
)
def test_self_play_configuration_rejects_invalid_values(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(SelfPlayError):
        _config(tmp_path, **{field: value})


def test_temperature_schedule_and_stochastic_seed_requirement(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        temperature_cutoff_plies=3,
        early_temperature=1.25,
        late_temperature=0.0,
    )
    assert [config.temperature_for_ply(index) for index in range(5)] == [
        1.25,
        1.25,
        1.25,
        0.0,
        0.0,
    ]


def test_complete_game_visit_targets_values_and_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    game = play_self_play_game(
        config,
        UniformEvaluator(),
        source_checkpoint_sha256="a" * 64,
        game_index=0,
    )
    assert game.status == "completed"
    assert game.raw_game.final_result is not None
    assert len(game.policies) == game.raw_game.total_plies
    assert all(sum(policy.visit_counts) == 1 for policy in game.policies)
    examples = self_play_examples(game)
    assert len(examples) == game.raw_game.total_plies
    assert all(
        np.isclose(np.asarray(item.example.policy_probabilities).sum(), 1.0)
        for item in examples
    )
    assert all(item.example.value_target in {-1.0, 0.0, 1.0} for item in examples)
    assert any(
        item.example.value_target != policy.root_value
        for item, policy in zip(examples, game.policies, strict=True)
    )
    assert replay_game(game.raw_game).total_plies == game.raw_game.total_plies


def test_truncated_game_is_not_a_draw_or_training_data(tmp_path: Path) -> None:
    game = play_self_play_game(
        _config(tmp_path, max_game_plies=1),
        UniformEvaluator(),
        source_checkpoint_sha256="b" * 64,
        game_index=0,
    )
    assert game.status == "truncated"
    assert game.raw_game.final_result is None
    assert game.termination_reason == "maximum_plies_reached"
    assert self_play_examples(game) == ()


def test_corpus_persistence_dataset_and_adapter(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "source.pt")
    config = _config(tmp_path)
    corpus = generate_self_play_corpus(config, evaluator=UniformEvaluator())
    loaded = load_self_play_corpus(config.output_dir)
    assert loaded.fingerprint == corpus.fingerprint
    assert loaded.manifest["completed_games"] == 1
    dataset_path = tmp_path / "dataset"
    manifest = build_self_play_dataset(loaded, dataset_path, shard_size=3)
    examples = load_dataset(dataset_path)
    assert len(examples) == manifest.total_examples
    assert len(ShardDataset(dataset_path)) == len(examples)
    assert any(len(item.example.policy_action_indices) >= 1 for item in examples)
    game_path = next((config.output_dir / "self_play_games").glob("*.json"))
    value = json.loads(game_path.read_text(encoding="utf-8"))
    value["game_seed"] += 1
    game_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SelfPlayError, match="fingerprint"):
        load_self_play_corpus(config.output_dir)


def test_one_cpu_iteration_creates_unpromoted_reloadable_candidate(
    tmp_path: Path,
) -> None:
    source = _checkpoint(tmp_path / "source.pt")
    source_bytes = source.read_bytes()
    output = tmp_path / "iteration"
    self_play = _config(
        tmp_path,
        checkpoint_path=source,
        output_dir=output / "self_play",
    )
    result = run_iteration(
        IterationConfig(
            source,
            output,
            self_play,
            TrainingConfig(
                batch_size=2,
                epochs=2,
                max_steps=2,
                gradient_accumulation_steps=1,
                random_seed=19,
                device="cpu",
            ),
        )
    )
    assert source.read_bytes() == source_bytes
    loaded = load_checkpoint(result.candidate_checkpoint)
    evaluation = NeuralPositionEvaluator(loaded.model, device="cpu").evaluate(
        create_initial_state(Empire.A)
    )
    assert evaluation.action_indices
    assert loaded.metadata["provenance"]["artifact_role"] == "candidate"
    assert result.summary["candidate_semantics"] == "unpromoted_candidate"
    training = result.summary["training"]
    assert isinstance(training, dict)
    assert training["parameters_changed"] is True
    assert inspect_iteration(output)["iteration_id"] == result.summary["iteration_id"]


def test_amp_is_rejected_for_cpu() -> None:
    with pytest.raises(ValueError, match="AMP"):
        TrainingConfig(device="cpu", amp_enabled=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_training_is_finite(
    neural_dataset_dir: Path, small_model_config: ModelConfig
) -> None:
    result = train_model(
        PolicyValueNetwork(small_model_config),
        ShardDataset(neural_dataset_dir),
        TrainingConfig(
            batch_size=2,
            max_steps=1,
            device="cuda",
            amp_enabled=True,
        ),
    )
    assert result.summary.amp_enabled
    assert result.summary.peak_cuda_memory_allocated > 0
    assert all(np.isfinite(value) for value in result.summary.latest_metrics.values())
