from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from nubia_training.neural import (
    ModelConfig,
    PolicyValueNetwork,
    ShardDataset,
    TrainingConfig,
    collate_examples,
    train_model,
)
from nubia_training.neural.errors import NeuralConfigurationError, NeuralInputError


def test_lazy_dataset_order_cache_dtypes_and_metadata(
    neural_dataset_dir: Path,
) -> None:
    dataset = ShardDataset(neural_dataset_dir, shard_cache_size=1)
    first = dataset[0]
    assert first.spatial.dtype == torch.float32
    assert first.global_features.dtype == torch.float32
    assert first.legal_indices.dtype == torch.int64
    assert first.repetition_counts.dtype == torch.uint8
    assert first.policy_indices.dtype == torch.int64
    assert first.policy_probabilities.dtype == torch.float32
    assert first.value_target.dtype == torch.float32
    assert first.provenance.ply_index == 0
    assert first.provenance.game_id
    assert first.provenance.perspective in {"A", "B"}
    _ = dataset[8]
    assert dataset.cached_shard_count == 1
    assert dataset[1].provenance.ply_index == 1


def test_collator_mask_padding_multiaction_and_copy_safety(
    neural_dataset_dir: Path,
) -> None:
    dataset = ShardDataset(neural_dataset_dir)
    first = dataset[0]
    legal = first.legal_indices[:2].clone()
    multi = replace(
        first,
        policy_indices=legal,
        policy_probabilities=torch.tensor([0.25, 0.75]),
    )
    batch = collate_examples([multi, dataset[1]])
    assert batch.spatial.dtype == torch.float32
    assert batch.global_features.dtype == torch.float32
    assert batch.legal_mask.dtype == torch.bool
    assert batch.policy_indices.dtype == torch.int64
    assert batch.policy_valid_mask[0].tolist() == [True, True]
    assert batch.policy_valid_mask[1].tolist() == [True, False]
    assert batch.policy_probabilities[0].tolist() == pytest.approx([0.25, 0.75])
    assert torch.equal(
        torch.nonzero(batch.legal_mask[0]).flatten(),
        multi.legal_indices.sort().values,
    )
    assert batch.repetition_counts.shape == batch.legal_indices.shape
    first.spatial.fill_(99)
    assert dataset[0].spatial.max() <= 1


def test_collator_rejects_empty_and_illegal_target(neural_dataset_dir: Path) -> None:
    with pytest.raises(NeuralInputError, match="empty"):
        collate_examples([])
    item = ShardDataset(neural_dataset_dir)[0]
    illegal = next(index for index in range(60_000) if index not in item.legal_indices)
    bad = replace(item, policy_indices=torch.tensor([illegal]))
    with pytest.raises(NeuralInputError, match="illegal"):
        collate_examples([bad])
    with pytest.raises(NeuralInputError, match="outside"):
        collate_examples([replace(item, legal_indices=torch.tensor([60_000]))])
    with pytest.raises(NeuralInputError, match="unique"):
        collate_examples([replace(item, legal_indices=item.legal_indices[[0, 0]])])
    with pytest.raises(NeuralInputError, match="repetition"):
        collate_examples([replace(item, repetition_counts=torch.zeros(1))])
    with pytest.raises(NeuralInputError, match="aligned"):
        collate_examples([replace(item, policy_probabilities=torch.ones(2))])


def test_dataset_index_validation_and_negative_index(neural_dataset_dir: Path) -> None:
    dataset = ShardDataset(neural_dataset_dir)
    assert dataset[-1].provenance.ply_index == len(dataset) - 1
    with pytest.raises(TypeError):
        _ = dataset[True]
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
    with pytest.raises(TypeError):
        ShardDataset(neural_dataset_dir, shard_cache_size=True)
    with pytest.raises(ValueError):
        ShardDataset(neural_dataset_dir, shard_cache_size=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("epochs", True),
        ("max_steps", 0),
        ("learning_rate", 0.0),
        ("gradient_accumulation_steps", 0),
        ("data_loader_workers", -1),
        ("random_seed", False),
        ("device", "tpu"),
        ("version", 2),
    ],
)
def test_invalid_training_configuration(field: str, value: object) -> None:
    with pytest.raises(NeuralConfigurationError):
        TrainingConfig(**{field: value})  # type: ignore[arg-type]


def test_one_step_changes_parameters_and_accounts_exactly(
    neural_dataset_dir: Path, small_model_config: ModelConfig
) -> None:
    torch.manual_seed(10)
    model = PolicyValueNetwork(small_model_config)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    result = train_model(
        model,
        ShardDataset(neural_dataset_dir),
        TrainingConfig(batch_size=2, epochs=2, max_steps=1, random_seed=10),
    )
    assert result.summary.optimizer_steps == 1
    assert result.summary.examples_seen == 2
    assert result.summary.completed_epochs == 1
    assert result.summary.latest_metrics["total_loss"] > 0.0
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    with pytest.raises(FrozenInstanceError):
        result.summary.optimizer_steps = 3  # type: ignore[misc]


def test_gradient_accumulation_and_deterministic_cpu(
    neural_dataset_dir: Path, small_model_config: ModelConfig
) -> None:
    states: list[dict[str, torch.Tensor]] = []
    summaries = []
    for _ in range(2):
        torch.manual_seed(22)
        model = PolicyValueNetwork(small_model_config)
        result = train_model(
            model,
            ShardDataset(neural_dataset_dir),
            TrainingConfig(
                batch_size=1,
                epochs=1,
                max_steps=1,
                gradient_accumulation_steps=2,
                random_seed=22,
            ),
        )
        states.append(
            {name: value.clone() for name, value in model.state_dict().items()}
        )
        summaries.append(result.summary)
    assert summaries[0] == summaries[1]
    assert summaries[0].examples_seen == 2
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
