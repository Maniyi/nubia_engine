from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nubia_training.neural import (
    CHECKPOINT_VERSION,
    ModelConfig,
    PolicyValueNetwork,
    ShardDataset,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    select_device,
    train_model,
)
from nubia_training.neural.checkpoints import checkpoint_sidecar_path
from nubia_training.neural.cli import main
from nubia_training.neural.errors import CheckpointError, DeviceUnavailableError


def test_device_selection_with_injected_availability() -> None:
    assert select_device("cpu") == torch.device("cpu")
    assert select_device(
        "auto", cuda_available=lambda: False, mps_available=lambda: False
    ) == torch.device("cpu")
    assert select_device(
        "auto", cuda_available=lambda: True, mps_available=lambda: False
    ) == torch.device("cuda")
    assert select_device(
        "auto", cuda_available=lambda: False, mps_available=lambda: True
    ) == torch.device("mps")
    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        select_device("cuda", cuda_available=lambda: False)
    with pytest.raises(DeviceUnavailableError, match="MPS"):
        select_device("mps", mps_available=lambda: False)


def _trained_checkpoint(
    tmp_path: Path, dataset_path: Path, config: ModelConfig
) -> tuple[Path, PolicyValueNetwork, ShardDataset, TrainingConfig]:
    dataset = ShardDataset(dataset_path)
    torch.manual_seed(30)
    model = PolicyValueNetwork(config)
    training = TrainingConfig(batch_size=2, max_steps=1, random_seed=30)
    result = train_model(model, dataset, training)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        training,
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.dataset_fingerprint,
        completed_epoch=result.summary.completed_epochs,
        optimizer_step=result.summary.optimizer_steps,
        examples_seen=result.summary.examples_seen,
        latest_metrics=result.summary.latest_metrics,
        optimizer=result.optimizer,
    )
    return path, model, dataset, training


def test_checkpoint_roundtrip_predictions_optimizer_and_resume(
    tmp_path: Path, neural_dataset_dir: Path, small_model_config: ModelConfig
) -> None:
    path, model, dataset, _training = _trained_checkpoint(
        tmp_path, neural_dataset_dir, small_model_config
    )
    loaded = load_checkpoint(
        path,
        expected_model_config=small_model_config,
        expected_dataset_id=dataset.dataset_id,
        expected_dataset_fingerprint=dataset.dataset_fingerprint,
    )
    assert loaded.metadata["checkpoint_version"] == CHECKPOINT_VERSION
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                collate_fn=__import__(
                    "nubia_training.neural", fromlist=["collate_examples"]
                ).collate_examples,
            )
        )
    )
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        expected = model(batch.spatial, batch.global_features)
        actual = loaded.model(batch.spatial, batch.global_features)
    assert torch.equal(expected.policy_logits, actual.policy_logits)
    assert torch.equal(expected.values, actual.values)
    optimizer = loaded.restore_optimizer()
    resumed = train_model(
        loaded.model,
        dataset,
        TrainingConfig(batch_size=2, max_steps=1, random_seed=30),
        optimizer=optimizer,
        initial_optimizer_steps=int(loaded.metadata["optimizer_step"]),
        initial_examples_seen=int(loaded.metadata["examples_seen"]),
    )
    assert resumed.summary.optimizer_steps == 2


def test_checkpoint_conflict_checksum_and_mismatches(
    tmp_path: Path, neural_dataset_dir: Path, small_model_config: ModelConfig
) -> None:
    path, model, dataset, training = _trained_checkpoint(
        tmp_path, neural_dataset_dir, small_model_config
    )
    with pytest.raises(CheckpointError, match="overwrite"):
        save_checkpoint(
            path,
            model,
            training,
            dataset_id=dataset.dataset_id,
            dataset_fingerprint=dataset.dataset_fingerprint,
            completed_epoch=1,
            optimizer_step=1,
            examples_seen=2,
            latest_metrics={},
        )
    with pytest.raises(CheckpointError, match="dataset ID"):
        load_checkpoint(path, expected_dataset_id="wrong")
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_checkpoint(path, expected_dataset_fingerprint="wrong")
    with pytest.raises(CheckpointError, match="architecture"):
        load_checkpoint(
            path,
            expected_model_config=ModelConfig(
                trunk_channels=16,
                residual_blocks=1,
                policy_embedding_dim=2,
                value_hidden_dim=8,
                normalization_groups=2,
            ),
        )
    data = path.read_bytes()
    path.write_bytes(data + b"x")
    with pytest.raises(CheckpointError, match="checksum"):
        load_checkpoint(path)
    assert checkpoint_sidecar_path(path).is_file()


def test_checkpoint_without_optimizer_and_invalid_progress(
    tmp_path: Path,
    small_model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync

    def assert_writable_and_sync(descriptor: int) -> None:
        assert os.write(descriptor, b"") == 0
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", assert_writable_and_sync)
    model = PolicyValueNetwork(small_model_config)
    training = TrainingConfig()
    invalid = tmp_path / "invalid.pt"
    with pytest.raises(CheckpointError, match="progress"):
        save_checkpoint(
            invalid,
            model,
            training,
            dataset_id="dataset",
            dataset_fingerprint="0" * 64,
            completed_epoch=-1,
            optimizer_step=0,
            examples_seen=0,
            latest_metrics={},
        )
    path = tmp_path / "weights.pt"
    save_checkpoint(
        path,
        model,
        training,
        dataset_id="dataset",
        dataset_fingerprint="0" * 64,
        completed_epoch=0,
        optimizer_step=0,
        examples_seen=0,
        latest_metrics={},
    )
    with pytest.raises(CheckpointError, match="optimizer"):
        load_checkpoint(path).restore_optimizer()
    checkpoint_sidecar_path(path).unlink()
    with pytest.raises(CheckpointError, match="sidecar"):
        load_checkpoint(path)


def test_cli_help_model_checkpoint_and_training(
    tmp_path: Path,
    neural_dataset_dir: Path,
    small_model_config: ModelConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "inspect-model",
                "--trunk-channels",
                "8",
                "--residual-blocks",
                "1",
                "--policy-embedding-dim",
                "2",
                "--value-hidden-dim",
                "8",
                "--normalization-groups",
                "2",
            ]
        )
        == 0
    )
    model_summary = json.loads(capsys.readouterr().out)
    assert model_summary["memory"]["trainable_parameters"] < 2_000_000
    output = tmp_path / "run"
    assert (
        main(
            [
                "train",
                str(neural_dataset_dir),
                str(output),
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--epochs",
                "1",
                "--max-steps",
                "1",
                "--trunk-channels",
                "8",
                "--residual-blocks",
                "1",
                "--policy-embedding-dim",
                "2",
                "--value-hidden-dim",
                "8",
                "--normalization-groups",
                "2",
            ]
        )
        == 0
    )
    train_summary = json.loads(capsys.readouterr().out)
    checkpoint = Path(train_summary["checkpoint"])
    assert checkpoint.parent == output
    assert main(["inspect-checkpoint", str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out)["metadata"]["optimizer_step"] == 1
    completed = subprocess.run(
        [sys.executable, "-m", "nubia_training.neural", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "inspect-model" in completed.stdout


def test_base_import_does_not_load_torch() -> None:
    code = "import sys, nubia_training; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
