"""Portable versioned neural checkpoints with SHA-256 integrity sidecars."""

from __future__ import annotations

import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from nubia_training.action_space import ACTION_SPACE_VERSION
from nubia_training.datasets import DATASET_MANIFEST_VERSION
from nubia_training.encoding import STATE_ENCODING_VERSION
from nubia_training.examples import TRAINING_EXAMPLE_VERSION
from nubia_training.neural.errors import CheckpointError
from nubia_training.neural.model import (
    MODEL_ARCHITECTURE_VERSION,
    ModelConfig,
    PolicyValueNetwork,
)
from nubia_training.neural.training import (
    TRAINING_CONFIG_VERSION,
    TrainingConfig,
    TrainingSummary,
)
from nubia_training.storage import DATASET_SHARD_VERSION, file_sha256

CHECKPOINT_VERSION = 1


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _canonical_json(data: dict[str, Any]) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def checkpoint_sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(target.name + ".sha256.json")


def _metadata(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_id: str,
    dataset_fingerprint: str,
    completed_epoch: int,
    optimizer_step: int,
    examples_seen: int,
    latest_metrics: dict[str, float],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "training_config_version": TRAINING_CONFIG_VERSION,
        "state_encoding_version": STATE_ENCODING_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "training_example_version": TRAINING_EXAMPLE_VERSION,
        "dataset_manifest_version": DATASET_MANIFEST_VERSION,
        "dataset_shard_version": DATASET_SHARD_VERSION,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "completed_epoch": completed_epoch,
        "optimizer_step": optimizer_step,
        "examples_seen": examples_seen,
        "training_seed": training_config.random_seed,
        "latest_metrics": dict(sorted(latest_metrics.items())),
    }
    if provenance is not None:
        metadata["provenance"] = dict(sorted(provenance.items()))
    return metadata


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNetwork,
    training_config: TrainingConfig,
    *,
    dataset_id: str,
    dataset_fingerprint: str,
    completed_epoch: int,
    optimizer_step: int,
    examples_seen: int,
    latest_metrics: dict[str, float],
    optimizer: Optimizer | None = None,
    provenance: dict[str, Any] | None = None,
) -> str:
    """Atomically publish a new checkpoint and canonical checksum sidecar."""

    target = Path(path)
    sidecar = checkpoint_sidecar_path(target)
    if target.exists() or sidecar.exists():
        raise CheckpointError(f"refusing to overwrite existing checkpoint: {target}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (completed_epoch, optimizer_step, examples_seen)
    ):
        raise CheckpointError(
            "checkpoint progress values must be non-negative integers"
        )
    metadata = _metadata(
        model.config,
        training_config,
        dataset_id,
        dataset_fingerprint,
        completed_epoch,
        optimizer_step,
        examples_seen,
        latest_metrics,
        provenance,
    )
    payload = {
        "metadata": metadata,
        "model_state_dict": _cpu_tree(model.state_dict()),
        "optimizer_state_dict": (
            None if optimizer is None else _cpu_tree(optimizer.state_dict())
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = file_sha256(temporary)
        sidecar_data = _canonical_json(
            {"sha256": checksum, "checkpoint_file": target.name, **metadata}
        )
        side_descriptor, side_name = tempfile.mkstemp(
            prefix=".checkpoint-sidecar-", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(side_descriptor, "wb") as stream:
                stream.write(sidecar_data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            os.replace(side_name, sidecar)
        except BaseException:
            Path(side_name).unlink(missing_ok=True)
            raise
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return checksum


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: PolicyValueNetwork
    model_config: ModelConfig
    training_config: TrainingConfig
    metadata: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    path: Path
    sha256: str

    def restore_optimizer(self) -> Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        if self.optimizer_state_dict is None:
            raise CheckpointError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(self.optimizer_state_dict)
        return optimizer


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = checkpoint_sidecar_path(path)
    try:
        decoded = json.loads(sidecar.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read checkpoint sidecar: {error}") from error
    if not isinstance(decoded, dict):
        raise CheckpointError("checkpoint sidecar must be a JSON object")
    return decoded


def load_checkpoint(
    path: str | Path,
    *,
    expected_model_config: ModelConfig | None = None,
    expected_dataset_id: str | None = None,
    expected_dataset_fingerprint: str | None = None,
) -> LoadedCheckpoint:
    """Verify integrity and versions, then safely load tensors onto CPU."""

    target = Path(path)
    sidecar = _read_sidecar(target)
    expected_sha = sidecar.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise CheckpointError("checkpoint sidecar checksum is invalid")
    actual_sha = file_sha256(target)
    if not hmac.compare_digest(actual_sha, expected_sha):
        raise CheckpointError("checkpoint checksum mismatch")
    try:
        raw = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"cannot safely load checkpoint: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        raise CheckpointError("checkpoint payload is malformed")
    metadata = cast(dict[str, Any], raw["metadata"])
    versions = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "training_config_version": TRAINING_CONFIG_VERSION,
        "state_encoding_version": STATE_ENCODING_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "training_example_version": TRAINING_EXAMPLE_VERSION,
        "dataset_manifest_version": DATASET_MANIFEST_VERSION,
        "dataset_shard_version": DATASET_SHARD_VERSION,
    }
    for name, expected in versions.items():
        if metadata.get(name) != expected:
            raise CheckpointError(f"unsupported {name}: {metadata.get(name)!r}")
    sidecar_metadata = {
        key: value
        for key, value in sidecar.items()
        if key not in {"sha256", "checkpoint_file"}
    }
    if metadata != sidecar_metadata:
        raise CheckpointError("checkpoint metadata does not match its sidecar")
    model_values = metadata.get("model_config")
    training_values = metadata.get("training_config")
    if not isinstance(model_values, dict) or not isinstance(training_values, dict):
        raise CheckpointError("checkpoint configuration is malformed")
    try:
        model_config = ModelConfig.from_dict(model_values)
        training_config = TrainingConfig.from_dict(training_values)
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"invalid checkpoint configuration: {error}") from error
    if expected_model_config is not None and model_config != expected_model_config:
        raise CheckpointError("checkpoint model architecture mismatch")
    if (
        expected_dataset_id is not None
        and metadata.get("dataset_id") != expected_dataset_id
    ):
        raise CheckpointError("checkpoint dataset ID mismatch")
    if (
        expected_dataset_fingerprint is not None
        and metadata.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise CheckpointError("checkpoint dataset fingerprint mismatch")
    model_state = raw.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise CheckpointError("checkpoint model state is malformed")
    model = PolicyValueNetwork(model_config)
    try:
        model.load_state_dict(model_state)
    except (RuntimeError, TypeError) as error:
        raise CheckpointError(
            f"checkpoint model state is incompatible: {error}"
        ) from error
    model.to(torch.device("cpu"))
    optimizer_state = raw.get("optimizer_state_dict")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise CheckpointError("checkpoint optimizer state is malformed")
    return LoadedCheckpoint(
        model,
        model_config,
        training_config,
        metadata,
        cast(dict[str, Any] | None, optimizer_state),
        target,
        actual_sha,
    )


def summary_checkpoint_arguments(
    summary: TrainingSummary,
) -> tuple[int, int, int, dict[str, float]]:
    return (
        summary.completed_epochs,
        summary.optimizer_steps,
        summary.examples_seen,
        summary.latest_metrics,
    )
