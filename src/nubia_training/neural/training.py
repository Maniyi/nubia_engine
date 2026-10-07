"""Small deterministic AdamW training loop for validated shard datasets."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from nubia_training.neural.data import (
    NeuralBatch,
    ShardDataset,
    collate_examples,
    dataloader_generator,
)
from nubia_training.neural.device import select_device
from nubia_training.neural.errors import (
    NeuralConfigurationError,
    NonFiniteTrainingError,
)
from nubia_training.neural.losses import sparse_policy_value_loss
from nubia_training.neural.model import PolicyValueNetwork

TRAINING_CONFIG_VERSION = 1


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeuralConfigurationError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeuralConfigurationError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise NeuralConfigurationError(f"{name} must be finite and non-negative")
    return converted


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    shuffle: bool = True
    random_seed: int = 0
    device: str = "cpu"
    data_loader_workers: int = 0
    amp_enabled: bool = False
    version: int = TRAINING_CONFIG_VERSION

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "gradient_accumulation_steps"):
            _positive_int(name, getattr(self, name))
        if self.max_steps is not None:
            _positive_int("max_steps", self.max_steps)
        if (
            isinstance(self.data_loader_workers, bool)
            or not isinstance(self.data_loader_workers, int)
            or self.data_loader_workers < 0
        ):
            raise NeuralConfigurationError(
                "data_loader_workers must be a non-negative integer"
            )
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise NeuralConfigurationError("random_seed must be an integer")
        if not isinstance(self.shuffle, bool):
            raise NeuralConfigurationError("shuffle must be Boolean")
        if not isinstance(self.amp_enabled, bool):
            raise NeuralConfigurationError("amp_enabled must be Boolean")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise NeuralConfigurationError("invalid device selection")
        if self.amp_enabled and self.device not in {"cuda", "auto"}:
            raise NeuralConfigurationError(
                "AMP requires a CUDA-capable device selection"
            )
        if self.version != TRAINING_CONFIG_VERSION:
            raise NeuralConfigurationError(
                f"training configuration version must be {TRAINING_CONFIG_VERSION}"
            )
        if _finite_nonnegative("learning_rate", self.learning_rate) == 0.0:
            raise NeuralConfigurationError("learning_rate must be positive")
        for name in (
            "weight_decay",
            "policy_loss_weight",
            "value_loss_weight",
            "gradient_clip_norm",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.policy_loss_weight == 0.0 and self.value_loss_weight == 0.0:
            raise NeuralConfigurationError("at least one loss weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingConfig:
        return cls(**values)


@dataclass(frozen=True, slots=True)
class EpochSummary:
    epoch: int
    examples_seen: int
    optimizer_steps: int
    mean_total_loss: float
    mean_policy_loss: float
    mean_value_loss: float


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    training_config_version: int
    selected_device: str
    completed_epochs: int
    optimizer_steps: int
    examples_seen: int
    epochs: tuple[EpochSummary, ...]
    initial_metrics: dict[str, float]
    amp_enabled: bool
    peak_cuda_memory_allocated: int
    peak_cuda_memory_reserved: int

    @property
    def latest_metrics(self) -> dict[str, float]:
        if not self.epochs:
            return {}
        latest = self.epochs[-1]
        return {
            "total_loss": latest.mean_total_loss,
            "policy_loss": latest.mean_policy_loss,
            "value_loss": latest.mean_value_loss,
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    summary: TrainingSummary
    optimizer: Optimizer


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _finite_gradients(model: PolicyValueNetwork) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_model(
    model: PolicyValueNetwork,
    dataset: ShardDataset,
    config: TrainingConfig,
    *,
    optimizer: Optimizer | None = None,
    initial_completed_epochs: int = 0,
    initial_optimizer_steps: int = 0,
    initial_examples_seen: int = 0,
) -> TrainingResult:
    """Train for the bounded configured run and return immutable accounting."""

    if len(dataset) == 0:
        raise NeuralConfigurationError("cannot train on an empty dataset")
    for name, value in (
        ("initial_completed_epochs", initial_completed_epochs),
        ("initial_optimizer_steps", initial_optimizer_steps),
        ("initial_examples_seen", initial_examples_seen),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NeuralConfigurationError(f"{name} must be a non-negative integer")
    seed_training(config.random_seed)
    device = select_device(config.device)
    if config.amp_enabled and device.type != "cuda":
        raise NeuralConfigurationError("AMP requested but CUDA is unavailable")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    active_optimizer = optimizer or AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    _move_optimizer_state(active_optimizer, device)
    active_optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp_enabled, init_scale=256.0)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.data_loader_workers,
        collate_fn=collate_examples,
        generator=dataloader_generator(config.random_seed),
    )
    steps = initial_optimizer_steps
    examples_seen = initial_examples_seen
    run_steps = 0
    epoch_summaries: list[EpochSummary] = []
    initial_metrics: dict[str, float] = {}
    stop = False
    for epoch_index in range(config.epochs):
        totals: list[float] = []
        policies: list[float] = []
        values: list[float] = []
        epoch_examples = 0
        accumulation = 0
        for raw_batch in loader:
            batch: NeuralBatch = raw_batch.to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=config.amp_enabled
            ):
                output = model(batch.spatial, batch.global_features)
                losses = sparse_policy_value_loss(
                    output.policy_logits,
                    output.values,
                    batch.legal_mask,
                    batch.policy_indices,
                    batch.policy_probabilities,
                    batch.policy_valid_mask,
                    batch.value_targets,
                    policy_weight=config.policy_loss_weight,
                    value_weight=config.value_loss_weight,
                )
            if not bool(torch.isfinite(losses.total_loss)):
                raise NonFiniteTrainingError("training loss became non-finite")
            # PyTorch's generated Tensor.backward stub remains dynamically typed.
            scaled_loss = losses.total_loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()  # type: ignore[no-untyped-call]
            accumulation += 1
            seen = batch.spatial.shape[0]
            examples_seen += seen
            epoch_examples += seen
            totals.append(float(losses.total_loss.detach().cpu()))
            policies.append(float(losses.policy_loss.detach().cpu()))
            values.append(float(losses.value_loss.detach().cpu()))
            if not initial_metrics:
                initial_metrics = {
                    "total_loss": totals[-1],
                    "policy_loss": policies[-1],
                    "value_loss": values[-1],
                }
            is_last_batch = epoch_examples >= len(dataset)
            if accumulation == config.gradient_accumulation_steps or is_last_batch:
                scaler.unscale_(active_optimizer)
                if not _finite_gradients(model):
                    raise NonFiniteTrainingError("model gradient became non-finite")
                if config.gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                if not _finite_gradients(model):
                    raise NonFiniteTrainingError("clipped gradient became non-finite")
                scaler.step(active_optimizer)
                scaler.update()
                active_optimizer.zero_grad(set_to_none=True)
                steps += 1
                run_steps += 1
                accumulation = 0
                if config.max_steps is not None and run_steps >= config.max_steps:
                    stop = True
                    break
        if totals:
            epoch_summaries.append(
                EpochSummary(
                    initial_completed_epochs + epoch_index + 1,
                    epoch_examples,
                    steps,
                    sum(totals) / len(totals),
                    sum(policies) / len(policies),
                    sum(values) / len(values),
                )
            )
        if stop:
            break
    return TrainingResult(
        TrainingSummary(
            TRAINING_CONFIG_VERSION,
            str(device),
            initial_completed_epochs + len(epoch_summaries),
            steps,
            examples_seen,
            tuple(epoch_summaries),
            initial_metrics,
            config.amp_enabled,
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        ),
        active_optimizer,
    )
