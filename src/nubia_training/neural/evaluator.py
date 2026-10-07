"""Checkpoint-backed policy-value evaluation for immutable NUBIA states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from nubia_engine import Empire, GameState
from nubia_training import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    STATE_ENCODING_VERSION,
    encode_state,
)
from nubia_training.neural.checkpoints import CHECKPOINT_VERSION, load_checkpoint
from nubia_training.neural.device import select_device
from nubia_training.neural.errors import NeuralInferenceError
from nubia_training.neural.model import MODEL_ARCHITECTURE_VERSION, PolicyValueNetwork

PRIOR_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class PositionEvaluation:
    """Sparse policy and scalar value, both for ``perspective``."""

    action_indices: tuple[int, ...]
    priors: tuple[float, ...]
    value: float
    perspective: Empire
    state_encoding_version: int = STATE_ENCODING_VERSION
    action_space_version: int = ACTION_SPACE_VERSION
    model_architecture_version: int | None = None
    checkpoint_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_indices, tuple) or not self.action_indices:
            raise NeuralInferenceError("evaluation must contain legal action indices")
        if len(self.action_indices) != len(self.priors):
            raise NeuralInferenceError("evaluation indices and priors must align")
        if not isinstance(self.priors, tuple):
            raise NeuralInferenceError("evaluation priors must be an immutable tuple")
        if len(set(self.action_indices)) != len(self.action_indices):
            raise NeuralInferenceError("evaluation action indices must be unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.action_indices
        ):
            raise NeuralInferenceError("evaluation action indices must be integers")
        if any(not 0 <= index < ACTION_SPACE_SIZE for index in self.action_indices):
            raise NeuralInferenceError(
                "evaluation action index is outside the action space"
            )
        if any(not math.isfinite(prior) or prior < 0.0 for prior in self.priors):
            raise NeuralInferenceError(
                "evaluation priors must be finite and non-negative"
            )
        if not math.isclose(
            sum(self.priors), 1.0, rel_tol=0.0, abs_tol=PRIOR_SUM_TOLERANCE
        ):
            raise NeuralInferenceError("evaluation priors must sum to one")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise NeuralInferenceError("evaluation value must be finite and in [-1, 1]")
        if not isinstance(self.perspective, Empire):
            raise NeuralInferenceError("evaluation perspective must be an Empire")
        if self.state_encoding_version != STATE_ENCODING_VERSION:
            raise NeuralInferenceError("unsupported state encoding version")
        if self.action_space_version != ACTION_SPACE_VERSION:
            raise NeuralInferenceError("unsupported action-space version")
        for name in ("model_architecture_version", "checkpoint_version"):
            version = getattr(self, name)
            if version is not None and (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version <= 0
            ):
                raise NeuralInferenceError(f"{name} must be a positive integer or None")

    @property
    def legal_action_indices(self) -> tuple[int, ...]:
        return self.action_indices

    @property
    def prior_probabilities(self) -> tuple[float, ...]:
        return self.priors

    @property
    def evaluation_perspective(self) -> Empire:
        return self.perspective


@runtime_checkable
class PositionEvaluator(Protocol):
    """Structural evaluator accepted by MCTS; implementations need not use Torch."""

    def evaluate(self, state: GameState) -> PositionEvaluation:
        """Evaluate one non-terminal state from its side-to-move perspective."""
        ...


class NeuralPositionEvaluator:
    """Single-position inference using an in-memory or safely loaded model."""

    def __init__(self, model: PolicyValueNetwork, *, device: str = "cpu") -> None:
        if not isinstance(model, PolicyValueNetwork):
            raise TypeError("model must be a PolicyValueNetwork")
        config = model.config
        if config.architecture_version != MODEL_ARCHITECTURE_VERSION:
            raise NeuralInferenceError("model architecture version mismatch")
        if config.state_encoding_version != STATE_ENCODING_VERSION:
            raise NeuralInferenceError("model state encoding version mismatch")
        if config.action_space_version != ACTION_SPACE_VERSION:
            raise NeuralInferenceError("model action-space version mismatch")
        self._device = select_device(device)
        self._model = model.to(self._device)
        self._model.eval()
        self._checkpoint_version: int | None = None

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: str = "cpu"
    ) -> NeuralPositionEvaluator:
        loaded = load_checkpoint(path)
        evaluator = cls(loaded.model, device=device)
        evaluator._checkpoint_version = CHECKPOINT_VERSION
        return evaluator

    @property
    def model(self) -> PolicyValueNetwork:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def evaluate(self, state: GameState) -> PositionEvaluation:
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        if state.result is not None:
            raise NeuralInferenceError("cannot evaluate a terminal state")
        encoded = encode_state(state, perspective=state.side_to_move)
        indices = tuple(int(index) for index in encoded.legal_action_indices)
        if not indices:
            raise NeuralInferenceError("non-terminal state has no legal actions")
        spatial = torch.from_numpy(encoded.spatial.copy())[None].to(self._device)
        globals_ = torch.from_numpy(encoded.global_features.copy())[None].to(
            self._device
        )
        self._model.eval()
        with torch.inference_mode():
            output = self._model(spatial, globals_)
            legal = output.policy_logits[0, torch.tensor(indices, device=self._device)]
            probabilities = torch.softmax(legal, dim=0)
            value = float(output.values[0].detach().cpu())
            priors_array = probabilities.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(priors_array).all() or not math.isfinite(value):
            raise NeuralInferenceError("model produced non-finite output")
        priors = tuple(float(item) for item in priors_array)
        return PositionEvaluation(
            indices,
            priors,
            value,
            state.side_to_move,
            model_architecture_version=MODEL_ARCHITECTURE_VERSION,
            checkpoint_version=self._checkpoint_version,
        )


CheckpointPositionEvaluator = NeuralPositionEvaluator
PyTorchPositionEvaluator = NeuralPositionEvaluator
