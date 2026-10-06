"""Compact residual policy-value model for NUBIA's fixed representation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, cast

import torch
from torch import Tensor, nn

from nubia_training.action_space import (
    ACTION_KIND_ORDER,
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    BOARD_SQUARE_COUNT,
)
from nubia_training.encoding import (
    GLOBAL_FEATURE_SHAPE,
    SPATIAL_SHAPE,
    STATE_ENCODING_VERSION,
)
from nubia_training.examples import TRAINING_EXAMPLE_VERSION
from nubia_training.neural.errors import NeuralConfigurationError, NeuralInputError

MODEL_ARCHITECTURE_VERSION = 1


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeuralConfigurationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Immutable architecture definition tied to representation version 1."""

    spatial_channels: int = SPATIAL_SHAPE[0]
    global_features: int = GLOBAL_FEATURE_SHAPE[0]
    board_height: int = SPATIAL_SHAPE[1]
    board_width: int = SPATIAL_SHAPE[2]
    action_kinds: int = len(ACTION_KIND_ORDER)
    squares_per_board: int = BOARD_SQUARE_COUNT
    trunk_channels: int = 64
    residual_blocks: int = 4
    policy_embedding_dim: int = 16
    value_hidden_dim: int = 128
    normalization_groups: int = 8
    architecture_version: int = MODEL_ARCHITECTURE_VERSION
    state_encoding_version: int = STATE_ENCODING_VERSION
    action_space_version: int = ACTION_SPACE_VERSION
    training_example_version: int = TRAINING_EXAMPLE_VERSION

    def __post_init__(self) -> None:
        integer_fields = (
            "spatial_channels",
            "global_features",
            "board_height",
            "board_width",
            "action_kinds",
            "squares_per_board",
            "trunk_channels",
            "residual_blocks",
            "policy_embedding_dim",
            "value_hidden_dim",
            "normalization_groups",
        )
        for name in integer_fields:
            _positive_integer(name, getattr(self, name))
        expected = {
            "spatial_channels": SPATIAL_SHAPE[0],
            "global_features": GLOBAL_FEATURE_SHAPE[0],
            "board_height": SPATIAL_SHAPE[1],
            "board_width": SPATIAL_SHAPE[2],
            "action_kinds": len(ACTION_KIND_ORDER),
            "squares_per_board": BOARD_SQUARE_COUNT,
            "architecture_version": MODEL_ARCHITECTURE_VERSION,
            "state_encoding_version": STATE_ENCODING_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "training_example_version": TRAINING_EXAMPLE_VERSION,
        }
        for name, wanted in expected.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value != wanted:
                raise NeuralConfigurationError(f"{name} must be {wanted}")
        if self.board_height * self.board_width != self.squares_per_board:
            raise NeuralConfigurationError("board dimensions and square count disagree")
        if self.action_kinds * self.squares_per_board**2 != ACTION_SPACE_SIZE:
            raise NeuralConfigurationError("model action dimensions are incompatible")
        if self.trunk_channels % self.normalization_groups != 0:
            raise NeuralConfigurationError(
                "trunk_channels must be divisible by normalization_groups"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**values)


class ModelOutput(NamedTuple):
    policy_logits: Tensor
    values: Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.SiLU()

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.norm1(self.conv1(inputs)))
        return cast(Tensor, self.activation(inputs + self.norm2(self.conv2(hidden))))


class PolicyValueNetwork(nn.Module):
    """Residual trunk with factorized kind/source/destination policy logits."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        channels = config.trunk_channels
        self.input_projection = nn.Conv2d(
            config.spatial_channels, channels, 3, padding=1, bias=False
        )
        self.global_projection = nn.Linear(config.global_features, channels)
        self.input_norm = nn.GroupNorm(config.normalization_groups, channels)
        self.activation = nn.SiLU()
        self.residual_stack = nn.Sequential(
            *(
                ResidualBlock(channels, config.normalization_groups)
                for _ in range(config.residual_blocks)
            )
        )
        policy_channels = config.action_kinds * config.policy_embedding_dim
        self.policy_source = nn.Conv2d(channels, policy_channels, 1)
        self.policy_destination = nn.Conv2d(channels, policy_channels, 1)
        self.policy_kind_bias = nn.Parameter(torch.zeros(config.action_kinds))
        self.value_hidden = nn.Linear(
            channels + config.global_features, config.value_hidden_dim
        )
        self.value_output = nn.Linear(config.value_hidden_dim, 1)

    def _validate_inputs(self, spatial: Tensor, globals_: Tensor) -> None:
        batch = spatial.shape[0] if spatial.ndim else -1
        expected_spatial = (
            batch,
            self.config.spatial_channels,
            self.config.board_height,
            self.config.board_width,
        )
        if spatial.ndim != 4 or tuple(spatial.shape) != expected_spatial:
            raise NeuralInputError(
                "spatial must have shape "
                f"[batch,{self.config.spatial_channels},"
                f"{self.config.board_height},{self.config.board_width}]"
            )
        if globals_.ndim != 2 or tuple(globals_.shape) != (
            batch,
            self.config.global_features,
        ):
            raise NeuralInputError(
                f"globals must have shape [batch,{self.config.global_features}]"
            )
        if spatial.dtype != torch.float32 or globals_.dtype != torch.float32:
            raise NeuralInputError("model inputs must use float32")
        if spatial.device != globals_.device:
            raise NeuralInputError("model inputs must be on the same device")
        if not bool(torch.isfinite(spatial).all()) or not bool(
            torch.isfinite(globals_).all()
        ):
            raise NeuralInputError("model inputs must be finite")

    def forward(self, spatial: Tensor, globals_: Tensor) -> ModelOutput:
        self._validate_inputs(spatial, globals_)
        batch = spatial.shape[0]
        global_map = self.global_projection(globals_)[:, :, None, None]
        trunk = self.activation(
            self.input_norm(self.input_projection(spatial) + global_map)
        )
        trunk = self.residual_stack(trunk)

        kinds = self.config.action_kinds
        dimension = self.config.policy_embedding_dim
        squares = self.config.squares_per_board
        source = self.policy_source(trunk).reshape(batch, kinds, dimension, squares)
        destination = self.policy_destination(trunk).reshape(
            batch, kinds, dimension, squares
        )
        logits = torch.einsum("bkds,bkdt->bkst", source, destination)
        logits = logits / math.sqrt(dimension)
        logits = logits + self.policy_kind_bias[None, :, None, None]
        policy_logits = logits.reshape(batch, ACTION_SPACE_SIZE)

        pooled = torch.mean(trunk, dim=(2, 3))
        value_features = torch.cat((pooled, globals_), dim=1)
        values = torch.tanh(
            self.value_output(self.activation(self.value_hidden(value_features)))
        ).squeeze(1)
        return ModelOutput(policy_logits, values)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def model_memory_report(
    config: ModelConfig, batch_sizes: tuple[int, ...] = (1, 8, 16, 32)
) -> dict[str, object]:
    model = PolicyValueNetwork(config)
    parameters = trainable_parameter_count(model)
    return {
        "trainable_parameters": parameters,
        "parameter_bytes_float32": parameters * 4,
        "estimated_gradient_bytes": parameters * 4,
        "estimated_adamw_state_bytes": parameters * 8,
        "batches": {
            str(batch): {
                "policy_logit_bytes": batch * ACTION_SPACE_SIZE * 4,
                "dense_legal_mask_bytes": batch * ACTION_SPACE_SIZE,
            }
            for batch in batch_sizes
        },
        "note": "excludes framework/runtime overhead and some intermediate activations",
    }
