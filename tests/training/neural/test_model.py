from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from nubia_training.neural import (
    ModelConfig,
    PolicyValueNetwork,
    ResidualBlock,
    trainable_parameter_count,
)
from nubia_training.neural.errors import NeuralConfigurationError, NeuralInputError


def test_default_and_small_configuration_are_valid_and_deterministic(
    small_model_config: ModelConfig,
) -> None:
    default = ModelConfig()
    assert default.action_kinds == 6
    assert default.squares_per_board == 100
    assert ModelConfig.from_dict(small_model_config.to_dict()) == small_model_config
    with pytest.raises(FrozenInstanceError):
        small_model_config.trunk_channels = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spatial_channels", 20),
        ("global_features", 4),
        ("board_width", 9),
        ("action_kinds", 5),
        ("squares_per_board", 99),
        ("trunk_channels", 0),
        ("residual_blocks", True),
        ("architecture_version", 2),
        ("action_space_version", 2),
    ],
)
def test_invalid_model_configuration(field: str, value: object) -> None:
    values: dict[str, object] = {field: value}
    with pytest.raises(NeuralConfigurationError):
        ModelConfig(**values)  # type: ignore[arg-type]


def test_forward_shapes_range_gradients_and_no_mutation(
    small_model_config: ModelConfig,
) -> None:
    torch.manual_seed(3)
    model = PolicyValueNetwork(small_model_config)
    spatial = torch.rand(2, 21, 10, 10)
    globals_ = torch.rand(2, 5)
    original_spatial = spatial.clone()
    original_globals = globals_.clone()
    output = model(spatial, globals_)
    assert output.policy_logits.shape == (2, 60_000)
    assert output.policy_logits.dtype == torch.float32
    assert output.values.shape == (2,)
    assert bool((output.values.abs() <= 1.0).all())
    assert torch.equal(spatial, original_spatial)
    assert torch.equal(globals_, original_globals)
    (output.policy_logits[:, :2].sum() + output.values.sum()).backward()
    names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert any(name.startswith("residual_stack") for name in names)
    assert "policy_source.weight" in names
    assert "value_output.weight" in names


def test_globals_condition_policy_and_value(small_model_config: ModelConfig) -> None:
    torch.manual_seed(4)
    model = PolicyValueNetwork(small_model_config)
    spatial = torch.zeros(1, 21, 10, 10)
    first = model(spatial, torch.zeros(1, 5))
    second = model(spatial, torch.ones(1, 5))
    assert not torch.equal(first.policy_logits, second.policy_logits)
    assert not torch.equal(first.values, second.values)


def test_seed_reproducibility_and_batch_sizes(small_model_config: ModelConfig) -> None:
    torch.manual_seed(9)
    first = PolicyValueNetwork(small_model_config)
    torch.manual_seed(9)
    second = PolicyValueNetwork(small_model_config)
    for batch in (1, 3):
        spatial = torch.zeros(batch, 21, 10, 10)
        globals_ = torch.zeros(batch, 5)
        assert torch.equal(first(spatial, globals_)[0], second(spatial, globals_)[0])


@pytest.mark.parametrize(
    ("spatial", "globals_"),
    [
        (torch.zeros(21, 10, 10), torch.zeros(1, 5)),
        (torch.zeros(1, 20, 10, 10), torch.zeros(1, 5)),
        (torch.zeros(1, 21, 10, 10), torch.zeros(1, 4)),
        (torch.zeros(1, 21, 10, 10, dtype=torch.float64), torch.zeros(1, 5)),
        (torch.full((1, 21, 10, 10), torch.nan), torch.zeros(1, 5)),
    ],
)
def test_bad_model_inputs_rejected(
    small_model_config: ModelConfig, spatial: torch.Tensor, globals_: torch.Tensor
) -> None:
    with pytest.raises(NeuralInputError):
        PolicyValueNetwork(small_model_config)(spatial, globals_)


def test_residual_shape_factorized_order_and_parameter_bound(
    small_model_config: ModelConfig,
) -> None:
    block = ResidualBlock(8, 2)
    assert block(torch.zeros(2, 8, 10, 10)).shape == (2, 8, 10, 10)
    model = PolicyValueNetwork(small_model_config)
    assert trainable_parameter_count(PolicyValueNetwork()) < 2_000_000
    assert model.policy_source.bias is not None
    assert model.policy_destination.bias is not None
    with torch.no_grad():
        model.policy_source.weight.zero_()
        model.policy_source.bias.zero_()
        model.policy_destination.weight.zero_()
        model.policy_destination.bias.zero_()
        model.policy_source.bias[0] = 2.0
        model.policy_destination.bias[0] = 3.0
    logits = model(torch.zeros(1, 21, 10, 10), torch.zeros(1, 5)).policy_logits
    assert logits[0, 0] == logits[0, 99 * 100 + 99]
    assert logits[0, 10_000] != logits[0, 0]
