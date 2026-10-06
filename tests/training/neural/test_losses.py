from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from nubia_training.neural import (
    mask_policy_logits,
    policy_probabilities,
    sparse_policy_value_loss,
)
from nubia_training.neural.errors import NeuralInputError


def _inputs() -> tuple[torch.Tensor, ...]:
    logits = torch.zeros(2, 60_000, requires_grad=True)
    legal = torch.zeros(2, 60_000, dtype=torch.bool)
    legal[0, [2, 5]] = True
    legal[1, [7, 9]] = True
    indices = torch.tensor([[2, 5], [7, 0]])
    probabilities = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    valid = torch.tensor([[True, True], [True, False]])
    values = torch.tensor([0.2, -0.5], requires_grad=True)
    targets = torch.tensor([1.0, -1.0])
    return logits, legal, indices, probabilities, valid, values, targets


def test_mask_and_probabilities_are_legal_stable_and_differentiable() -> None:
    logits, legal, *_ = _inputs()
    logits = logits.detach().clone().requires_grad_()
    logits.data[0, 4] = 1e30
    original = logits.detach().clone()
    masked = mask_policy_logits(logits, legal)
    probabilities = policy_probabilities(logits, legal)
    assert torch.equal(logits, original)
    assert bool(torch.isneginf(masked[~legal]).all())
    assert bool((probabilities[~legal] == 0.0).all())
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    probabilities[0, 2].backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert logits.grad[0, 4] == 0.0


@pytest.mark.parametrize(
    ("logits", "mask"),
    [
        (torch.zeros(1, 10), torch.ones(1, 10, dtype=torch.bool)),
        (torch.zeros(1, 60_000), torch.ones(60_000, dtype=torch.bool)),
        (torch.zeros(1, 60_000), torch.ones(1, 60_000)),
        (torch.zeros(1, 60_000), torch.zeros(1, 60_000, dtype=torch.bool)),
        (torch.full((1, 60_000), torch.inf), torch.ones(1, 60_000, dtype=torch.bool)),
    ],
)
def test_invalid_masking_inputs(logits: torch.Tensor, mask: torch.Tensor) -> None:
    with pytest.raises(NeuralInputError):
        policy_probabilities(logits, mask)


def test_sparse_loss_known_uniform_value_weights_and_backward() -> None:
    logits, legal, indices, probabilities, valid, values, targets = _inputs()
    result = sparse_policy_value_loss(
        logits,
        values,
        legal,
        indices,
        probabilities,
        valid,
        targets,
        policy_weight=2.0,
        value_weight=0.5,
    )
    assert result.policy_loss.item() == pytest.approx(math.log(2.0))
    assert result.value_loss.item() == pytest.approx((0.8**2 + 0.5**2) / 2)
    assert result.total_loss.item() == pytest.approx(
        2.0 * math.log(2.0) + 0.5 * result.value_loss.item()
    )
    result.total_loss.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())
    assert bool((logits.grad[~legal] == 0.0).all())


def test_illegal_logits_do_not_affect_loss_and_perfect_is_better() -> None:
    logits, legal, indices, probabilities, valid, values, targets = _inputs()
    first = sparse_policy_value_loss(
        logits, values, legal, indices, probabilities, valid, targets
    )
    altered = logits.detach().clone()
    altered[~legal] = 1e20
    second = sparse_policy_value_loss(
        altered, values, legal, indices, probabilities, valid, targets
    )
    assert first.policy_loss == second.policy_loss
    perfect_values = targets.clone()
    perfect = sparse_policy_value_loss(
        logits, perfect_values, legal, indices, probabilities, valid, targets
    )
    assert perfect.value_loss < first.value_loss


def test_loss_rejects_illegal_target_and_invalid_mass() -> None:
    logits, legal, indices, probabilities, valid, values, targets = _inputs()
    indices[0, 0] = 3
    with pytest.raises(NeuralInputError, match="illegal"):
        sparse_policy_value_loss(
            logits, values, legal, indices, probabilities, valid, targets
        )


def test_loss_rejects_malformed_sparse_and_value_inputs() -> None:
    logits, legal, indices, probabilities, valid, values, targets = _inputs()

    def call(
        *,
        predictions: torch.Tensor = values,
        target_indices: torch.Tensor = indices,
        target_probabilities: torch.Tensor = probabilities,
        validity: torch.Tensor = valid,
        value_targets: torch.Tensor = targets,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
    ) -> None:
        sparse_policy_value_loss(
            logits,
            predictions,
            legal,
            target_indices,
            target_probabilities,
            validity,
            value_targets,
            policy_weight=policy_weight,
            value_weight=value_weight,
        )

    invalid_calls: tuple[Callable[[], None], ...] = (
        lambda: call(predictions=values[:1]),
        lambda: call(target_indices=indices.flatten()),
        lambda: call(target_indices=torch.zeros((1, 2), dtype=torch.int64)),
        lambda: call(target_probabilities=torch.ones(2, 1)),
        lambda: call(validity=torch.ones(2, 1, dtype=torch.bool)),
        lambda: call(target_indices=indices.to(torch.int32)),
        lambda: call(validity=valid.to(torch.int32)),
        lambda: call(target_probabilities=probabilities.to(torch.float64)),
        lambda: call(value_targets=targets.to(torch.float64)),
        lambda: call(validity=torch.zeros_like(valid)),
        lambda: call(target_probabilities=torch.full_like(probabilities, torch.nan)),
        lambda: call(target_probabilities=torch.tensor([[-0.5, 1.5], [1.0, 0.0]])),
        lambda: call(target_indices=torch.tensor([[2, 2], [7, 0]])),
        lambda: call(predictions=torch.tensor([torch.inf, 0.0])),
        lambda: call(value_targets=torch.tensor([2.0, 0.0])),
        lambda: call(policy_weight=-1.0),
        lambda: call(value_weight=float("nan")),
    )
    for invalid_call in invalid_calls:
        with pytest.raises(NeuralInputError):
            invalid_call()
    indices[0, 0] = 2
    probabilities[0] = 0.2
    with pytest.raises(NeuralInputError, match="mass"):
        sparse_policy_value_loss(
            logits, values, legal, indices, probabilities, valid, targets
        )
