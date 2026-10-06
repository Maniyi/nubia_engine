"""Numerically stable legal masking and sparse policy-value objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from nubia_training.action_space import ACTION_SPACE_SIZE
from nubia_training.neural.errors import NeuralInputError


def _validate_logits_and_mask(policy_logits: Tensor, legal_mask: Tensor) -> None:
    if policy_logits.ndim != 2 or policy_logits.shape[1] != ACTION_SPACE_SIZE:
        raise NeuralInputError(
            f"policy logits must have shape [batch,{ACTION_SPACE_SIZE}]"
        )
    if legal_mask.shape != policy_logits.shape:
        raise NeuralInputError("legal mask shape must match policy logits")
    if legal_mask.dtype != torch.bool:
        raise NeuralInputError("legal mask dtype must be bool")
    if legal_mask.device != policy_logits.device:
        raise NeuralInputError("legal mask and logits must be on the same device")
    if not policy_logits.is_floating_point():
        raise NeuralInputError("policy logits must be floating point")
    if not bool(torch.isfinite(policy_logits).all()):
        raise NeuralInputError("policy logits must be finite")
    if not bool(legal_mask.any(dim=1).all()):
        raise NeuralInputError("each sample must contain a legal action")


def mask_policy_logits(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Return new logits with illegal entries replaced by negative infinity."""

    _validate_logits_and_mask(policy_logits, legal_mask)
    return policy_logits.masked_fill(~legal_mask, -torch.inf)


def policy_probabilities(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Normalize probability mass over legal actions only."""

    masked = mask_policy_logits(policy_logits, legal_mask)
    return torch.softmax(masked, dim=1)


@dataclass(frozen=True, slots=True)
class LossResult:
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    regularization_loss: Tensor | None
    example_count: int


def sparse_policy_value_loss(
    policy_logits: Tensor,
    value_predictions: Tensor,
    legal_mask: Tensor,
    policy_indices: Tensor,
    policy_target_probabilities: Tensor,
    policy_valid_mask: Tensor,
    value_targets: Tensor,
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> LossResult:
    """Compute sparse target cross-entropy plus mean squared value error."""

    _validate_logits_and_mask(policy_logits, legal_mask)
    batch = policy_logits.shape[0]
    if value_predictions.shape != (batch,) or value_targets.shape != (batch,):
        raise NeuralInputError("value predictions and targets must have shape [batch]")
    if policy_indices.ndim != 2:
        raise NeuralInputError("policy indices must have shape [batch,max_targets]")
    if policy_indices.shape[0] != batch:
        raise NeuralInputError("policy indices batch dimension is wrong")
    if policy_target_probabilities.shape != policy_indices.shape:
        raise NeuralInputError("policy probability shape must match policy indices")
    if policy_valid_mask.shape != policy_indices.shape:
        raise NeuralInputError("policy validity mask shape must match policy indices")
    if policy_indices.dtype != torch.int64:
        raise NeuralInputError("policy indices must use int64")
    if policy_valid_mask.dtype != torch.bool:
        raise NeuralInputError("policy validity mask must use bool")
    if policy_target_probabilities.dtype != torch.float32:
        raise NeuralInputError("policy target probabilities must use float32")
    if value_targets.dtype != torch.float32:
        raise NeuralInputError("value targets must use float32")
    tensors = (
        policy_indices,
        policy_target_probabilities,
        policy_valid_mask,
        value_predictions,
        value_targets,
    )
    if any(tensor.device != policy_logits.device for tensor in tensors):
        raise NeuralInputError("all loss tensors must be on the same device")
    if not bool(policy_valid_mask.any(dim=1).all()):
        raise NeuralInputError("each sample must have a policy target")
    if not bool(torch.isfinite(policy_target_probabilities).all()):
        raise NeuralInputError("policy target probabilities must be finite")
    valid_probabilities = policy_target_probabilities[policy_valid_mask]
    if bool((valid_probabilities < 0).any()):
        raise NeuralInputError("policy target probabilities cannot be negative")
    probability_mass = torch.sum(policy_target_probabilities * policy_valid_mask, dim=1)
    if not bool(
        torch.allclose(
            probability_mass, torch.ones_like(probability_mass), atol=1e-6, rtol=0.0
        )
    ):
        raise NeuralInputError("policy probability mass must sum to one")
    safe_indices = policy_indices.masked_fill(~policy_valid_mask, 0)
    if bool((safe_indices < 0).any()) or bool(
        (safe_indices >= ACTION_SPACE_SIZE).any()
    ):
        raise NeuralInputError("policy target index is outside the action space")
    for row in range(batch):
        valid_indices = policy_indices[row][policy_valid_mask[row]]
        if torch.unique(valid_indices).numel() != valid_indices.numel():
            raise NeuralInputError("policy target indices must be unique")
    target_is_legal = torch.gather(legal_mask, 1, safe_indices)
    if not bool((target_is_legal | ~policy_valid_mask).all()):
        raise NeuralInputError("policy target references an illegal action")
    if not bool(torch.isfinite(value_predictions).all()) or not bool(
        torch.isfinite(value_targets).all()
    ):
        raise NeuralInputError("value tensors must be finite")
    if bool((value_targets.abs() > 1.0).any()):
        raise NeuralInputError("value targets must be in [-1, 1]")
    if not math.isfinite(policy_weight) or policy_weight < 0.0:
        raise NeuralInputError("policy weight must be finite and non-negative")
    if not math.isfinite(value_weight) or value_weight < 0.0:
        raise NeuralInputError("value weight must be finite and non-negative")

    log_probabilities = torch.log_softmax(
        policy_logits.masked_fill(~legal_mask, -torch.inf), dim=1
    )
    gathered = torch.gather(log_probabilities, 1, safe_indices)
    gathered = torch.where(policy_valid_mask, gathered, torch.zeros_like(gathered))
    policy_loss = -torch.sum(policy_target_probabilities * gathered, dim=1).mean()
    value_loss = torch.mean(torch.square(value_predictions - value_targets))
    total = policy_weight * policy_loss + value_weight * value_loss
    if not bool(torch.isfinite(total)):
        raise NeuralInputError("loss is non-finite")
    return LossResult(total, policy_loss, value_loss, None, batch)
