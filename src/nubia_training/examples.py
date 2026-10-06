"""Immutable sparse supervised-target contracts for future training code."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nubia_engine import Empire
from nubia_training.action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
from nubia_training.encoding import STATE_ENCODING_VERSION, EncodedState
from nubia_training.errors import InvalidPolicyTargetError, VersionMismatchError

TRAINING_EXAMPLE_VERSION = 1
POLICY_SUM_TOLERANCE = 1e-6


def _immutable_1d(
    values: NDArray[np.generic] | tuple[int, ...] | tuple[float, ...],
    dtype: np.dtype[np.generic],
) -> NDArray[np.generic]:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise InvalidPolicyTargetError("policy arrays must be one-dimensional")
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype)


@dataclass(frozen=True, slots=True, eq=False)
class TrainingExample:
    """A validated sparse policy and scalar value paired with an encoded state."""

    encoded_state: EncodedState
    policy_action_indices: NDArray[np.uint16] | tuple[int, ...]
    policy_probabilities: NDArray[np.float32] | tuple[float, ...]
    value_target: float
    version: int = TRAINING_EXAMPLE_VERSION
    state_encoding_version: int = STATE_ENCODING_VERSION
    action_space_version: int = ACTION_SPACE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.encoded_state, EncodedState):
            raise TypeError("encoded_state must be an EncodedState")
        if self.version != TRAINING_EXAMPLE_VERSION:
            raise VersionMismatchError(
                f"training-example version must be {TRAINING_EXAMPLE_VERSION}"
            )
        if (
            self.state_encoding_version != self.encoded_state.version
            or self.state_encoding_version != STATE_ENCODING_VERSION
        ):
            raise VersionMismatchError("state-encoding versions do not match")
        if (
            self.action_space_version != self.encoded_state.action_space_version
            or self.action_space_version != ACTION_SPACE_VERSION
        ):
            raise VersionMismatchError("action-space versions do not match")

        raw_indices = np.asarray(self.policy_action_indices)
        if raw_indices.ndim != 1:
            raise InvalidPolicyTargetError("policy indices must be one-dimensional")
        if raw_indices.dtype.kind not in "iu" or raw_indices.dtype.kind == "b":
            raise InvalidPolicyTargetError("policy indices must be integers")
        if raw_indices.size == 0:
            raise InvalidPolicyTargetError("policy target must contain an action")
        if (raw_indices < 0).any() or (raw_indices >= ACTION_SPACE_SIZE).any():
            raise InvalidPolicyTargetError("policy index is outside the action space")
        indices = cast(
            NDArray[np.uint16],
            _immutable_1d(
                tuple(int(value) for value in raw_indices), np.dtype(np.uint16)
            ),
        )
        probabilities = cast(
            NDArray[np.float32],
            _immutable_1d(self.policy_probabilities, np.dtype(np.float32)),
        )
        if indices.shape != probabilities.shape:
            raise InvalidPolicyTargetError(
                "policy indices and probabilities must have equal length"
            )
        if np.unique(indices).size != indices.size:
            raise InvalidPolicyTargetError("policy indices must be unique")
        if not self.encoded_state.legal_action_mask[indices].all():
            raise InvalidPolicyTargetError("every policy index must be legal")
        if not np.isfinite(probabilities).all():
            raise InvalidPolicyTargetError("policy probabilities must be finite")
        if (probabilities < 0.0).any():
            raise InvalidPolicyTargetError("policy probabilities cannot be negative")
        probability_sum = float(np.sum(probabilities, dtype=np.float64))
        if not math.isclose(
            probability_sum, 1.0, rel_tol=0.0, abs_tol=POLICY_SUM_TOLERANCE
        ):
            raise InvalidPolicyTargetError(
                f"policy probability mass must sum to 1 within {POLICY_SUM_TOLERANCE}"
            )
        value = float(self.value_target)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise InvalidPolicyTargetError("value target must be finite and in [-1, 1]")

        object.__setattr__(self, "policy_action_indices", indices)
        object.__setattr__(self, "policy_probabilities", probabilities)
        object.__setattr__(self, "value_target", value)

    @property
    def perspective(self) -> Empire:
        """The player for whom the policy and value targets are interpreted."""

        return self.encoded_state.perspective

    def dense_policy(self) -> NDArray[np.float32]:
        """Materialize an immutable dense float32 policy only when requested."""

        dense = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        indices = cast(NDArray[np.uint16], self.policy_action_indices)
        probabilities = cast(NDArray[np.float32], self.policy_probabilities)
        dense[indices] = probabilities
        return np.frombuffer(dense.tobytes(), dtype=np.float32)

    def fingerprint(self) -> str:
        """Return a stable SHA-256 digest of observation, versions, and targets."""

        digest = hashlib.sha256()
        digest.update(b"nubia-training-example\0")
        digest.update(self.version.to_bytes(4, "big"))
        digest.update(self.state_encoding_version.to_bytes(4, "big"))
        digest.update(self.action_space_version.to_bytes(4, "big"))
        digest.update(self.encoded_state.fingerprint().encode("ascii"))
        indices = cast(NDArray[np.uint16], self.policy_action_indices)
        probabilities = cast(NDArray[np.float32], self.policy_probabilities)
        digest.update(indices.tobytes())
        digest.update(probabilities.tobytes())
        digest.update(np.float32(self.value_target).tobytes())
        return digest.hexdigest()
