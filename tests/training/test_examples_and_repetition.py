from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import cast

import numpy as np
import pytest
from conftest import piece, state_with
from numpy.typing import NDArray

from nubia_engine import (
    ActionKind,
    Empire,
    PieceType,
    Square,
    apply_action,
    create_initial_state,
    legal_actions,
    legal_actions_from,
    position_key,
)
from nubia_training import (
    ACTION_SPACE_SIZE,
    InvalidPolicyTargetError,
    TrainingExample,
    VersionMismatchError,
    action_to_index,
    encode_state,
    immediate_repetition_consequences,
)


def _move_action(state: object, source: Square, destination: Square) -> object:
    return next(
        action
        for action in legal_actions_from(state, source)  # type: ignore[arg-type]
        if action.kind is ActionKind.MOVE and action.destination == destination
    )


@pytest.mark.parametrize("prior_count", [0, 1, 2])
def test_immediate_repetition_prior_occurrence_counts(prior_count: int) -> None:
    source, destination = Square(4, 4), Square(4, 5)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    action = _move_action(state, source, destination)
    successor = apply_action(state, action)  # type: ignore[arg-type]
    successor_key = position_key(successor)
    history = (position_key(state),) + (successor_key,) * prior_count
    seeded = replace(state, position_history=history)
    index = action_to_index(seeded, action)  # type: ignore[arg-type]
    before = seeded
    values = immediate_repetition_consequences(seeded)
    assert values[index] == prior_count
    assert values.dtype == np.uint8
    assert not values.flags.writeable
    assert not values[~encode_state(seeded).legal_action_mask].any()
    assert seeded == before


def test_third_repetition_is_encoded_as_two_and_engine_draws() -> None:
    source, destination = Square(4, 4), Square(4, 5)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    action = _move_action(state, source, destination)
    successor_key = position_key(apply_action(state, action))  # type: ignore[arg-type]
    seeded = replace(
        state, position_history=(position_key(state), successor_key, successor_key)
    )
    index = action_to_index(seeded, action)  # type: ignore[arg-type]
    assert immediate_repetition_consequences(seeded)[index] == 2
    assert apply_action(seeded, action).result is not None  # type: ignore[arg-type]


def test_mine_victory_precedence_remains_engine_controlled() -> None:
    mine = Square.from_notation("B:o6")
    source = Square(4, mine.column)
    state = state_with([(source, piece("p", PieceType.PEASANT))])
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.destination == mine
    )
    successor_key = position_key(apply_action(state, action))
    seeded = replace(
        state, position_history=(position_key(state), successor_key, successor_key)
    )
    index = action_to_index(seeded, action)
    assert immediate_repetition_consequences(seeded)[index] == 2
    result = apply_action(seeded, action).result
    assert result is not None and result.winner is Empire.A


def test_valid_one_hot_and_multi_action_policy_targets() -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    first, second = (int(value) for value in encoded.legal_action_indices[:2])
    one_hot = TrainingExample(encoded, (first,), (1.0,), 1.0)
    multi = TrainingExample(encoded, (first, second), (0.25, 0.75), -1.0)
    for example, expected_value in ((one_hot, 1.0), (multi, -1.0)):
        dense = example.dense_policy()
        assert dense.dtype == np.float32
        assert dense.shape == (ACTION_SPACE_SIZE,)
        assert float(dense.sum()) == pytest.approx(1.0)
        assert example.value_target == expected_value
        assert example.perspective is Empire.A
        assert not dense.flags.writeable


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_all_boundary_value_targets_are_valid(value: float) -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    index = int(encoded.legal_action_indices[0])
    assert TrainingExample(encoded, (index,), (1.0,), value).value_target == value


@pytest.mark.parametrize("value", [-1.01, 1.01, float("nan"), float("inf")])
def test_bad_value_targets_are_rejected(value: float) -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    index = int(encoded.legal_action_indices[0])
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, (index,), (1.0,), value)


@pytest.mark.parametrize(
    ("indices", "probabilities"),
    [
        ((0,), (1.0,)),
        ((1, 1), (0.5, 0.5)),
        ((1,), (-1.0,)),
        ((1,), (float("nan"),)),
        ((1,), (float("inf"),)),
        ((1,), (0.9,)),
    ],
)
def test_invalid_policy_targets_are_rejected(
    indices: tuple[int, ...], probabilities: tuple[float, ...]
) -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    legal = int(encoded.legal_action_indices[0])
    adjusted = tuple(legal if index == 1 else index for index in indices)
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, adjusted, probabilities, 0.0)


def test_version_mismatch_is_rejected() -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    index = int(encoded.legal_action_indices[0])
    with pytest.raises(VersionMismatchError):
        TrainingExample(encoded, (index,), (1.0,), 0.0, state_encoding_version=2)
    with pytest.raises(VersionMismatchError):
        TrainingExample(encoded, (index,), (1.0,), 0.0, action_space_version=2)


def test_example_owns_arrays_is_immutable_and_has_stable_fingerprint() -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    index = int(encoded.legal_action_indices[0])
    indices = np.asarray([index], dtype=np.uint16)
    probabilities = np.asarray([1.0], dtype=np.float32)
    example = TrainingExample(encoded, indices, probabilities, 0.0)
    same = TrainingExample(encoded, (index,), (1.0,), 0.0)
    indices[0] = 0
    probabilities[0] = 0.0
    assert int(example.policy_action_indices[0]) == index
    assert float(example.policy_probabilities[0]) == 1.0
    assert example.fingerprint() == same.fingerprint()
    stored_indices = cast(NDArray[np.uint16], example.policy_action_indices)
    stored_probabilities = cast(NDArray[np.float32], example.policy_probabilities)
    assert not stored_indices.flags.writeable
    assert not stored_probabilities.flags.writeable
    with pytest.raises(FrozenInstanceError):
        example.value_target = 1.0  # type: ignore[misc]


def test_repetition_action_indices_align_with_engine_order() -> None:
    state = create_initial_state(Empire.B)
    encoded = encode_state(state, perspective=Empire.B)
    assert tuple(
        action_to_index(state, action, perspective=Empire.B)
        for action in legal_actions(state)
    ) == tuple(int(value) for value in encoded.legal_action_indices)


def test_example_rejects_wrong_model_version_and_index_shapes() -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    legal = int(encoded.legal_action_indices[0])
    with pytest.raises(TypeError):
        TrainingExample("encoded", (legal,), (1.0,), 0.0)  # type: ignore[arg-type]
    with pytest.raises(VersionMismatchError):
        TrainingExample(encoded, (legal,), (1.0,), 0.0, version=2)
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(
            encoded,
            np.asarray([[legal]], dtype=np.uint16),
            (1.0,),
            0.0,
        )
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, (float(legal),), (1.0,), 0.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, (), (), 0.0)
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, (-1,), (1.0,), 0.0)
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, (ACTION_SPACE_SIZE,), (1.0,), 0.0)


def test_example_rejects_probability_shape_and_length_mismatch() -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    legal = int(encoded.legal_action_indices[0])
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(
            encoded,
            (legal,),
            np.asarray([[1.0]], dtype=np.float32),
            0.0,
        )
    with pytest.raises(InvalidPolicyTargetError):
        TrainingExample(encoded, (legal,), (0.5, 0.5), 0.0)
