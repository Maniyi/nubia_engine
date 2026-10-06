from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import piece, state_with

from nubia_engine import (
    ActionKind,
    BrainwashAvailability,
    Empire,
    GameResult,
    Outcome,
    Piece,
    PieceType,
    ResultReason,
    Square,
    apply_action,
    create_initial_state,
    legal_actions_from,
    position_key,
)
from nubia_training import (
    ACTION_SPACE_VERSION,
    GLOBAL_FEATURE_NAMES,
    PIECE_TYPE_ORDER,
    SPATIAL_CHANNEL_NAMES,
    STATE_ENCODING_SPEC,
    EncodedState,
    InvalidEncodingError,
    encode_state,
    encoded_states_equal,
)


def _channel(name: str) -> int:
    return SPATIAL_CHANNEL_NAMES.index(name)


def test_initial_shape_dtype_manifest_and_contents() -> None:
    encoded = encode_state(create_initial_state(Empire.A), perspective=Empire.A)
    assert encoded.spatial.shape == (21, 10, 10)
    assert encoded.spatial.dtype == np.float32
    assert encoded.global_features.shape == (5,)
    assert encoded.global_features.dtype == np.float32
    assert len(PIECE_TYPE_ORDER) == len(PieceType) == 7
    assert STATE_ENCODING_SPEC.spatial_channel_names == SPATIAL_CHANNEL_NAMES
    assert STATE_ENCODING_SPEC.global_feature_names == GLOBAL_FEATURE_NAMES
    assert int(encoded.spatial[_channel("self_peasant")].sum()) == 10
    assert int(encoded.spatial[_channel("opponent_peasant")].sum()) == 10
    assert int(encoded.spatial[_channel("original_self")].sum()) == 20
    assert int(encoded.spatial[_channel("original_opponent")].sum()) == 20
    assert int(encoded.spatial[_channel("mystic_power_available")].sum()) == 4
    assert int(encoded.spatial[_channel("land")].sum()) == 50
    assert int(encoded.spatial[_channel("sea")].sum()) == 50
    assert int(encoded.spatial[_channel("self_resource")].sum()) == 5
    assert int(encoded.spatial[_channel("opponent_resource")].sum()) == 5


def test_empire_b_rotates_180_and_relabels_allegiances() -> None:
    square = Square(2, 3)
    state = state_with([(square, piece("q", PieceType.QUEEN, Empire.B))])
    from_a = encode_state(state, perspective=Empire.A)
    from_b = encode_state(state, perspective=Empire.B)
    assert from_a.spatial[_channel("opponent_queen"), 2, 3] == 1.0
    assert from_b.spatial[_channel("self_queen"), 7, 6] == 1.0
    assert from_b.spatial[_channel("self_queen"), 2, 3] == 0.0
    assert from_a.global_features[0] == 1.0
    assert from_b.global_features[0] == 0.0


def test_symmetric_initial_positions_normalize_spatially() -> None:
    as_a = encode_state(create_initial_state(Empire.A), perspective=Empire.A)
    as_b = encode_state(create_initial_state(Empire.B), perspective=Empire.B)
    assert np.array_equal(as_a.spatial, as_b.spatial)
    assert np.array_equal(as_a.global_features, as_b.global_features)
    assert np.array_equal(as_a.legal_action_mask, as_b.legal_action_mask)


def test_brainwash_and_rebrainwash_keep_current_and_original_separate() -> None:
    source, target = Square(4, 4), Square(2, 4)
    state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (target, piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.BRAINWASH
    )
    converted_state = apply_action(state, action)
    converted = encode_state(converted_state, perspective=Empire.A)
    assert converted.spatial[_channel("self_queen"), 2, 4] == 1.0
    assert converted.spatial[_channel("original_opponent"), 2, 4] == 1.0
    assert converted.spatial[_channel("mystic_power_available"), 4, 4] == 0.0

    restoring = state_with(
        [
            (source, piece("restore", PieceType.WEST_AFRICAN_MYSTIC)),
            (
                target,
                piece(
                    "q",
                    PieceType.QUEEN,
                    Empire.B,
                    original_empire=Empire.A,
                ),
            ),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    restore_action = next(
        action
        for action in legal_actions_from(restoring, source)
        if action.kind is ActionKind.REBRAINWASH
    )
    restored = encode_state(
        apply_action(restoring, restore_action), perspective=Empire.A
    )
    assert restored.spatial[_channel("self_queen"), 2, 4] == 1.0
    assert restored.spatial[_channel("original_self"), 2, 4] == 1.0


def test_mystic_power_spent_and_available_planes() -> None:
    available = piece("a", PieceType.WEST_AFRICAN_MYSTIC)
    spent = piece(
        "s",
        PieceType.WEST_AFRICAN_MYSTIC,
        brainwash=BrainwashAvailability.SPENT,
    )
    encoded = encode_state(
        state_with([(Square(4, 4), available), (Square(5, 5), spent)])
    )
    plane = encoded.spatial[_channel("mystic_power_available")]
    assert plane[4, 4] == 1.0
    assert plane[5, 5] == 0.0


def test_global_quiet_repetition_and_terminal_features() -> None:
    base = state_with([(Square(4, 4), piece("p", PieceType.PEASANT))], quiet=20)
    key = position_key(base)
    repeated = replace(base, position_history=(key, key, key, key))
    encoded = encode_state(repeated)
    assert encoded.global_features.tolist() == pytest.approx([1.0, 0.5, 1.0, 0.0, 0.0])

    outcomes = (
        (GameResult(Outcome.WIN, Empire.A, (ResultReason.MINE_VICTORY,)), 1.0),
        (GameResult(Outcome.WIN, Empire.B, (ResultReason.MINE_VICTORY,)), -1.0),
        (GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)), 0.0),
    )
    for result, expected in outcomes:
        terminal = replace(base, result=result)
        values = encode_state(terminal, perspective=Empire.A).global_features
        assert values[3] == 1.0
        assert values[4] == expected


def test_piece_ids_do_not_change_encoding_or_fingerprint() -> None:
    first = state_with([(Square(4, 4), piece("one", PieceType.QUEEN))])
    second = state_with([(Square(4, 4), piece("two", PieceType.QUEEN))])
    encoded_first = encode_state(first)
    encoded_second = encode_state(second)
    assert encoded_states_equal(encoded_first, encoded_second)
    assert encoded_first.fingerprint() == encoded_second.fingerprint()


def test_encoded_arrays_are_owned_and_cannot_be_made_writeable() -> None:
    encoded = encode_state(create_initial_state(Empire.A))
    for array in (
        encoded.spatial,
        encoded.global_features,
        encoded.legal_action_mask,
        encoded.legal_action_indices,
        encoded.immediate_repetition,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_encoded_state_copies_caller_arrays() -> None:
    original = encode_state(create_initial_state(Empire.A))
    spatial = original.spatial.copy()
    constructed = EncodedState(
        original.version,
        original.perspective,
        spatial,
        original.global_features.copy(),
        original.legal_action_mask.copy(),
        original.legal_action_indices.copy(),
        original.immediate_repetition.copy(),
        original.action_space_version,
    )
    spatial.fill(0.0)
    assert np.array_equal(constructed.spatial, original.spatial)


def test_non_rules_piece_display_data_is_not_encoded() -> None:
    base_piece = Piece("x", PieceType.QUEEN, Empire.A, Empire.A)
    state = state_with([(Square(4, 4), base_piece)])
    assert encode_state(state).spatial[_channel("self_queen"), 4, 4] == 1.0


def test_encode_state_validates_state_and_perspective() -> None:
    with pytest.raises(TypeError):
        encode_state("state")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_state(create_initial_state(Empire.A), perspective="A")  # type: ignore[arg-type]
    assert not encoded_states_equal(
        encode_state(create_initial_state(Empire.A)),
        object(),  # type: ignore[arg-type]
    )


def _components() -> tuple[EncodedState, dict[str, object]]:
    encoded = encode_state(create_initial_state(Empire.A))
    values: dict[str, object] = {
        "version": encoded.version,
        "perspective": encoded.perspective,
        "spatial": encoded.spatial.copy(),
        "global_features": encoded.global_features.copy(),
        "legal_action_mask": encoded.legal_action_mask.copy(),
        "legal_action_indices": encoded.legal_action_indices.copy(),
        "immediate_repetition": encoded.immediate_repetition.copy(),
        "action_space_version": encoded.action_space_version,
    }
    return encoded, values


def _invalid_encoded(**changes: object) -> EncodedState:
    _encoded, values = _components()
    values.update(changes)
    return EncodedState(**values)  # type: ignore[arg-type]


def test_encoded_state_rejects_versions_perspective_and_dtypes() -> None:
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(version=2)
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(action_space_version=ACTION_SPACE_VERSION + 1)
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(perspective="A")
    _encoded, values = _components()
    for field in (
        "spatial",
        "global_features",
        "legal_action_mask",
        "legal_action_indices",
        "immediate_repetition",
    ):
        changed = dict(values)
        changed[field] = np.asarray(changed[field], dtype=np.float64)
        with pytest.raises(InvalidEncodingError):
            EncodedState(**changed)  # type: ignore[arg-type]


def test_encoded_state_rejects_bad_shapes_and_floating_values() -> None:
    encoded, _values = _components()
    cases = (
        {"spatial": encoded.spatial[:, :, :9].copy()},
        {"global_features": encoded.global_features[:4].copy()},
        {"legal_action_mask": encoded.legal_action_mask[:10].copy()},
        {"legal_action_indices": encoded.legal_action_indices.reshape(2, 20).copy()},
        {"immediate_repetition": encoded.immediate_repetition[:10].copy()},
    )
    for changes in cases:
        with pytest.raises(InvalidEncodingError):
            _invalid_encoded(**changes)

    spatial = encoded.spatial.copy()
    spatial[0, 0, 0] = np.float32(np.nan)
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(spatial=spatial)
    spatial[0, 0, 0] = np.float32(0.5)
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(spatial=spatial)


def test_encoded_state_rejects_bad_global_and_action_consistency() -> None:
    encoded, _values = _components()
    for feature, value in ((1, 1.1), (0, 0.5), (4, 2.0)):
        globals_ = encoded.global_features.copy()
        globals_[feature] = np.float32(value)
        with pytest.raises(InvalidEncodingError):
            _invalid_encoded(global_features=globals_)
    globals_ = encoded.global_features.copy()
    globals_[4] = np.float32(1.0)
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(global_features=globals_)

    duplicate_indices = encoded.legal_action_indices.copy()
    duplicate_indices[1] = duplicate_indices[0]
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(legal_action_indices=duplicate_indices)
    mask = encoded.legal_action_mask.copy()
    mask[int(encoded.legal_action_indices[0])] = False
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(legal_action_mask=mask)
    repetition = encoded.immediate_repetition.copy()
    repetition[int(encoded.legal_action_indices[0])] = 3
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(immediate_repetition=repetition)
    repetition = encoded.immediate_repetition.copy()
    repetition[0] = 1
    with pytest.raises(InvalidEncodingError):
        _invalid_encoded(immediate_repetition=repetition)
