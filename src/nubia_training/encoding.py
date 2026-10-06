"""Deterministic numerical encoding of immutable NUBIA game states."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nubia_engine import (
    ALL_SQUARES,
    BrainwashAvailability,
    Empire,
    GameState,
    Outcome,
    PieceType,
    Terrain,
    apply_action,
    legal_actions,
    position_key,
)
from nubia_training.action_space import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    legal_action_indices,
)
from nubia_training.errors import InvalidEncodingError, UnsupportedPieceTypeError

STATE_ENCODING_VERSION = 1

# This order is a compatibility contract. It deliberately does not use enum order.
PIECE_TYPE_ORDER: tuple[PieceType, ...] = (
    PieceType.IMPERION,
    PieceType.QUEEN,
    PieceType.NORTH_CENTRAL_WAR_CHIEF,
    PieceType.EAST_AFRICAN_HIGH_CHIEF,
    PieceType.SOUTH_AFRICAN_ADVISOR,
    PieceType.WEST_AFRICAN_MYSTIC,
    PieceType.PEASANT,
)
SPATIAL_CHANNEL_NAMES: tuple[str, ...] = (
    *(f"self_{piece_type.value.lower()}" for piece_type in PIECE_TYPE_ORDER),
    *(f"opponent_{piece_type.value.lower()}" for piece_type in PIECE_TYPE_ORDER),
    "original_self",
    "original_opponent",
    "mystic_power_available",
    "land",
    "sea",
    "self_resource",
    "opponent_resource",
)
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "side_to_move_is_self",
    "quiet_ply_progress",
    "current_repetition_count_normalized",
    "is_terminal",
    "terminal_outcome_for_self",
)
SPATIAL_SHAPE = (len(SPATIAL_CHANNEL_NAMES), 10, 10)
GLOBAL_FEATURE_SHAPE = (len(GLOBAL_FEATURE_NAMES),)


@dataclass(frozen=True, slots=True)
class StateEncodingSpec:
    """Immutable public metadata for state-encoding version 1."""

    version: int
    spatial_channel_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    spatial_shape: tuple[int, int, int]
    spatial_dtype: str
    global_shape: tuple[int]
    global_dtype: str
    mask_dtype: str
    repetition_dtype: str
    orientation: str
    perspective_rule: str
    global_formulas: tuple[str, ...]


STATE_ENCODING_SPEC = StateEncodingSpec(
    version=STATE_ENCODING_VERSION,
    spatial_channel_names=SPATIAL_CHANNEL_NAMES,
    global_feature_names=GLOBAL_FEATURE_NAMES,
    spatial_shape=SPATIAL_SHAPE,
    spatial_dtype="float32",
    global_shape=GLOBAL_FEATURE_SHAPE,
    global_dtype="float32",
    mask_dtype="bool",
    repetition_dtype="uint8",
    orientation="Empire A canonical; Empire B rotated 180 degrees",
    perspective_rule="requested empire is self; omitted perspective is side to move",
    global_formulas=(
        "1 iff side_to_move == perspective, else 0",
        "min(quiet_ply_count, 40) / 40",
        "min(current_position_occurrences, 3) / 3",
        "1 iff result is present, else 0",
        "+1 win, -1 loss, 0 draw/non-terminal; terminal flag disambiguates",
    ),
)

_PIECE_CHANNEL = {
    piece_type: index for index, piece_type in enumerate(PIECE_TYPE_ORDER)
}
_SELF_PIECE_OFFSET = 0
_OPPONENT_PIECE_OFFSET = len(PIECE_TYPE_ORDER)
_ORIGINAL_SELF_CHANNEL = len(PIECE_TYPE_ORDER) * 2
_ORIGINAL_OPPONENT_CHANNEL = _ORIGINAL_SELF_CHANNEL + 1
_MYSTIC_POWER_CHANNEL = _ORIGINAL_SELF_CHANNEL + 2
_LAND_CHANNEL = _ORIGINAL_SELF_CHANNEL + 3
_SEA_CHANNEL = _ORIGINAL_SELF_CHANNEL + 4
_SELF_RESOURCE_CHANNEL = _ORIGINAL_SELF_CHANNEL + 5
_OPPONENT_RESOURCE_CHANNEL = _ORIGINAL_SELF_CHANNEL + 6


def _validate_piece_types() -> None:
    if set(PIECE_TYPE_ORDER) != set(PieceType) or len(PIECE_TYPE_ORDER) != len(
        PieceType
    ):
        raise UnsupportedPieceTypeError(
            "PIECE_TYPE_ORDER must explicitly contain every engine PieceType"
        )


_validate_piece_types()


def _immutable_array(
    array: NDArray[np.generic], dtype: np.dtype[np.generic]
) -> NDArray[np.generic]:
    contiguous = np.asarray(array, dtype=dtype, order="C")
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )


def _perspective_or_active(state: GameState, perspective: Empire | None) -> Empire:
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if perspective is None:
        return state.side_to_move
    if not isinstance(perspective, Empire):
        raise TypeError("perspective must be an Empire or None")
    return perspective


def _normalized_coordinates(
    row: int, column: int, perspective: Empire
) -> tuple[int, int]:
    return (row, column) if perspective is Empire.A else (9 - row, 9 - column)


def _terminal_value(state: GameState, perspective: Empire) -> float:
    if state.result is None or state.result.outcome is Outcome.DRAW:
        return 0.0
    return 1.0 if state.result.winner is perspective else -1.0


def _current_occurrences(state: GameState) -> int:
    key = position_key(state)
    history = state.position_history or (key,)
    return history.count(key)


def immediate_repetition_consequences(
    state: GameState, *, perspective: Empire | None = None
) -> NDArray[np.uint8]:
    """Return prior successor occurrence counts, clipped to 0, 1, or 2.

    The complete engine ``GameState`` remains authoritative for deeper search.
    This fixed action-aligned feature describes only each immediate successor.
    """

    normalized_for = _perspective_or_active(state, perspective)
    indices = legal_action_indices(state, perspective=normalized_for)
    actions = legal_actions(state)
    prior_history = state.position_history or (position_key(state),)
    writable = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    for index, action in zip(indices, actions, strict=True):
        successor = apply_action(state, action)
        prior_count = prior_history.count(position_key(successor))
        writable[index] = min(prior_count, 2)
    return cast(NDArray[np.uint8], _immutable_array(writable, np.dtype(np.uint8)))


@dataclass(frozen=True, slots=True, eq=False)
class EncodedState:
    """Owned, immutable versioned numerical observation of a game state."""

    version: int
    perspective: Empire
    spatial: NDArray[np.float32]
    global_features: NDArray[np.float32]
    legal_action_mask: NDArray[np.bool_]
    legal_action_indices: NDArray[np.uint16]
    immediate_repetition: NDArray[np.uint8]
    action_space_version: int

    def __post_init__(self) -> None:
        if self.version != STATE_ENCODING_VERSION:
            raise InvalidEncodingError(
                f"state encoding version must be {STATE_ENCODING_VERSION}"
            )
        if self.action_space_version != ACTION_SPACE_VERSION:
            raise InvalidEncodingError(
                f"action-space version must be {ACTION_SPACE_VERSION}"
            )
        if not isinstance(self.perspective, Empire):
            raise InvalidEncodingError("perspective must be an Empire")

        supplied = (
            ("spatial", self.spatial, np.dtype(np.float32)),
            ("global features", self.global_features, np.dtype(np.float32)),
            ("legal-action mask", self.legal_action_mask, np.dtype(np.bool_)),
            ("legal action indices", self.legal_action_indices, np.dtype(np.uint16)),
            ("repetition vector", self.immediate_repetition, np.dtype(np.uint8)),
        )
        for name, array, expected_dtype in supplied:
            if not isinstance(array, np.ndarray) or array.dtype != expected_dtype:
                raise InvalidEncodingError(
                    f"{name} dtype must be {expected_dtype.name}"
                )

        spatial = cast(
            NDArray[np.float32],
            _immutable_array(self.spatial, np.dtype(np.float32)),
        )
        globals_ = cast(
            NDArray[np.float32],
            _immutable_array(self.global_features, np.dtype(np.float32)),
        )
        mask = cast(
            NDArray[np.bool_],
            _immutable_array(self.legal_action_mask, np.dtype(np.bool_)),
        )
        indices = cast(
            NDArray[np.uint16],
            _immutable_array(self.legal_action_indices, np.dtype(np.uint16)),
        )
        repetition = cast(
            NDArray[np.uint8],
            _immutable_array(self.immediate_repetition, np.dtype(np.uint8)),
        )
        if spatial.shape != SPATIAL_SHAPE:
            raise InvalidEncodingError(f"spatial shape must be {SPATIAL_SHAPE}")
        if globals_.shape != GLOBAL_FEATURE_SHAPE:
            raise InvalidEncodingError(
                f"global feature shape must be {GLOBAL_FEATURE_SHAPE}"
            )
        if mask.shape != (ACTION_SPACE_SIZE,):
            raise InvalidEncodingError("legal-action mask has the wrong shape")
        if indices.ndim != 1:
            raise InvalidEncodingError("legal action indices must be one-dimensional")
        if repetition.shape != (ACTION_SPACE_SIZE,):
            raise InvalidEncodingError("repetition vector has the wrong shape")
        if not np.isfinite(spatial).all() or not np.isfinite(globals_).all():
            raise InvalidEncodingError("floating-point features must be finite")
        if not np.isin(spatial, (np.float32(0.0), np.float32(1.0))).all():
            raise InvalidEncodingError("spatial values must be binary")
        if (
            not 0.0 <= float(globals_[0]) <= 1.0
            or not 0.0 <= float(globals_[1]) <= 1.0
            or not 0.0 <= float(globals_[2]) <= 1.0
            or not 0.0 <= float(globals_[3]) <= 1.0
            or not -1.0 <= float(globals_[4]) <= 1.0
        ):
            raise InvalidEncodingError("global features are outside their ranges")
        if globals_[0] not in (0.0, 1.0) or globals_[3] not in (0.0, 1.0):
            raise InvalidEncodingError("indicator global features must be binary")
        if globals_[3] == 0.0 and globals_[4] != 0.0:
            raise InvalidEncodingError(
                "a non-terminal observation cannot have a terminal outcome"
            )
        if indices.size != np.unique(indices).size:
            raise InvalidEncodingError("legal action indices must be unique")
        mask_indices = np.flatnonzero(mask)
        if not np.array_equal(mask_indices, np.sort(indices.astype(np.int64))):
            raise InvalidEncodingError("mask and legal action indices disagree")
        if (repetition > 2).any() or repetition[~mask].any():
            raise InvalidEncodingError(
                "repetition values must be 0..2 and zero for illegal actions"
            )
        object.__setattr__(self, "spatial", spatial)
        object.__setattr__(self, "global_features", globals_)
        object.__setattr__(self, "legal_action_mask", mask)
        object.__setattr__(self, "legal_action_indices", indices)
        object.__setattr__(self, "immediate_repetition", repetition)

    def fingerprint(self) -> str:
        """Return a stable SHA-256 digest of version metadata and array contents."""

        digest = hashlib.sha256()
        digest.update(b"nubia-encoded-state\0")
        digest.update(self.version.to_bytes(4, "big"))
        digest.update(self.action_space_version.to_bytes(4, "big"))
        digest.update(self.perspective.value.encode("ascii"))
        for array in (
            self.spatial,
            self.global_features,
            self.legal_action_mask,
            self.legal_action_indices,
            self.immediate_repetition,
        ):
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


def encoded_states_equal(first: EncodedState, second: EncodedState) -> bool:
    """Compare encoded states without NumPy's ambiguous array equality."""

    if not isinstance(first, EncodedState) or not isinstance(second, EncodedState):
        return False
    return first.fingerprint() == second.fingerprint()


def encode_state(
    state: GameState, *, perspective: Empire | None = None
) -> EncodedState:
    """Encode ``state`` from either empire's normalized point of view."""

    normalized_for = _perspective_or_active(state, perspective)
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)
    for index, square in enumerate(ALL_SQUARES):
        row, column = _normalized_coordinates(square.row, square.column, normalized_for)
        terrain_channel = (
            _LAND_CHANNEL if square.terrain is Terrain.LAND else _SEA_CHANNEL
        )
        spatial[terrain_channel, row, column] = np.float32(1.0)
        if square.is_resource:
            resource_channel = (
                _SELF_RESOURCE_CHANNEL
                if square.half is normalized_for
                else _OPPONENT_RESOURCE_CHANNEL
            )
            spatial[resource_channel, row, column] = np.float32(1.0)

        piece = state.board[index]
        if piece is None:
            continue
        try:
            type_channel = _PIECE_CHANNEL[piece.piece_type]
        except KeyError as error:
            raise UnsupportedPieceTypeError(
                f"unsupported piece type: {piece.piece_type!r}"
            ) from error
        allegiance_offset = (
            _SELF_PIECE_OFFSET
            if piece.current_empire is normalized_for
            else _OPPONENT_PIECE_OFFSET
        )
        spatial[allegiance_offset + type_channel, row, column] = np.float32(1.0)
        original_channel = (
            _ORIGINAL_SELF_CHANNEL
            if piece.original_empire is normalized_for
            else _ORIGINAL_OPPONENT_CHANNEL
        )
        spatial[original_channel, row, column] = np.float32(1.0)
        if piece.brainwash is BrainwashAvailability.AVAILABLE:
            spatial[_MYSTIC_POWER_CHANNEL, row, column] = np.float32(1.0)

    occurrences = min(_current_occurrences(state), 3)
    globals_ = np.asarray(
        (
            1.0 if state.side_to_move is normalized_for else 0.0,
            min(state.quiet_ply_count, 40) / 40.0,
            occurrences / 3.0,
            1.0 if state.result is not None else 0.0,
            _terminal_value(state, normalized_for),
        ),
        dtype=np.float32,
    )
    indices_tuple = legal_action_indices(state, perspective=normalized_for)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    if indices_tuple:
        mask[np.asarray(indices_tuple, dtype=np.uint16)] = True
    indices = np.asarray(indices_tuple, dtype=np.uint16)
    repetition = immediate_repetition_consequences(state, perspective=normalized_for)
    return EncodedState(
        version=STATE_ENCODING_VERSION,
        perspective=normalized_for,
        spatial=spatial,
        global_features=globals_,
        legal_action_mask=mask,
        legal_action_indices=indices,
        immediate_repetition=repetition,
        action_space_version=ACTION_SPACE_VERSION,
    )
