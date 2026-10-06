"""Deterministic compressed NumPy shard storage without object arrays."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nubia_engine import Empire
from nubia_training.action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
from nubia_training.encoding import (
    GLOBAL_FEATURE_SHAPE,
    SPATIAL_SHAPE,
    STATE_ENCODING_VERSION,
    EncodedState,
)
from nubia_training.errors import DatasetValidationError, RecordIntegrityError
from nubia_training.examples import TRAINING_EXAMPLE_VERSION, TrainingExample
from nubia_training.replay import ProvenancedExample

DATASET_SHARD_VERSION = 1

_DTYPES = {
    "spatial": np.dtype(np.float32),
    "global_features": np.dtype(np.float32),
    "legal_offsets": np.dtype(np.int64),
    "legal_indices": np.dtype(np.uint16),
    "repetition_counts": np.dtype(np.uint8),
    "policy_offsets": np.dtype(np.int64),
    "policy_indices": np.dtype(np.uint16),
    "policy_probabilities": np.dtype(np.float32),
    "value_targets": np.dtype(np.float32),
    "perspectives": np.dtype("<U1"),
    "source_game_ids": np.dtype("<U24"),
    "source_ply_indices": np.dtype(np.int64),
    "agent_identities": None,
    "example_fingerprints": np.dtype("<U64"),
    "encoded_state_fingerprints": np.dtype("<U64"),
    "shard_version": np.dtype(np.int64),
    "state_encoding_version": np.dtype(np.int64),
    "action_space_version": np.dtype(np.int64),
    "training_example_version": np.dtype(np.int64),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrays(examples: tuple[ProvenancedExample, ...]) -> dict[str, NDArray[np.generic]]:
    spatial = (
        np.stack([item.example.encoded_state.spatial for item in examples])
        if examples
        else np.empty((0, *SPATIAL_SHAPE), dtype=np.float32)
    )
    globals_ = (
        np.stack([item.example.encoded_state.global_features for item in examples])
        if examples
        else np.empty((0, *GLOBAL_FEATURE_SHAPE), dtype=np.float32)
    )
    legal_lengths = [
        item.example.encoded_state.legal_action_indices.size for item in examples
    ]
    policy_lengths = [
        np.asarray(item.example.policy_action_indices).size for item in examples
    ]
    legal_offsets = np.asarray((0, *np.cumsum(legal_lengths).tolist()), dtype=np.int64)
    policy_offsets = np.asarray(
        (0, *np.cumsum(policy_lengths).tolist()), dtype=np.int64
    )
    legal_indices = (
        np.concatenate(
            [item.example.encoded_state.legal_action_indices for item in examples]
        )
        if examples
        else np.empty(0, dtype=np.uint16)
    )
    repetitions = (
        np.concatenate(
            [
                item.example.encoded_state.immediate_repetition[
                    item.example.encoded_state.legal_action_indices
                ]
                for item in examples
            ]
        )
        if examples
        else np.empty(0, dtype=np.uint8)
    )
    policy_indices = (
        np.concatenate(
            [
                cast(NDArray[np.uint16], item.example.policy_action_indices)
                for item in examples
            ]
        )
        if examples
        else np.empty(0, dtype=np.uint16)
    )
    probabilities = (
        np.concatenate(
            [
                cast(NDArray[np.float32], item.example.policy_probabilities)
                for item in examples
            ]
        )
        if examples
        else np.empty(0, dtype=np.float32)
    )
    max_agent = max((len(item.agent_identity) for item in examples), default=1)
    return {
        "spatial": np.asarray(spatial, dtype=np.float32),
        "global_features": np.asarray(globals_, dtype=np.float32),
        "legal_offsets": legal_offsets,
        "legal_indices": np.asarray(legal_indices, dtype=np.uint16),
        "repetition_counts": np.asarray(repetitions, dtype=np.uint8),
        "policy_offsets": policy_offsets,
        "policy_indices": np.asarray(policy_indices, dtype=np.uint16),
        "policy_probabilities": np.asarray(probabilities, dtype=np.float32),
        "value_targets": np.asarray(
            [item.example.value_target for item in examples], dtype=np.float32
        ),
        "perspectives": np.asarray(
            [item.acting_empire.value for item in examples], dtype="<U1"
        ),
        "source_game_ids": np.asarray(
            [item.game_id for item in examples], dtype="<U24"
        ),
        "source_ply_indices": np.asarray(
            [item.ply_index for item in examples], dtype=np.int64
        ),
        "agent_identities": np.asarray(
            [item.agent_identity for item in examples], dtype=f"<U{max_agent}"
        ),
        "example_fingerprints": np.asarray(
            [item.example.fingerprint() for item in examples], dtype="<U64"
        ),
        "encoded_state_fingerprints": np.asarray(
            [item.example.encoded_state.fingerprint() for item in examples],
            dtype="<U64",
        ),
        "shard_version": np.asarray(DATASET_SHARD_VERSION, dtype=np.int64),
        "state_encoding_version": np.asarray(STATE_ENCODING_VERSION, dtype=np.int64),
        "action_space_version": np.asarray(ACTION_SPACE_VERSION, dtype=np.int64),
        "training_example_version": np.asarray(
            TRAINING_EXAMPLE_VERSION, dtype=np.int64
        ),
    }


def _npz_bytes(arrays: dict[str, NDArray[np.generic]]) -> bytes:
    """Create deterministic NPZ bytes (fixed member order and ZIP timestamps)."""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                member.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    return output.getvalue()


def write_shard(examples: tuple[ProvenancedExample, ...], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _npz_bytes(_arrays(examples))
    checksum = hashlib.sha256(data).hexdigest()
    if destination.exists():
        if destination.read_bytes() == data:
            return checksum
        raise RecordIntegrityError(f"conflicting shard already exists: {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".shard-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return checksum


def _validate_offsets(
    name: str, offsets: NDArray[np.generic], flat_size: int, count: int
) -> None:
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != flat_size:
        raise DatasetValidationError(f"invalid {name} offsets")
    if (np.diff(offsets) < 0).any():
        raise DatasetValidationError(f"non-monotonic {name} offsets")


def read_shard(path: str | Path) -> tuple[ProvenancedExample, ...]:
    try:
        with np.load(Path(path), allow_pickle=False) as loaded:
            missing = set(_DTYPES) - set(loaded.files)
            if missing:
                raise DatasetValidationError(f"shard missing arrays: {sorted(missing)}")
            arrays = {name: loaded[name] for name in _DTYPES}
    except DatasetValidationError:
        raise
    except Exception as error:
        raise DatasetValidationError(f"cannot load shard {path}: {error}") from error
    for name, expected in _DTYPES.items():
        if expected is not None and arrays[name].dtype != expected:
            actual = arrays[name].dtype
            raise DatasetValidationError(
                f"shard array {name} has dtype {actual}, expected {expected}"
            )
        if arrays[name].dtype == np.dtype(object):
            raise DatasetValidationError(f"shard array {name} uses object dtype")
    versions = {
        "shard_version": DATASET_SHARD_VERSION,
        "state_encoding_version": STATE_ENCODING_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "training_example_version": TRAINING_EXAMPLE_VERSION,
    }
    for name, expected in versions.items():
        if arrays[name].shape != () or int(arrays[name]) != expected:
            raise DatasetValidationError(f"unsupported {name}")
    count = arrays["spatial"].shape[0]
    if arrays["spatial"].shape != (count, *SPATIAL_SHAPE) or arrays[
        "global_features"
    ].shape != (count, *GLOBAL_FEATURE_SHAPE):
        raise DatasetValidationError("invalid dense feature shapes")
    for name in (
        "value_targets",
        "perspectives",
        "source_game_ids",
        "source_ply_indices",
        "agent_identities",
        "example_fingerprints",
        "encoded_state_fingerprints",
    ):
        if arrays[name].shape != (count,):
            raise DatasetValidationError(f"invalid {name} shape")
    _validate_offsets(
        "legal", arrays["legal_offsets"], arrays["legal_indices"].size, count
    )
    _validate_offsets(
        "policy", arrays["policy_offsets"], arrays["policy_indices"].size, count
    )
    if arrays["repetition_counts"].shape != arrays["legal_indices"].shape:
        raise DatasetValidationError(
            "repetition counts are not aligned with legal indices"
        )
    if arrays["policy_probabilities"].shape != arrays["policy_indices"].shape:
        raise DatasetValidationError("policy arrays are not aligned")
    if (arrays["legal_indices"] >= ACTION_SPACE_SIZE).any() or (
        arrays["policy_indices"] >= ACTION_SPACE_SIZE
    ).any():
        raise DatasetValidationError("shard contains an out-of-range action index")
    output: list[ProvenancedExample] = []
    for index in range(count):
        try:
            legal_start, legal_end = arrays["legal_offsets"][index : index + 2]
            policy_start, policy_end = arrays["policy_offsets"][index : index + 2]
            legal = arrays["legal_indices"][legal_start:legal_end]
            repetition = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
            repetition[legal] = arrays["repetition_counts"][legal_start:legal_end]
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
            mask[legal] = True
            encoded = EncodedState(
                STATE_ENCODING_VERSION,
                Empire(str(arrays["perspectives"][index])),
                arrays["spatial"][index],
                arrays["global_features"][index],
                mask,
                legal,
                repetition,
                ACTION_SPACE_VERSION,
            )
            example = TrainingExample(
                encoded,
                arrays["policy_indices"][policy_start:policy_end],
                arrays["policy_probabilities"][policy_start:policy_end],
                float(arrays["value_targets"][index]),
            )
        except Exception as error:
            raise DatasetValidationError(
                f"invalid reconstructed example at index {index}: {error}"
            ) from error
        if encoded.fingerprint() != str(
            arrays["encoded_state_fingerprints"][index]
        ) or example.fingerprint() != str(arrays["example_fingerprints"][index]):
            raise DatasetValidationError(
                f"example fingerprint mismatch at index {index}"
            )
        output.append(
            ProvenancedExample(
                str(arrays["source_game_ids"][index]),
                int(arrays["source_ply_indices"][index]),
                Empire(str(arrays["perspectives"][index])),
                str(arrays["agent_identities"][index]),
                example,
            )
        )
    return tuple(output)
