"""Lazy validated shard adapter and deterministic neural batch collation."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from nubia_training.action_space import ACTION_SPACE_SIZE
from nubia_training.datasets import DatasetManifest, read_manifest
from nubia_training.neural.errors import NeuralInputError
from nubia_training.storage import file_sha256, read_shard


@dataclass(frozen=True, slots=True)
class ExampleProvenance:
    game_id: str
    ply_index: int
    perspective: str
    agent_identity: str


@dataclass(frozen=True, slots=True)
class NeuralDatasetItem:
    spatial: Tensor
    global_features: Tensor
    legal_indices: Tensor
    repetition_counts: Tensor
    policy_indices: Tensor
    policy_probabilities: Tensor
    value_target: Tensor
    provenance: ExampleProvenance


class ShardDataset(Dataset[NeuralDatasetItem]):
    """Map-style dataset retaining only a bounded number of decoded shards."""

    def __init__(self, path: str | Path, *, shard_cache_size: int = 2) -> None:
        if isinstance(shard_cache_size, bool) or not isinstance(shard_cache_size, int):
            raise TypeError("shard_cache_size must be an integer")
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        self.root = Path(path)
        self.manifest: DatasetManifest = read_manifest(self.root)
        raw_shards = self.manifest.data["shards"]
        if not isinstance(raw_shards, list):
            raise NeuralInputError("manifest shards must be a list")
        self._shards: tuple[dict[str, object], ...] = tuple(
            dict(entry) for entry in raw_shards if isinstance(entry, dict)
        )
        if len(self._shards) != len(raw_shards):
            raise NeuralInputError("manifest contains an invalid shard entry")
        counts: list[int] = []
        for entry in self._shards:
            count = entry["example_count"]
            if isinstance(count, bool) or not isinstance(count, int):
                raise NeuralInputError("manifest shard count must be an integer")
            counts.append(count)
            filename = entry["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise NeuralInputError("unsafe shard filename")
            shard_path = self.root / "shards" / filename
            if not shard_path.is_file():
                raise NeuralInputError(f"missing shard: {filename}")
            if file_sha256(shard_path) != entry["sha256"]:
                raise NeuralInputError(f"shard checksum mismatch: {filename}")
            # The established validator reconstructs one shard at a time. Its
            # dense validation arrays are released before the next shard, so a
            # complete dataset's 60,000-entry masks are never retained here.
            if len(read_shard(shard_path)) != count:
                raise NeuralInputError(f"shard count mismatch: {filename}")
        self._ends = tuple(np.cumsum(counts).tolist())
        self._cache_size = shard_cache_size
        self._cache: OrderedDict[int, dict[str, NDArray[np.generic]]] = OrderedDict()

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    @property
    def dataset_fingerprint(self) -> str:
        return self.manifest.dataset_fingerprint

    @property
    def cached_shard_count(self) -> int:
        return len(self._cache)

    def __len__(self) -> int:
        return self.manifest.total_examples

    def _load_shard(self, shard_index: int) -> dict[str, NDArray[np.generic]]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        entry = self._shards[shard_index]
        filename = entry["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise NeuralInputError("unsafe shard filename")
        path = self.root / "shards" / filename
        if file_sha256(path) != entry["sha256"]:
            raise NeuralInputError(f"shard checksum mismatch: {filename}")
        try:
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
        except Exception as error:
            raise NeuralInputError(f"cannot load shard {filename}: {error}") from error
        if any(array.dtype == np.dtype(object) for array in arrays.values()):
            raise NeuralInputError("object arrays are forbidden in neural datasets")
        self._cache[shard_index] = arrays
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> NeuralDatasetItem:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        local = index - shard_start
        arrays = self._load_shard(shard_index)
        legal_start = int(arrays["legal_offsets"][local])
        legal_end = int(arrays["legal_offsets"][local + 1])
        policy_start = int(arrays["policy_offsets"][local])
        policy_end = int(arrays["policy_offsets"][local + 1])

        def tensor_copy(array: NDArray[np.generic], dtype: torch.dtype) -> Tensor:
            return torch.as_tensor(np.array(array, copy=True), dtype=dtype)

        return NeuralDatasetItem(
            tensor_copy(arrays["spatial"][local], torch.float32),
            tensor_copy(arrays["global_features"][local], torch.float32),
            tensor_copy(arrays["legal_indices"][legal_start:legal_end], torch.int64),
            tensor_copy(
                arrays["repetition_counts"][legal_start:legal_end], torch.uint8
            ),
            tensor_copy(arrays["policy_indices"][policy_start:policy_end], torch.int64),
            tensor_copy(
                arrays["policy_probabilities"][policy_start:policy_end],
                torch.float32,
            ),
            tensor_copy(arrays["value_targets"][local], torch.float32),
            ExampleProvenance(
                str(arrays["source_game_ids"][local]),
                int(arrays["source_ply_indices"][local]),
                str(arrays["perspectives"][local]),
                str(arrays["agent_identities"][local]),
            ),
        )


@dataclass(frozen=True, slots=True)
class NeuralBatch:
    spatial: Tensor
    global_features: Tensor
    legal_mask: Tensor
    legal_indices: Tensor
    legal_valid_mask: Tensor
    repetition_counts: Tensor
    policy_indices: Tensor
    policy_probabilities: Tensor
    policy_valid_mask: Tensor
    value_targets: Tensor
    provenance: tuple[ExampleProvenance, ...]

    def to(self, device: torch.device) -> NeuralBatch:
        return NeuralBatch(
            self.spatial.to(device),
            self.global_features.to(device),
            self.legal_mask.to(device),
            self.legal_indices.to(device),
            self.legal_valid_mask.to(device),
            self.repetition_counts.to(device),
            self.policy_indices.to(device),
            self.policy_probabilities.to(device),
            self.policy_valid_mask.to(device),
            self.value_targets.to(device),
            self.provenance,
        )


def collate_examples(items: list[NeuralDatasetItem]) -> NeuralBatch:
    if not items:
        raise NeuralInputError("cannot collate an empty batch")
    batch = len(items)
    max_legal = max(item.legal_indices.numel() for item in items)
    max_policy = max(item.policy_indices.numel() for item in items)
    legal_mask = torch.zeros((batch, ACTION_SPACE_SIZE), dtype=torch.bool)
    legal_indices = torch.zeros((batch, max_legal), dtype=torch.int64)
    legal_valid = torch.zeros((batch, max_legal), dtype=torch.bool)
    repetitions = torch.zeros((batch, max_legal), dtype=torch.uint8)
    policy_indices = torch.zeros((batch, max_policy), dtype=torch.int64)
    policy_probabilities = torch.zeros((batch, max_policy), dtype=torch.float32)
    policy_valid = torch.zeros((batch, max_policy), dtype=torch.bool)
    for row, item in enumerate(items):
        legal_count = item.legal_indices.numel()
        policy_count = item.policy_indices.numel()
        if legal_count == 0 or policy_count == 0:
            raise NeuralInputError(
                "each training example requires legal and target actions"
            )
        if bool((item.legal_indices < 0).any()) or bool(
            (item.legal_indices >= ACTION_SPACE_SIZE).any()
        ):
            raise NeuralInputError("legal action index is outside the action space")
        if torch.unique(item.legal_indices).numel() != legal_count:
            raise NeuralInputError("legal action indices must be unique")
        if item.repetition_counts.shape != item.legal_indices.shape:
            raise NeuralInputError("repetition counts must align with legal actions")
        if item.policy_probabilities.shape != item.policy_indices.shape:
            raise NeuralInputError("policy arrays must be aligned")
        legal_indices[row, :legal_count] = item.legal_indices
        legal_valid[row, :legal_count] = True
        repetitions[row, :legal_count] = item.repetition_counts
        legal_mask[row, item.legal_indices] = True
        policy_indices[row, :policy_count] = item.policy_indices
        policy_probabilities[row, :policy_count] = item.policy_probabilities
        policy_valid[row, :policy_count] = True
        if not bool(legal_mask[row, item.policy_indices].all()):
            raise NeuralInputError("policy target references an illegal action")
    return NeuralBatch(
        torch.stack([item.spatial for item in items]).to(torch.float32),
        torch.stack([item.global_features for item in items]).to(torch.float32),
        legal_mask,
        legal_indices,
        legal_valid,
        repetitions,
        policy_indices,
        policy_probabilities,
        policy_valid,
        torch.stack([item.value_target for item in items]).to(torch.float32),
        tuple(item.provenance for item in items),
    )


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
