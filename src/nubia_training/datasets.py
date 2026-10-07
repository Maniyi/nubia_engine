"""Dataset manifests, deterministic builds, loading, and inspection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nubia_engine import Empire, Outcome
from nubia_training.action_space import ACTION_SPACE_VERSION
from nubia_training.encoding import STATE_ENCODING_VERSION
from nubia_training.errors import (
    DatasetValidationError,
    RecordIntegrityError,
    SchemaVersionError,
)
from nubia_training.examples import TRAINING_EXAMPLE_VERSION
from nubia_training.records import RawGameRecord, canonical_json_bytes, read_raw_game
from nubia_training.replay import ProvenancedExample, game_to_examples, replay_game
from nubia_training.storage import (
    DATASET_SHARD_VERSION,
    file_sha256,
    read_shard,
    write_shard,
)

DATASET_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    data: dict[str, object]

    @property
    def dataset_id(self) -> str:
        return str(self.data["dataset_id"])

    @property
    def dataset_fingerprint(self) -> str:
        return str(self.data["dataset_fingerprint"])

    @property
    def total_examples(self) -> int:
        value = self.data["total_example_count"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise DatasetValidationError("total_example_count must be an integer")
        return value

    def content_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.data.items()
            if key not in {"dataset_id", "dataset_fingerprint"}
        }

    def validate_fingerprint(self) -> None:
        if self.data.get("version") != DATASET_MANIFEST_VERSION:
            raise SchemaVersionError(
                f"unsupported manifest version {self.data.get('version')}"
            )
        digest = hashlib.sha256(canonical_json_bytes(self.content_dict())).hexdigest()
        if self.dataset_fingerprint != digest or self.dataset_id != digest[:24]:
            raise DatasetValidationError("dataset manifest fingerprint mismatch")


def _atomic_json(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise RecordIntegrityError(f"conflicting manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".manifest-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_dataset(
    records: tuple[RawGameRecord, ...],
    output_dir: str | Path,
    *,
    shard_size: int,
    build_config: tuple[tuple[str, str | int | float | bool | None], ...] = (),
    example_converter: Callable[
        [RawGameRecord], tuple[ProvenancedExample, ...]
    ] = game_to_examples,
) -> DatasetManifest:
    """Validate records, write verified shards, then publish one manifest."""

    if isinstance(shard_size, bool) or not isinstance(shard_size, int):
        raise TypeError("shard_size must be an integer")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    examples: list[ProvenancedExample] = []
    skipped: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    agents: Counter[str] = Counter()
    completed_plies: list[int] = []
    for record in records:
        replay_game(record)
        agents.update((record.agent_a.agent_type, record.agent_b.agent_type))
        if record.status == "aborted":
            skipped[record.abort_reason or "unknown_abort"] += 1
            outcomes["aborted"] += 1
            continue
        completed_plies.append(record.total_plies)
        if (
            record.final_result is not None
            and record.final_result.outcome is Outcome.DRAW
        ):
            outcomes["draw"] += 1
        elif record.final_result is not None and record.final_result.winner is Empire.A:
            outcomes["win_A"] += 1
        else:
            outcomes["win_B"] += 1
        examples.extend(example_converter(record))
    root = Path(output_dir)
    shard_dir = root / "shards"
    shard_entries: list[dict[str, object]] = []
    for number, start in enumerate(range(0, len(examples), shard_size)):
        batch = tuple(examples[start : start + shard_size])
        filename = f"shard-{number:05d}.npz"
        path = shard_dir / filename
        checksum = write_shard(batch, path)
        loaded = read_shard(path)
        if tuple(item.example.fingerprint() for item in loaded) != tuple(
            item.example.fingerprint() for item in batch
        ):
            raise DatasetValidationError(
                f"written shard failed verification: {filename}"
            )
        shard_entries.append(
            {"filename": filename, "example_count": len(batch), "sha256": checksum}
        )
    content: dict[str, object] = {
        "version": DATASET_MANIFEST_VERSION,
        "state_encoding_version": STATE_ENCODING_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "training_example_version": TRAINING_EXAMPLE_VERSION,
        "shard_version": DATASET_SHARD_VERSION,
        "source_games": [
            {
                "game_id": record.game_id,
                "content_fingerprint": record.content_fingerprint,
            }
            for record in records
        ],
        "shards": shard_entries,
        "total_game_count": len(records),
        "completed_game_count": len(completed_plies),
        "aborted_game_count": len(records) - len(completed_plies),
        "total_example_count": len(examples),
        "outcome_counts": dict(sorted(outcomes.items())),
        "agent_type_counts": dict(sorted(agents.items())),
        "total_plies": sum(record.total_plies for record in records),
        "completed_plies_min": min(completed_plies, default=0),
        "completed_plies_max": max(completed_plies, default=0),
        "completed_plies_average": (
            sum(completed_plies) / len(completed_plies) if completed_plies else 0.0
        ),
        "build_config": dict(build_config),
        "skipped_record_reasons": dict(sorted(skipped.items())),
    }
    digest = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    data = {**content, "dataset_id": digest[:24], "dataset_fingerprint": digest}
    manifest = DatasetManifest(data)
    manifest.validate_fingerprint()
    _atomic_json(root / "manifest.json", canonical_json_bytes(data))
    validate_dataset(root)
    return manifest


def read_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"cannot read manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise DatasetValidationError("manifest must be a JSON object")
    required = {
        "version",
        "dataset_id",
        "dataset_fingerprint",
        "state_encoding_version",
        "action_space_version",
        "training_example_version",
        "shard_version",
        "source_games",
        "shards",
        "total_example_count",
    }
    missing = required - set(data)
    if missing:
        raise DatasetValidationError(f"manifest missing fields: {sorted(missing)}")
    manifest = DatasetManifest(data)
    manifest.validate_fingerprint()
    versions = (
        ("state encoding", data["state_encoding_version"], STATE_ENCODING_VERSION),
        ("action space", data["action_space_version"], ACTION_SPACE_VERSION),
        (
            "training example",
            data["training_example_version"],
            TRAINING_EXAMPLE_VERSION,
        ),
        ("shard", data["shard_version"], DATASET_SHARD_VERSION),
    )
    for name, actual, expected in versions:
        if actual != expected:
            raise SchemaVersionError(f"unsupported {name} version {actual}")
    _validate_manifest_counts(manifest)
    return manifest


def _manifest_int(data: dict[str, object], name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetValidationError(f"manifest {name} must be non-negative integer")
    return value


def _validate_manifest_counts(manifest: DatasetManifest) -> None:
    data = manifest.data
    sources = data.get("source_games")
    shards = data.get("shards")
    if not isinstance(sources, list) or not isinstance(shards, list):
        raise DatasetValidationError("source_games and shards must be lists")
    game_count = _manifest_int(data, "total_game_count")
    completed = _manifest_int(data, "completed_game_count")
    aborted = _manifest_int(data, "aborted_game_count")
    total_examples = _manifest_int(data, "total_example_count")
    if game_count != len(sources) or completed + aborted != game_count:
        raise DatasetValidationError("manifest game counts are inconsistent")
    game_ids: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            raise DatasetValidationError("source game entry must be an object")
        game_id = source.get("game_id")
        fingerprint = source.get("content_fingerprint")
        if not isinstance(game_id, str) or len(game_id) != 24:
            raise DatasetValidationError("source game ID is invalid")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise DatasetValidationError("source game fingerprint is invalid")
        game_ids.append(game_id)
    if len(game_ids) != len(set(game_ids)):
        raise DatasetValidationError("source game IDs must be unique")
    shard_total = 0
    for index, entry in enumerate(shards):
        if not isinstance(entry, dict):
            raise DatasetValidationError("shard entry must be an object")
        if entry.get("filename") != f"shard-{index:05d}.npz":
            raise DatasetValidationError(
                "shard filenames are not canonical and ordered"
            )
        count = entry.get("example_count")
        checksum = entry.get("sha256")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise DatasetValidationError("shard example count must be positive")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise DatasetValidationError("shard checksum is invalid")
        shard_total += count
    if shard_total != total_examples:
        raise DatasetValidationError("manifest shard counts do not match total")


def load_dataset(path: str | Path) -> tuple[ProvenancedExample, ...]:
    root = Path(path)
    manifest = read_manifest(root)
    shards = manifest.data["shards"]
    if not isinstance(shards, list):
        raise DatasetValidationError("manifest shards must be a list")
    output: list[ProvenancedExample] = []
    for entry in shards:
        if not isinstance(entry, dict):
            raise DatasetValidationError("shard entry must be an object")
        filename = entry.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise DatasetValidationError("unsafe shard filename")
        shard_path = root / "shards" / filename
        if not shard_path.is_file():
            raise DatasetValidationError(f"missing shard: {filename}")
        if file_sha256(shard_path) != entry.get("sha256"):
            raise DatasetValidationError(f"shard checksum mismatch: {filename}")
        examples = read_shard(shard_path)
        if len(examples) != entry.get("example_count"):
            raise DatasetValidationError(f"shard count mismatch: {filename}")
        output.extend(examples)
    if len(output) != manifest.total_examples:
        raise DatasetValidationError("total example count mismatch")
    return tuple(output)


def validate_dataset(
    path: str | Path, games_dir: str | Path | None = None
) -> DatasetManifest:
    manifest = read_manifest(path)
    load_dataset(path)
    if games_dir is not None:
        declared = manifest.data["source_games"]
        if not isinstance(declared, list):
            raise DatasetValidationError("source_games must be a list")
        for source in declared:
            if not isinstance(source, dict):
                raise DatasetValidationError("source game entry must be an object")
            record = read_raw_game(Path(games_dir) / f"{source['game_id']}.json")
            if record.content_fingerprint != source.get("content_fingerprint"):
                raise DatasetValidationError(
                    f"source game integrity mismatch: {record.game_id}"
                )
            replay_game(record)
    return manifest


def inspect_dataset(path: str | Path) -> str:
    manifest = read_manifest(path)
    data = manifest.data
    outcomes = json.dumps(data["outcome_counts"], sort_keys=True, separators=(",", ":"))
    shards = data["shards"]
    if not isinstance(shards, list):
        raise DatasetValidationError("manifest shards must be a list")
    return (
        f"dataset {manifest.dataset_id}: games={data['total_game_count']} "
        f"completed={data['completed_game_count']} "
        f"aborted={data['aborted_game_count']} "
        f"examples={data['total_example_count']} shards={len(shards)} "
        f"outcomes={outcomes}"
    )


def records_from_directory(path: str | Path) -> tuple[RawGameRecord, ...]:
    root = Path(path)
    return tuple(read_raw_game(item) for item in sorted(root.glob("*.json")))
