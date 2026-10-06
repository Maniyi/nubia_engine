from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nubia_engine import Empire
from nubia_training.cli import main
from nubia_training.datasets import (
    build_dataset,
    inspect_dataset,
    load_dataset,
    validate_dataset,
)
from nubia_training.errors import (
    AgentSpecError,
    DatasetValidationError,
    RecordIntegrityError,
    RecordValidationError,
    ReplayValidationError,
    SchemaVersionError,
)
from nubia_training.generation import (
    agent_spec,
    create_agent,
    generate_game,
    generate_series,
)
from nubia_training.records import (
    AgentSpec,
    RawGameRecord,
    canonical_json_bytes,
    create_raw_game_record,
    read_raw_game,
    write_raw_game,
)
from nubia_training.replay import game_to_examples, replay_game
from nubia_training.storage import read_shard, write_shard


def _random(seed: int = 7) -> AgentSpec:
    return agent_spec("random", seed=seed)


def _completed() -> RawGameRecord:
    return generate_game(
        _random(1),
        _random(2),
        first_player=Empire.A,
        max_plies=100,
    )


def _rebuild(record: RawGameRecord, **changes: object) -> RawGameRecord:
    fields = {
        "first_player": record.first_player,
        "agent_a": record.agent_a,
        "agent_b": record.agent_b,
        "plies": record.plies,
        "status": record.status,
        "final_result": record.final_result,
        "total_plies": record.total_plies,
        "abort_reason": record.abort_reason,
        "final_state_fingerprint": record.final_state_fingerprint,
        "generator_config": record.generator_config,
    }
    fields.update(changes)
    return create_raw_game_record(**fields)


@pytest.mark.parametrize("kind", ["random", "heuristic", "minimax", "iterative"])
def test_all_agent_specs_construct_deterministic_agents(kind: str) -> None:
    spec = agent_spec(kind, seed=5 if kind == "random" else None)
    assert create_agent(spec, Empire.A).name == spec.name
    assert canonical_json_bytes(spec.to_dict()) == canonical_json_bytes(spec.to_dict())


def test_agent_spec_rejects_unknown_missing_seed_and_time_limit() -> None:
    with pytest.raises(AgentSpecError):
        agent_spec("unknown")
    with pytest.raises(AgentSpecError):
        AgentSpec("random", "r")
    with pytest.raises(AgentSpecError):
        AgentSpec(
            "iterative",
            "i",
            (("time_limit_seconds", 0.1),),
        )


def test_generation_is_deterministic_and_schedules_sides_and_seeds() -> None:
    first = generate_series(
        _random(),
        _random(),
        game_count=2,
        base_seed=20,
        alternate_first_player=True,
        swap_sides=True,
    )
    second = generate_series(
        _random(),
        _random(),
        game_count=2,
        base_seed=20,
        alternate_first_player=True,
        swap_sides=True,
    )
    assert first == second
    assert [item.first_player for item in first] == [Empire.A, Empire.B]
    assert first[0].agent_a.seed == 20
    assert first[1].agent_b.seed == 22


def test_aborted_game_has_no_fabricated_result_or_examples() -> None:
    record = generate_game(_random(1), _random(2), first_player=Empire.A, max_plies=1)
    assert record.status == "aborted"
    assert record.final_result is None
    assert record.abort_reason == "maximum_plies_reached"
    assert game_to_examples(record) == ()


def test_raw_record_round_trip_idempotency_and_conflict(tmp_path: Path) -> None:
    record = _completed()
    path = write_raw_game(record, tmp_path)
    first_bytes = path.read_bytes()
    assert read_raw_game(path) == record
    assert write_raw_game(record, tmp_path) == path
    assert path.read_bytes() == first_bytes
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RecordIntegrityError):
        write_raw_game(record, tmp_path)


def test_raw_reader_rejects_malformed_and_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RecordValidationError):
        read_raw_game(path)
    data = _completed().to_dict()
    data["version"] = 999
    path.write_bytes(canonical_json_bytes(data))
    with pytest.raises(SchemaVersionError):
        read_raw_game(path)


def test_replay_and_conversion_validate_every_ply() -> None:
    record = _completed()
    report = replay_game(record)
    examples = game_to_examples(record)
    assert report.total_plies == record.total_plies == len(examples)
    assert all(
        np.asarray(item.example.policy_probabilities).tolist() == [1.0]
        for item in examples
    )
    assert [item.ply_index for item in examples] == list(range(len(examples)))


def test_replay_identifies_tampered_ply_and_field() -> None:
    record = _completed()
    bad_ply = replace(record.plies[0], notation="tampered")
    tampered = _rebuild(record, plies=(bad_ply, *record.plies[1:]))
    with pytest.raises(ReplayValidationError, match=r"ply 0: notation"):
        replay_game(tampered)


def test_content_fingerprint_tampering_is_detected() -> None:
    record = _completed()
    with pytest.raises(RecordIntegrityError):
        replay_game(replace(record, content_fingerprint="0" * 64))


def test_shard_round_trip_dtypes_and_no_pickle(tmp_path: Path) -> None:
    examples = game_to_examples(_completed())[:3]
    path = tmp_path / "shard.npz"
    checksum = write_shard(examples, path)
    assert len(checksum) == 64
    with np.load(path, allow_pickle=False) as arrays:
        assert all(arrays[name].dtype != object for name in arrays.files)
        assert arrays["spatial"].dtype == np.float32
        assert arrays["legal_indices"].dtype == np.uint16
        assert arrays["repetition_counts"].dtype == np.uint8
    loaded = read_shard(path)
    assert [item.example.fingerprint() for item in loaded] == [
        item.example.fingerprint() for item in examples
    ]


def test_dataset_manifest_multiple_shards_and_checksum(tmp_path: Path) -> None:
    record = _completed()
    manifest = build_dataset((record,), tmp_path, shard_size=7)
    loaded = load_dataset(tmp_path)
    assert len(loaded) == record.total_plies == manifest.total_examples
    shards = manifest.data["shards"]
    assert isinstance(shards, list)
    assert len(shards) > 1
    assert validate_dataset(tmp_path).dataset_id == manifest.dataset_id
    assert f"dataset {manifest.dataset_id}" in inspect_dataset(tmp_path)
    shard = tmp_path / "shards" / "shard-00000.npz"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(DatasetValidationError, match="checksum"):
        validate_dataset(tmp_path)


def test_cli_complete_small_aborted_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    games = tmp_path / "raw"
    dataset = tmp_path / "dataset"
    assert (
        main(
            [
                "generate-games",
                "--output-dir",
                str(games),
                "--games",
                "1",
                "--max-plies",
                "1",
            ]
        )
        == 0
    )
    assert main(["validate-games", "--games-dir", str(games / "games")]) == 0
    assert (
        main(
            [
                "build-dataset",
                "--games-dir",
                str(games / "games"),
                "--output-dir",
                str(dataset),
                "--shard-size",
                "2",
            ]
        )
        == 0
    )
    assert main(["validate-dataset", "--dataset-dir", str(dataset)]) == 0
    assert main(["inspect-dataset", "--dataset-dir", str(dataset)]) == 0
    assert "examples=0" in capsys.readouterr().out


def test_manifest_json_is_canonical(tmp_path: Path) -> None:
    manifest = build_dataset((_completed(),), tmp_path, shard_size=100)
    raw = (tmp_path / "manifest.json").read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw))
    assert manifest.dataset_fingerprint in raw.decode("ascii")
