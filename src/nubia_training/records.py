"""Versioned immutable raw-game records and canonical JSON persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nubia_engine import ActionKind, Empire, GameResult, Outcome, ResultReason
from nubia_training.action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
from nubia_training.encoding import STATE_ENCODING_VERSION
from nubia_training.errors import (
    AgentSpecError,
    RecordIntegrityError,
    RecordValidationError,
    SchemaVersionError,
)
from nubia_training.examples import TRAINING_EXAMPLE_VERSION

RAW_GAME_RECORD_VERSION = 1
AGENT_SPEC_VERSION = 1
STANDARD_SETUP_VERSION = 1
_SAFE_ID = re.compile(r"[0-9a-f]{24}")
_SAFE_DIGEST = re.compile(r"[0-9a-f]{64}")
JsonScalar = str | int | float | bool | None


def canonical_json_bytes(value: object) -> bytes:
    """Return the project's canonical UTF-8 JSON representation."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordValidationError(f"{field} must be an integer")
    if positive and value <= 0:
        raise RecordValidationError(f"{field} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Stable JSON-safe description of one reproducible agent."""

    agent_type: str
    name: str
    configuration: tuple[tuple[str, JsonScalar], ...] = ()
    seed: int | None = None
    version: int = AGENT_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != AGENT_SPEC_VERSION:
            raise SchemaVersionError(f"unsupported agent spec version {self.version}")
        if self.agent_type not in {"random", "heuristic", "minimax", "iterative"}:
            raise AgentSpecError(f"unknown agent type: {self.agent_type!r}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise AgentSpecError("agent name must be non-empty")
        if not isinstance(self.configuration, tuple):
            raise AgentSpecError("agent configuration must be an immutable tuple")
        keys = tuple(pair[0] for pair in self.configuration)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise AgentSpecError("agent configuration keys must be unique and sorted")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise AgentSpecError("agent seed must be an integer")
        if self.agent_type == "random" and self.seed is None:
            raise AgentSpecError("random agents require an explicit seed")
        if self.agent_type != "random" and self.seed is not None:
            raise AgentSpecError("only random agents accept a seed")
        config = dict(self.configuration)
        for key in ("depth", "max_depth", "node_limit"):
            candidate = config.get(key)
            if key in config and (
                isinstance(candidate, bool)
                or not isinstance(candidate, int)
                or candidate <= 0
            ):
                raise AgentSpecError(f"{key} must be a positive integer")
        if config.get("time_limit_seconds") is not None:
            raise AgentSpecError("wall-clock limits are not reproducible")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "agent_type": self.agent_type,
            "name": self.name,
            "configuration": dict(self.configuration),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: object) -> AgentSpec:
        if not isinstance(value, dict):
            raise RecordValidationError("agent specification must be an object")
        expected = {"version", "agent_type", "name", "configuration", "seed"}
        if set(value) != expected:
            raise RecordValidationError("agent specification fields are invalid")
        try:
            config = value["configuration"]
            if not isinstance(config, dict):
                raise RecordValidationError("agent configuration must be an object")
            return cls(
                version=_integer(value["version"], "agent version"),
                agent_type=str(value["agent_type"]),
                name=str(value["name"]),
                configuration=tuple(sorted(config.items())),
                seed=value["seed"],
            )
        except KeyError as error:
            raise RecordValidationError(
                f"missing agent field {error.args[0]}"
            ) from error


@dataclass(frozen=True, slots=True)
class PlyRecord:
    """One zero-based decision and its integrity anchors."""

    ply_index: int
    acting_empire: Empire
    action_index: int
    action_space_version: int
    notation: str
    pre_state_fingerprint: str
    post_state_fingerprint: str
    action_kind: ActionKind

    def to_dict(self) -> dict[str, object]:
        return {
            "ply_index": self.ply_index,
            "acting_empire": self.acting_empire.value,
            "action_index": self.action_index,
            "action_space_version": self.action_space_version,
            "notation": self.notation,
            "pre_state_fingerprint": self.pre_state_fingerprint,
            "post_state_fingerprint": self.post_state_fingerprint,
            "action_kind": self.action_kind.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlyRecord:
        if not isinstance(value, dict):
            raise RecordValidationError("ply must be an object")
        expected = {
            "ply_index",
            "acting_empire",
            "action_index",
            "action_space_version",
            "notation",
            "pre_state_fingerprint",
            "post_state_fingerprint",
            "action_kind",
        }
        if set(value) != expected:
            raise RecordValidationError("ply fields are invalid")
        try:
            return cls(
                _integer(value["ply_index"], "ply_index"),
                Empire(value["acting_empire"]),
                _integer(value["action_index"], "action_index"),
                _integer(value["action_space_version"], "action_space_version"),
                str(value["notation"]),
                str(value["pre_state_fingerprint"]),
                str(value["post_state_fingerprint"]),
                ActionKind(value["action_kind"]),
            )
        except KeyError as error:
            raise RecordValidationError(f"missing ply field {error.args[0]}") from error
        except ValueError as error:
            raise RecordValidationError(f"invalid ply enum: {error}") from error


def _result_dict(result: GameResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "outcome": result.outcome.value,
        "winner": None if result.winner is None else result.winner.value,
        "reasons": [reason.value for reason in result.reasons],
        "officer_totals": result.scores,
    }


def _parse_result(value: object) -> GameResult | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RecordValidationError("final_result must be an object or null")
    try:
        winner = value["winner"]
        scores = value["officer_totals"]
        return GameResult(
            Outcome(value["outcome"]),
            None if winner is None else Empire(winner),
            tuple(ResultReason(item) for item in value["reasons"]),
            None if scores is None else tuple(scores),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RecordValidationError(f"invalid final result: {error}") from error


@dataclass(frozen=True, slots=True)
class RawGameRecord:
    version: int
    game_id: str
    content_fingerprint: str
    state_encoding_version: int
    action_space_version: int
    training_example_version: int
    setup_version: int
    first_player: Empire
    agent_a: AgentSpec
    agent_b: AgentSpec
    plies: tuple[PlyRecord, ...]
    status: str
    final_result: GameResult | None
    total_plies: int
    abort_reason: str | None
    final_state_fingerprint: str
    generator_config: tuple[tuple[str, JsonScalar], ...]

    def content_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "state_encoding_version": self.state_encoding_version,
            "action_space_version": self.action_space_version,
            "training_example_version": self.training_example_version,
            "setup_version": self.setup_version,
            "first_player": self.first_player.value,
            "agents": {"A": self.agent_a.to_dict(), "B": self.agent_b.to_dict()},
            "plies": [ply.to_dict() for ply in self.plies],
            "status": self.status,
            "final_result": _result_dict(self.final_result),
            "total_plies": self.total_plies,
            "abort_reason": self.abort_reason,
            "final_state_fingerprint": self.final_state_fingerprint,
            "generator_config": dict(self.generator_config),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.content_dict(),
            "game_id": self.game_id,
            "content_fingerprint": self.content_fingerprint,
        }

    def validate(self) -> None:
        if self.version != RAW_GAME_RECORD_VERSION:
            raise SchemaVersionError(f"unsupported raw record version {self.version}")
        expected_versions = (
            ("state encoding", self.state_encoding_version, STATE_ENCODING_VERSION),
            ("action space", self.action_space_version, ACTION_SPACE_VERSION),
            (
                "training example",
                self.training_example_version,
                TRAINING_EXAMPLE_VERSION,
            ),
            ("setup", self.setup_version, STANDARD_SETUP_VERSION),
        )
        for name, actual, expected in expected_versions:
            if actual != expected:
                raise SchemaVersionError(f"unsupported {name} version {actual}")
        if not _SAFE_ID.fullmatch(self.game_id):
            raise RecordValidationError("unsafe or invalid game ID")
        if self.total_plies != len(self.plies):
            raise RecordValidationError("total_plies does not match plies")
        if tuple(p.ply_index for p in self.plies) != tuple(range(len(self.plies))):
            raise RecordValidationError(
                "ply indices must be zero-based and consecutive"
            )
        for ply in self.plies:
            if ply.action_space_version != ACTION_SPACE_VERSION:
                raise SchemaVersionError(
                    f"unsupported ply action-space version {ply.action_space_version}"
                )
            if not 0 <= ply.action_index < ACTION_SPACE_SIZE:
                raise RecordValidationError(
                    "ply action index is outside the action space"
                )
            if not ply.notation:
                raise RecordValidationError("ply notation must be non-empty")
            if not _SAFE_DIGEST.fullmatch(ply.pre_state_fingerprint) or not (
                _SAFE_DIGEST.fullmatch(ply.post_state_fingerprint)
            ):
                raise RecordValidationError("ply state fingerprint is invalid")
        if self.status not in {"completed", "aborted"}:
            raise RecordValidationError("status must be completed or aborted")
        if (self.status == "completed") != (self.final_result is not None):
            raise RecordValidationError("completion and final result disagree")
        if self.status == "completed" and self.abort_reason is not None:
            raise RecordValidationError("completed records cannot have an abort reason")
        if self.status == "aborted" and not self.abort_reason:
            raise RecordValidationError("aborted records require an abort reason")
        if not _SAFE_DIGEST.fullmatch(self.final_state_fingerprint):
            raise RecordValidationError("final state fingerprint is invalid")
        digest = _sha256(canonical_json_bytes(self.content_dict()))
        if self.content_fingerprint != digest:
            raise RecordIntegrityError("raw record content fingerprint mismatch")
        if self.game_id != digest[:24]:
            raise RecordIntegrityError("game ID does not match record fingerprint")


def create_raw_game_record(**fields: Any) -> RawGameRecord:
    """Create a record whose ID and digest derive only from canonical content."""

    shell = RawGameRecord(
        version=RAW_GAME_RECORD_VERSION,
        game_id="0" * 24,
        content_fingerprint="",
        state_encoding_version=STATE_ENCODING_VERSION,
        action_space_version=ACTION_SPACE_VERSION,
        training_example_version=TRAINING_EXAMPLE_VERSION,
        setup_version=STANDARD_SETUP_VERSION,
        **fields,
    )
    digest = _sha256(canonical_json_bytes(shell.content_dict()))
    record = replace(shell, game_id=digest[:24], content_fingerprint=digest)
    record.validate()
    return record


def raw_game_from_dict(value: object) -> RawGameRecord:
    if not isinstance(value, dict):
        raise RecordValidationError("raw game must be a JSON object")
    expected = {
        "version",
        "game_id",
        "content_fingerprint",
        "state_encoding_version",
        "action_space_version",
        "training_example_version",
        "setup_version",
        "first_player",
        "agents",
        "plies",
        "status",
        "final_result",
        "total_plies",
        "abort_reason",
        "final_state_fingerprint",
        "generator_config",
    }
    if set(value) != expected:
        raise RecordValidationError("raw-game fields are invalid")
    try:
        agents = value["agents"]
        config = value["generator_config"]
        if not isinstance(agents, dict) or not isinstance(config, dict):
            raise RecordValidationError("agents and generator_config must be objects")
        record = RawGameRecord(
            version=_integer(value["version"], "version"),
            game_id=str(value["game_id"]),
            content_fingerprint=str(value["content_fingerprint"]),
            state_encoding_version=_integer(
                value["state_encoding_version"], "state_encoding_version"
            ),
            action_space_version=_integer(
                value["action_space_version"], "action_space_version"
            ),
            training_example_version=_integer(
                value["training_example_version"], "training_example_version"
            ),
            setup_version=_integer(value["setup_version"], "setup_version"),
            first_player=Empire(value["first_player"]),
            agent_a=AgentSpec.from_dict(agents["A"]),
            agent_b=AgentSpec.from_dict(agents["B"]),
            plies=tuple(PlyRecord.from_dict(item) for item in value["plies"]),
            status=str(value["status"]),
            final_result=_parse_result(value["final_result"]),
            total_plies=_integer(value["total_plies"], "total_plies"),
            abort_reason=value["abort_reason"],
            final_state_fingerprint=str(value["final_state_fingerprint"]),
            generator_config=tuple(sorted(config.items())),
        )
    except KeyError as error:
        raise RecordValidationError(
            f"missing raw-game field {error.args[0]}"
        ) from error
    except ValueError as error:
        raise RecordValidationError(f"invalid raw-game enum: {error}") from error
    record.validate()
    return record


def read_raw_game(path: str | Path) -> RawGameRecord:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecordValidationError(f"cannot read raw game {path}: {error}") from error
    return raw_game_from_dict(value)


def write_raw_game(record: RawGameRecord, output_dir: str | Path) -> Path:
    record.validate()
    root = Path(output_dir)
    games = root / "games"
    games.mkdir(parents=True, exist_ok=True)
    path = games / f"{record.game_id}.json"
    data = canonical_json_bytes(record.to_dict())
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise RecordIntegrityError(f"conflicting raw game already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".game-", suffix=".tmp", dir=games)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path
