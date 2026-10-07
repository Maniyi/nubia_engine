"""Versioned neural-MCTS self-play records and bounded corpus generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from nubia_engine import (
    Empire,
    Outcome,
    apply_action,
    create_initial_state,
    format_action,
)
from nubia_training.action_space import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    action_to_index,
)
from nubia_training.datasets import (
    DATASET_MANIFEST_VERSION,
    DatasetManifest,
    build_dataset,
)
from nubia_training.encoding import STATE_ENCODING_VERSION, encode_state
from nubia_training.examples import TRAINING_EXAMPLE_VERSION, TrainingExample
from nubia_training.neural.checkpoints import CHECKPOINT_VERSION, load_checkpoint
from nubia_training.neural.errors import SelfPlayError
from nubia_training.neural.evaluator import NeuralPositionEvaluator, PositionEvaluator
from nubia_training.neural.mcts import (
    MCTS_VERSION,
    MCTSConfig,
    MCTSSearchResult,
    search_state,
)
from nubia_training.neural.model import MODEL_ARCHITECTURE_VERSION
from nubia_training.records import (
    RAW_GAME_RECORD_VERSION,
    AgentSpec,
    PlyRecord,
    RawGameRecord,
    canonical_json_bytes,
    create_raw_game_record,
    raw_game_from_dict,
    read_raw_game,
    write_raw_game,
)
from nubia_training.replay import ProvenancedExample, replay_game
from nubia_training.storage import DATASET_SHARD_VERSION

SELF_PLAY_VERSION = 1
SELF_PLAY_CORPUS_VERSION = 1
POLICY_SUM_TOLERANCE = 1e-6


def _int(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelfPlayError(f"{name} must be an integer")
    if positive and value <= 0:
        raise SelfPlayError(f"{name} must be positive")
    return value


def _number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfPlayError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise SelfPlayError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    checkpoint_path: Path
    output_dir: Path
    game_count: int = 1
    simulations: int = 8
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.25
    root_noise_enabled: bool = True
    temperature_cutoff_plies: int = 20
    early_temperature: float = 1.0
    late_temperature: float = 0.0
    max_game_plies: int = 200
    random_seed: int = 0
    device: str = "cpu"
    first_player_schedule: str = "alternate"
    shard_size: int = 256
    max_search_depth: int = 128
    version: int = SELF_PLAY_VERSION
    state_encoding_version: int = STATE_ENCODING_VERSION
    action_space_version: int = ACTION_SPACE_VERSION
    training_example_version: int = TRAINING_EXAMPLE_VERSION
    raw_game_record_version: int = RAW_GAME_RECORD_VERSION
    dataset_manifest_version: int = DATASET_MANIFEST_VERSION
    dataset_shard_version: int = DATASET_SHARD_VERSION
    model_architecture_version: int = MODEL_ARCHITECTURE_VERSION
    checkpoint_version: int = CHECKPOINT_VERSION
    mcts_version: int = MCTS_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "game_count",
            "simulations",
            "max_game_plies",
            "shard_size",
            "max_search_depth",
        ):
            _int(name, getattr(self, name), positive=True)
        _int("temperature_cutoff_plies", self.temperature_cutoff_plies)
        if self.temperature_cutoff_plies < 0:
            raise SelfPlayError("temperature_cutoff_plies must be non-negative")
        _int("random_seed", self.random_seed)
        _number("c_puct", self.c_puct, positive=True)
        _number("early_temperature", self.early_temperature)
        _number("late_temperature", self.late_temperature)
        _number("root_noise_fraction", self.root_noise_fraction)
        if not 0.0 <= self.root_noise_fraction <= 1.0:
            raise SelfPlayError("root_noise_fraction must be in [0, 1]")
        if not isinstance(self.root_noise_enabled, bool):
            raise SelfPlayError("root_noise_enabled must be Boolean")
        if self.root_noise_enabled:
            _number("root_dirichlet_alpha", self.root_dirichlet_alpha, positive=True)
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise SelfPlayError("invalid device selection")
        if self.first_player_schedule not in {"A", "B", "alternate"}:
            raise SelfPlayError("first_player_schedule must be A, B, or alternate")
        expected = {
            "version": SELF_PLAY_VERSION,
            "state_encoding_version": STATE_ENCODING_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "training_example_version": TRAINING_EXAMPLE_VERSION,
            "raw_game_record_version": RAW_GAME_RECORD_VERSION,
            "dataset_manifest_version": DATASET_MANIFEST_VERSION,
            "dataset_shard_version": DATASET_SHARD_VERSION,
            "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "mcts_version": MCTS_VERSION,
        }
        for name, wanted in expected.items():
            actual = getattr(self, name)
            if isinstance(actual, bool) or actual != wanted:
                raise SelfPlayError(f"{name} must be {wanted}")

    def temperature_for_ply(self, ply_index: int) -> float:
        _int("ply_index", ply_index)
        if ply_index < 0:
            raise SelfPlayError("ply_index must be non-negative")
        if ply_index < self.temperature_cutoff_plies:
            return self.early_temperature
        return self.late_temperature

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["checkpoint_path"] = str(self.checkpoint_path)
        values["output_dir"] = str(self.output_dir)
        return values


@dataclass(frozen=True, slots=True)
class SelfPlayPly:
    ply_index: int
    acting_empire: Empire
    pre_state_fingerprint: str
    selected_action_index: int
    action_indices: tuple[int, ...]
    visit_counts: tuple[int, ...]
    probabilities: tuple[float, ...]
    simulations: int
    evaluator_calls: int
    root_value: float

    def __post_init__(self) -> None:
        _int("ply_index", self.ply_index)
        _int("selected_action_index", self.selected_action_index)
        _int("simulations", self.simulations, positive=True)
        _int("evaluator_calls", self.evaluator_calls, positive=True)
        if not self.action_indices or not (
            len(self.action_indices)
            == len(self.visit_counts)
            == len(self.probabilities)
        ):
            raise SelfPlayError("policy arrays must be non-empty and aligned")
        if len(set(self.action_indices)) != len(self.action_indices):
            raise SelfPlayError("policy action indices must be unique")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < ACTION_SPACE_SIZE
            for index in self.action_indices
        ):
            raise SelfPlayError("policy contains an invalid action index")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.visit_counts
        ):
            raise SelfPlayError("persisted visit counts must be positive integers")
        if sum(self.visit_counts) != self.simulations:
            raise SelfPlayError("visit counts must sum to simulations")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in self.probabilities
        ):
            raise SelfPlayError("policy probabilities must be finite and positive")
        if not math.isclose(
            sum(self.probabilities), 1.0, abs_tol=POLICY_SUM_TOLERANCE, rel_tol=0.0
        ):
            raise SelfPlayError("policy probabilities must sum to one")
        if self.selected_action_index not in self.action_indices:
            raise SelfPlayError("selected action must have positive root visits")
        if not math.isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise SelfPlayError("root_value must be finite and in [-1, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "ply_index": self.ply_index,
            "acting_empire": self.acting_empire.value,
            "pre_state_fingerprint": self.pre_state_fingerprint,
            "selected_action_index": self.selected_action_index,
            "action_indices": list(self.action_indices),
            "visit_counts": list(self.visit_counts),
            "probabilities": list(self.probabilities),
            "simulations": self.simulations,
            "evaluator_calls": self.evaluator_calls,
            "root_value": self.root_value,
        }

    @classmethod
    def from_dict(cls, value: object) -> SelfPlayPly:
        if not isinstance(value, dict):
            raise SelfPlayError("self-play ply must be an object")
        try:
            indices = value["action_indices"]
            counts = value["visit_counts"]
            probabilities = value["probabilities"]
            if (
                not isinstance(indices, list)
                or not isinstance(counts, list)
                or not isinstance(probabilities, list)
            ):
                raise TypeError("policy arrays must be lists")
            return cls(
                _int("ply_index", value["ply_index"]),
                Empire(value["acting_empire"]),
                str(value["pre_state_fingerprint"]),
                _int("selected_action_index", value["selected_action_index"]),
                tuple(_int("action_index", item) for item in indices),
                tuple(_int("visit_count", item, positive=True) for item in counts),
                tuple(float(item) for item in probabilities),
                _int("simulations", value["simulations"], positive=True),
                _int("evaluator_calls", value["evaluator_calls"], positive=True),
                float(value["root_value"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SelfPlayError(f"invalid self-play ply: {error}") from error


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    raw_game: RawGameRecord
    source_checkpoint: str
    source_checkpoint_sha256: str
    game_seed: int
    game_index: int
    first_player_schedule: str
    policies: tuple[SelfPlayPly, ...]
    status: str
    termination_reason: str
    version: int = SELF_PLAY_VERSION
    model_architecture_version: int = MODEL_ARCHITECTURE_VERSION
    checkpoint_version: int = CHECKPOINT_VERSION
    mcts_version: int = MCTS_VERSION
    fingerprint: str = ""

    def content_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_architecture_version": self.model_architecture_version,
            "checkpoint_version": self.checkpoint_version,
            "mcts_version": self.mcts_version,
            "source_checkpoint": self.source_checkpoint,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "game_seed": self.game_seed,
            "game_index": self.game_index,
            "first_player_schedule": self.first_player_schedule,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "raw_game": self.raw_game.to_dict(),
            "policies": [policy.to_dict() for policy in self.policies],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_dict(), "fingerprint": self.fingerprint}

    def validate(self) -> None:
        self.raw_game.validate()
        if self.version != SELF_PLAY_VERSION or self.mcts_version != MCTS_VERSION:
            raise SelfPlayError("unsupported self-play or MCTS version")
        if self.model_architecture_version != MODEL_ARCHITECTURE_VERSION:
            raise SelfPlayError("unsupported model architecture version")
        if self.checkpoint_version != CHECKPOINT_VERSION:
            raise SelfPlayError("unsupported checkpoint version")
        if len(self.source_checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_checkpoint_sha256
        ):
            raise SelfPlayError("source checkpoint fingerprint is invalid")
        if self.status not in {"completed", "truncated"}:
            raise SelfPlayError("self-play status must be completed or truncated")
        if len(self.policies) != self.raw_game.total_plies:
            raise SelfPlayError("policies must align with raw-game plies")
        if (self.status == "completed") != (self.raw_game.status == "completed"):
            raise SelfPlayError("self-play and raw-game completion status disagree")
        if self.termination_reason != _termination_reason(self.raw_game):
            raise SelfPlayError("self-play termination reason mismatch")
        digest = hashlib.sha256(canonical_json_bytes(self.content_dict())).hexdigest()
        if self.fingerprint != digest:
            raise SelfPlayError("self-play game fingerprint mismatch")
        _validate_game_alignment(self)


def _new_game(**fields: Any) -> SelfPlayGame:
    shell = SelfPlayGame(**fields)
    game = replace(
        shell,
        fingerprint=hashlib.sha256(
            canonical_json_bytes(shell.content_dict())
        ).hexdigest(),
    )
    game.validate()
    return game


def _validate_game_alignment(game: SelfPlayGame) -> None:
    report = replay_game(game.raw_game)
    for raw_ply, policy, state in zip(
        game.raw_game.plies, game.policies, report.states[:-1], strict=True
    ):
        if policy.ply_index != raw_ply.ply_index:
            raise SelfPlayError("policy ply index mismatch")
        if policy.acting_empire is not raw_ply.acting_empire:
            raise SelfPlayError("policy acting perspective mismatch")
        if policy.selected_action_index != raw_ply.action_index:
            raise SelfPlayError("selected action mismatch")
        fingerprint = encode_state(state, perspective=state.side_to_move).fingerprint()
        if policy.pre_state_fingerprint != fingerprint:
            raise SelfPlayError("policy state fingerprint mismatch")
        legal = set(int(item) for item in encode_state(state).legal_action_indices)
        if not set(policy.action_indices) <= legal:
            raise SelfPlayError("policy contains an illegal action index")


def self_play_examples(game: SelfPlayGame) -> tuple[ProvenancedExample, ...]:
    game.validate()
    if game.status != "completed" or game.raw_game.final_result is None:
        return ()
    report = replay_game(game.raw_game)
    examples: list[ProvenancedExample] = []
    for policy, state in zip(game.policies, report.states[:-1], strict=True):
        result = game.raw_game.final_result
        value = (
            0.0
            if result.outcome is Outcome.DRAW
            else 1.0
            if result.winner is policy.acting_empire
            else -1.0
        )
        example = TrainingExample(
            encode_state(state, perspective=policy.acting_empire),
            np.asarray(policy.action_indices, dtype=np.uint16),
            np.asarray(policy.probabilities, dtype=np.float32),
            value,
        )
        examples.append(
            ProvenancedExample(
                game.raw_game.game_id,
                policy.ply_index,
                policy.acting_empire,
                f"neural-mcts:{game.source_checkpoint_sha256[:12]}",
                example,
            )
        )
    return tuple(examples)


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise SelfPlayError(f"conflicting artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".self-play-", suffix=".tmp", dir=path.parent
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


def write_self_play_game(game: SelfPlayGame, output_dir: str | Path) -> Path:
    game.validate()
    root = Path(output_dir)
    write_raw_game(game.raw_game, root)
    path = root / "self_play_games" / f"{game.raw_game.game_id}.json"
    _atomic_write(path, canonical_json_bytes(game.to_dict()))
    return path


def read_self_play_game(path: str | Path) -> SelfPlayGame:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise SelfPlayError("self-play game must be an object")
        policies = value["policies"]
        if not isinstance(policies, list):
            raise SelfPlayError("self-play policies must be a list")
        game = SelfPlayGame(
            raw_game_from_dict(value["raw_game"]),
            str(value["source_checkpoint"]),
            str(value["source_checkpoint_sha256"]),
            _int("game_seed", value["game_seed"]),
            _int("game_index", value["game_index"]),
            str(value["first_player_schedule"]),
            tuple(SelfPlayPly.from_dict(item) for item in policies),
            str(value["status"]),
            str(value["termination_reason"]),
            _int("version", value["version"]),
            _int("model_architecture_version", value["model_architecture_version"]),
            _int("checkpoint_version", value["checkpoint_version"]),
            _int("mcts_version", value["mcts_version"]),
            str(value["fingerprint"]),
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SelfPlayError(f"cannot read self-play game: {error}") from error
    game.validate()
    return game


@dataclass(frozen=True, slots=True)
class SelfPlayCorpus:
    root: Path
    manifest: dict[str, object]
    games: tuple[SelfPlayGame, ...]

    @property
    def corpus_id(self) -> str:
        return str(self.manifest["corpus_id"])

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["corpus_fingerprint"])


def _first_player(config: SelfPlayConfig, game_index: int) -> Empire:
    if config.first_player_schedule == "alternate":
        return Empire.A if game_index % 2 == 0 else Empire.B
    return Empire(config.first_player_schedule)


def _termination_reason(record: RawGameRecord) -> str:
    if record.final_result is None:
        return record.abort_reason or "incomplete"
    return "+".join(reason.value for reason in record.final_result.reasons)


def _persisted_policy(
    result: MCTSSearchResult, ply: int, actor: Empire, pre: str
) -> SelfPlayPly:
    positive = result.policy.visit_counts > 0
    indices = tuple(int(item) for item in result.policy.action_indices[positive])
    counts = tuple(int(item) for item in result.policy.visit_counts[positive])
    probabilities = tuple(
        float(item)
        for item in (np.asarray(counts, dtype=np.float64) / sum(counts)).astype(
            np.float32
        )
    )
    return SelfPlayPly(
        ply,
        actor,
        pre,
        result.action_index,
        indices,
        counts,
        probabilities,
        result.simulations,
        result.evaluator_calls,
        result.root_value,
    )


def play_self_play_game(
    config: SelfPlayConfig,
    evaluator: PositionEvaluator,
    *,
    source_checkpoint_sha256: str,
    game_index: int,
    search: Callable[
        [object, PositionEvaluator, MCTSConfig], MCTSSearchResult
    ] = search_state,  # type: ignore[assignment]
) -> SelfPlayGame:
    """Play one fresh-search-per-ply game and delay values until termination."""
    game_seed = config.random_seed + game_index
    opening = _first_player(config, game_index)
    state = create_initial_state(opening)
    agent = AgentSpec(
        "iterative",
        f"NeuralMCTS[{source_checkpoint_sha256[:12]}]",
        tuple(
            sorted(
                {
                    "max_depth": config.max_search_depth,
                    "mcts_version": MCTS_VERSION,
                    "simulations": config.simulations,
                }.items()
            )
        ),
    )
    raw_plies: list[PlyRecord] = []
    policies: list[SelfPlayPly] = []
    while state.result is None and len(raw_plies) < config.max_game_plies:
        actor = state.side_to_move
        ply = len(raw_plies)
        search_config = MCTSConfig(
            simulations=config.simulations,
            c_puct=config.c_puct,
            temperature=config.temperature_for_ply(ply),
            max_depth=config.max_search_depth,
            root_dirichlet_alpha=config.root_dirichlet_alpha,
            root_noise_fraction=config.root_noise_fraction,
            random_seed=game_seed * 1_000_003 + ply,
            root_noise_enabled=config.root_noise_enabled,
        )
        result = search(state, evaluator, search_config)
        legal = set(int(item) for item in encode_state(state).legal_action_indices)
        if result.action_index not in legal:
            raise SelfPlayError("MCTS selected an illegal action")
        pre = encode_state(state, perspective=actor).fingerprint()
        successor = apply_action(state, result.action)
        raw_plies.append(
            PlyRecord(
                ply,
                actor,
                action_to_index(state, result.action, perspective=actor),
                ACTION_SPACE_VERSION,
                format_action(state, result.action),
                pre,
                encode_state(successor, perspective=actor).fingerprint(),
                result.action.kind,
            )
        )
        policies.append(_persisted_policy(result, ply, actor, pre))
        state = successor
    completed = state.result is not None
    raw = create_raw_game_record(
        first_player=opening,
        agent_a=agent,
        agent_b=agent,
        plies=tuple(raw_plies),
        status="completed" if completed else "aborted",
        final_result=state.result,
        total_plies=len(raw_plies),
        abort_reason=None if completed else "maximum_plies_reached",
        final_state_fingerprint=encode_state(state).fingerprint(),
        generator_config=tuple(
            sorted(
                {
                    "game_index": game_index,
                    "game_seed": game_seed,
                    "mcts_version": MCTS_VERSION,
                    "self_play_version": SELF_PLAY_VERSION,
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                }.items()
            )
        ),
    )
    return _new_game(
        raw_game=raw,
        source_checkpoint=str(config.checkpoint_path),
        source_checkpoint_sha256=source_checkpoint_sha256,
        game_seed=game_seed,
        game_index=game_index,
        first_player_schedule=config.first_player_schedule,
        policies=tuple(policies),
        status="completed" if completed else "truncated",
        termination_reason=_termination_reason(raw),
    )


def generate_self_play_corpus(
    config: SelfPlayConfig,
    *,
    evaluator: PositionEvaluator | None = None,
    progress: Callable[[int, SelfPlayGame], None] | None = None,
) -> SelfPlayCorpus:
    loaded = load_checkpoint(config.checkpoint_path)
    active_evaluator = evaluator or NeuralPositionEvaluator(
        loaded.model, device=config.device
    )
    games: list[SelfPlayGame] = []
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    for game_index in range(config.game_count):
        try:
            game = play_self_play_game(
                config,
                active_evaluator,
                source_checkpoint_sha256=loaded.sha256,
                game_index=game_index,
            )
            write_self_play_game(game, config.output_dir)
            games.append(game)
            if progress is not None:
                progress(game_index, game)
        except Exception as error:
            failures.append({"game_index": game_index, "error": type(error).__name__})
    completed = [game for game in games if game.status == "completed"]
    truncated = [game for game in games if game.status == "truncated"]
    outcomes: Counter[str] = Counter()
    for game in completed:
        result = game.raw_game.final_result
        if result is not None and result.outcome is Outcome.DRAW:
            outcomes["draw"] += 1
        elif result is not None and result.winner is Empire.A:
            outcomes["win_A"] += 1
        else:
            outcomes["win_B"] += 1
    total_plies = sum(game.raw_game.total_plies for game in games)
    content: dict[str, object] = {
        "version": SELF_PLAY_CORPUS_VERSION,
        "self_play_version": SELF_PLAY_VERSION,
        "source_checkpoint": str(config.checkpoint_path),
        "source_checkpoint_sha256": loaded.sha256,
        "config": config.to_dict(),
        "games": [
            {
                "game_id": game.raw_game.game_id,
                "fingerprint": game.fingerprint,
                "status": game.status,
            }
            for game in games
        ],
        "attempted_games": config.game_count,
        "completed_games": len(completed),
        "truncated_games": len(truncated),
        "failed_games": len(failures),
        "failures": failures,
        "total_plies": total_plies,
        "usable_examples": sum(game.raw_game.total_plies for game in completed),
        "result_distribution": dict(sorted(outcomes.items())),
        "average_game_length": total_plies / len(games) if games else 0.0,
        "maximum_game_length": max(
            (game.raw_game.total_plies for game in games), default=0
        ),
        "evaluator_calls": sum(
            policy.evaluator_calls for game in games for policy in game.policies
        ),
    }
    digest = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    manifest = {
        **content,
        "corpus_id": digest[:24],
        "corpus_fingerprint": digest,
    }
    _atomic_write(
        config.output_dir / "self-play-manifest.json",
        canonical_json_bytes(manifest),
    )
    diagnostics_path = config.output_dir / "self-play-diagnostics.json"
    if not diagnostics_path.exists():
        _atomic_write(
            diagnostics_path,
            canonical_json_bytes(
                {
                    "corpus_fingerprint": digest,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
        )
    return SelfPlayCorpus(config.output_dir, manifest, tuple(games))


def load_self_play_corpus(path: str | Path) -> SelfPlayCorpus:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "self-play-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelfPlayError(f"cannot read self-play manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != SELF_PLAY_CORPUS_VERSION
    ):
        raise SelfPlayError("unsupported self-play corpus manifest")
    config_value = manifest.get("config")
    if not isinstance(config_value, dict):
        raise SelfPlayError("self-play corpus configuration is invalid")
    try:
        SelfPlayConfig(**config_value)
    except (TypeError, ValueError) as error:
        raise SelfPlayError(
            f"invalid self-play corpus configuration: {error}"
        ) from error
    content = {
        key: value
        for key, value in manifest.items()
        if key not in {"corpus_id", "corpus_fingerprint"}
    }
    digest = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if (
        manifest.get("corpus_fingerprint") != digest
        or manifest.get("corpus_id") != digest[:24]
    ):
        raise SelfPlayError("self-play corpus fingerprint mismatch")
    entries = manifest.get("games")
    if not isinstance(entries, list):
        raise SelfPlayError("self-play corpus games must be a list")
    counts = {
        name: manifest.get(name)
        for name in (
            "attempted_games",
            "completed_games",
            "truncated_games",
            "failed_games",
        )
    }
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts.values()
    ):
        raise SelfPlayError("self-play corpus counts must be non-negative integers")
    attempted = counts["attempted_games"]
    completed = counts["completed_games"]
    truncated = counts["truncated_games"]
    failed = counts["failed_games"]
    if (
        not isinstance(attempted, int)
        or not isinstance(completed, int)
        or not isinstance(truncated, int)
        or not isinstance(failed, int)
        or attempted != len(entries) + failed
        or len(entries) != completed + truncated
    ):
        raise SelfPlayError("self-play corpus counts are inconsistent")
    games: list[SelfPlayGame] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SelfPlayError("invalid self-play game entry")
        game_id = entry.get("game_id")
        if not isinstance(game_id, str):
            raise SelfPlayError("invalid self-play game ID")
        game = read_self_play_game(root / "self_play_games" / f"{game_id}.json")
        raw = read_raw_game(root / "games" / f"{game_id}.json")
        if raw.content_fingerprint != game.raw_game.content_fingerprint:
            raise SelfPlayError("canonical raw-game fingerprint mismatch")
        if game.fingerprint != entry.get("fingerprint"):
            raise SelfPlayError("self-play game fingerprint mismatch")
        games.append(game)
    return SelfPlayCorpus(root, manifest, tuple(games))


def build_self_play_dataset(
    corpus: SelfPlayCorpus | str | Path,
    output_dir: str | Path,
    *,
    shard_size: int | None = None,
) -> DatasetManifest:
    loaded = (
        corpus if isinstance(corpus, SelfPlayCorpus) else load_self_play_corpus(corpus)
    )
    by_id = {game.raw_game.game_id: game for game in loaded.games}

    def convert(record: RawGameRecord) -> tuple[ProvenancedExample, ...]:
        return self_play_examples(by_id[record.game_id])

    if shard_size is None:
        config = loaded.manifest.get("config")
        shard_size = (
            int(config.get("shard_size", 256)) if isinstance(config, dict) else 256
        )
    return build_dataset(
        tuple(game.raw_game for game in loaded.games),
        output_dir,
        shard_size=shard_size,
        build_config=(
            ("self_play_corpus_fingerprint", loaded.fingerprint),
            ("self_play_version", SELF_PLAY_VERSION),
        ),
        example_converter=convert,
    )


def inspect_self_play(path: str | Path) -> dict[str, object]:
    return dict(load_self_play_corpus(path).manifest)
