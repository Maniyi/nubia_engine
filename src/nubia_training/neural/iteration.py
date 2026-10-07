"""One bounded self-play, dataset, and candidate-training iteration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from nubia_training.neural.checkpoints import load_checkpoint, save_checkpoint
from nubia_training.neural.data import ShardDataset
from nubia_training.neural.errors import IterationError
from nubia_training.neural.self_play import (
    SELF_PLAY_VERSION,
    SelfPlayConfig,
    build_self_play_dataset,
    generate_self_play_corpus,
)
from nubia_training.neural.training import TrainingConfig, train_model
from nubia_training.records import canonical_json_bytes
from nubia_training.storage import file_sha256

TRAINING_ITERATION_VERSION = 1


def _atomic_json(path: Path, value: object) -> None:
    data = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise IterationError(f"conflicting iteration artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".iteration-", suffix=".tmp", dir=path.parent
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


@dataclass(frozen=True, slots=True)
class IterationConfig:
    source_checkpoint: Path
    output_dir: Path
    self_play: SelfPlayConfig
    training: TrainingConfig
    iteration_id: str | None = None
    resume_checkpoint: Path | None = None
    version: int = TRAINING_ITERATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_checkpoint", Path(self.source_checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))
        if self.version != TRAINING_ITERATION_VERSION:
            raise IterationError(
                f"iteration configuration version must be {TRAINING_ITERATION_VERSION}"
            )
        if self.self_play.checkpoint_path != self.source_checkpoint:
            raise IterationError("self-play source checkpoint mismatch")
        if self.iteration_id is not None and (
            not self.iteration_id
            or any(
                char
                not in (
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                )
                for char in self.iteration_id
            )
        ):
            raise IterationError("iteration_id contains unsafe characters")

    def identity_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_checkpoint": str(self.source_checkpoint),
            "output_dir": str(self.output_dir),
            "self_play": self.self_play.to_dict(),
            "training": self.training.to_dict(),
            "resume_checkpoint": (
                None if self.resume_checkpoint is None else str(self.resume_checkpoint)
            ),
        }

    @property
    def run_id(self) -> str:
        if self.iteration_id is not None:
            return self.iteration_id
        digest = hashlib.sha256(canonical_json_bytes(self.identity_dict())).hexdigest()
        return digest[:24]


@dataclass(frozen=True, slots=True)
class IterationResult:
    summary: dict[str, object]
    summary_path: Path
    candidate_checkpoint: Path


def _parameters_changed(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> bool:
    return any(
        not torch.equal(before[name], after[name].detach().cpu()) for name in before
    )


def _train_candidate(
    config: IterationConfig,
    dataset: ShardDataset,
    candidate: Path,
    *,
    corpus_fingerprint: str,
) -> tuple[str, dict[str, object]]:
    resume = config.resume_checkpoint
    checkpoint_path = config.source_checkpoint if resume is None else resume
    loaded = load_checkpoint(
        checkpoint_path,
        expected_dataset_id=dataset.dataset_id if resume is not None else None,
        expected_dataset_fingerprint=(
            dataset.dataset_fingerprint if resume is not None else None
        ),
    )
    optimizer = loaded.restore_optimizer() if resume is not None else None
    before = {
        name: value.detach().cpu().clone()
        for name, value in loaded.model.state_dict().items()
    }
    result = train_model(
        loaded.model,
        dataset,
        config.training,
        optimizer=optimizer,
        initial_completed_epochs=(
            int(loaded.metadata["completed_epoch"]) if resume is not None else 0
        ),
        initial_optimizer_steps=(
            int(loaded.metadata["optimizer_step"]) if resume is not None else 0
        ),
        initial_examples_seen=(
            int(loaded.metadata["examples_seen"]) if resume is not None else 0
        ),
    )
    changed = _parameters_changed(before, loaded.model.state_dict())
    if not changed:
        raise IterationError("training completed without changing model parameters")
    provenance: dict[str, Any] = {
        "artifact_role": "candidate",
        "iteration_version": TRAINING_ITERATION_VERSION,
        "iteration_id": config.run_id,
        "source_checkpoint_sha256": file_sha256(config.source_checkpoint),
        "self_play_corpus_fingerprint": corpus_fingerprint,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "resumed_from_sha256": None if resume is None else file_sha256(resume),
    }
    checksum = save_checkpoint(
        candidate,
        loaded.model,
        config.training,
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.dataset_fingerprint,
        completed_epoch=result.summary.completed_epochs,
        optimizer_step=result.summary.optimizer_steps,
        examples_seen=result.summary.examples_seen,
        latest_metrics=result.summary.latest_metrics,
        optimizer=result.optimizer,
        provenance=provenance,
    )
    metrics: dict[str, object] = {
        "selected_device": result.summary.selected_device,
        "amp_enabled": result.summary.amp_enabled,
        "optimizer_steps": result.summary.optimizer_steps,
        "examples_seen": result.summary.examples_seen,
        "completed_epochs": result.summary.completed_epochs,
        "initial_metrics": result.summary.initial_metrics,
        "final_metrics": result.summary.latest_metrics,
        "parameters_changed": changed,
        "peak_cuda_memory_allocated": result.summary.peak_cuda_memory_allocated,
        "peak_cuda_memory_reserved": result.summary.peak_cuda_memory_reserved,
    }
    return checksum, metrics


def run_iteration(config: IterationConfig) -> IterationResult:
    """Run exactly one bounded iteration, publish a candidate, and stop."""
    source_sha = file_sha256(config.source_checkpoint)
    root = config.output_dir
    candidate = root / "candidate.pt"
    if candidate.resolve() == config.source_checkpoint.resolve():
        raise IterationError("candidate checkpoint cannot overwrite the source")
    if (
        candidate.exists()
        or candidate.with_name(candidate.name + ".sha256.json").exists()
    ):
        raise IterationError("candidate checkpoint already exists")
    durations: dict[str, float] = {}
    stage = time.perf_counter()
    corpus_config = replace(config.self_play, output_dir=root / "self_play")
    corpus = generate_self_play_corpus(corpus_config)
    durations["self_play_seconds"] = time.perf_counter() - stage
    completed_games = corpus.manifest["completed_games"]
    if (
        isinstance(completed_games, bool)
        or not isinstance(completed_games, int)
        or completed_games <= 0
    ):
        raise IterationError("self-play produced no completed games")
    stage = time.perf_counter()
    manifest = build_self_play_dataset(corpus, root / "dataset")
    dataset = ShardDataset(root / "dataset")
    durations["dataset_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    candidate_sha, training = _train_candidate(
        config,
        dataset,
        candidate,
        corpus_fingerprint=corpus.fingerprint,
    )
    durations["training_seconds"] = time.perf_counter() - stage
    if file_sha256(config.source_checkpoint) != source_sha:
        raise IterationError("source checkpoint changed during iteration")
    content: dict[str, object] = {
        "version": TRAINING_ITERATION_VERSION,
        "iteration_id": config.run_id,
        "source_checkpoint": str(config.source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": candidate_sha,
        "candidate_semantics": "unpromoted_candidate",
        "self_play_version": SELF_PLAY_VERSION,
        "self_play_config": corpus_config.to_dict(),
        "self_play_corpus": str(corpus.root),
        "self_play_corpus_fingerprint": corpus.fingerprint,
        "games_attempted": corpus.manifest["attempted_games"],
        "games_completed": corpus.manifest["completed_games"],
        "games_truncated": corpus.manifest["truncated_games"],
        "examples_produced": corpus.manifest["usable_examples"],
        "dataset_path": str(root / "dataset"),
        "dataset_id": manifest.dataset_id,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "training_config": config.training.to_dict(),
        "training": training,
        "artifacts": {
            "candidate": str(candidate),
            "corpus": str(corpus.root),
            "dataset": str(root / "dataset"),
        },
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    summary = {
        **content,
        "iteration_fingerprint": fingerprint,
        "diagnostic_durations": durations,
    }
    summary_path = root / "iteration-summary.json"
    _atomic_json(summary_path, summary)
    _atomic_json(
        root / "iteration-diagnostics.json",
        {"iteration_fingerprint": fingerprint, "durations": durations},
    )
    return IterationResult(summary, summary_path, candidate)


def inspect_iteration(path: str | Path) -> dict[str, object]:
    summary_path = Path(path)
    if summary_path.is_dir():
        summary_path /= "iteration-summary.json"
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IterationError(f"cannot read iteration summary: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("version") != TRAINING_ITERATION_VERSION
    ):
        raise IterationError("unsupported iteration summary")
    fingerprint = value.get("iteration_fingerprint")
    content = {
        key: item
        for key, item in value.items()
        if key not in {"iteration_fingerprint", "diagnostic_durations"}
    }
    if fingerprint != hashlib.sha256(canonical_json_bytes(content)).hexdigest():
        raise IterationError("iteration summary fingerprint mismatch")
    return value
