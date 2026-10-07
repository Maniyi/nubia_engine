from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nubia_engine import Empire, create_initial_state, legal_actions
from nubia_training import legal_action_indices
from nubia_training.neural import (
    ModelConfig,
    NeuralPositionEvaluator,
    PolicyValueNetwork,
    PositionEvaluation,
    TrainingConfig,
    save_checkpoint,
)
from nubia_training.neural.errors import NeuralInferenceError


def _model() -> PolicyValueNetwork:
    torch.manual_seed(7)
    return PolicyValueNetwork(
        ModelConfig(
            trunk_channels=8,
            residual_blocks=1,
            policy_embedding_dim=2,
            value_hidden_dim=8,
            normalization_groups=2,
        )
    )


def test_in_memory_cpu_evaluation_is_sparse_deterministic_and_inference_only() -> None:
    state = create_initial_state(Empire.B)
    evaluator = NeuralPositionEvaluator(_model(), device="cpu")
    first = evaluator.evaluate(state)
    second = evaluator.evaluate(state)
    assert first == second
    assert first.perspective is Empire.B
    assert first.action_indices == legal_action_indices(state)
    assert len(first.priors) == len(legal_actions(state))
    assert np.isclose(sum(first.priors), 1.0)
    assert all(np.isfinite(first.priors))
    assert -1.0 <= first.value <= 1.0
    assert not evaluator.model.training
    assert all(parameter.grad is None for parameter in evaluator.model.parameters())


def test_checkpoint_evaluation_and_terminal_rejection(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        _model(),
        TrainingConfig(),
        dataset_id="smoke",
        dataset_fingerprint="0" * 64,
        completed_epoch=0,
        optimizer_step=0,
        examples_seen=0,
        latest_metrics={},
    )
    evaluator = NeuralPositionEvaluator.from_checkpoint(path, device="cpu")
    result = evaluator.evaluate(create_initial_state(Empire.A))
    assert result.checkpoint_version == 1
    terminal = create_initial_state(Empire.A)
    object.__setattr__(
        terminal,
        "result",
        __import__("nubia_engine").GameResult(
            __import__("nubia_engine").Outcome.DRAW,
            None,
            (__import__("nubia_engine").ResultReason.THREEFOLD_REPETITION,),
        ),
    )
    with pytest.raises(NeuralInferenceError, match="terminal"):
        evaluator.evaluate(terminal)


@pytest.mark.parametrize(
    ("indices", "priors", "value", "message"),
    [
        ((), (), 0.0, "contain legal"),
        ((1,), (0.5, 0.5), 0.0, "align"),
        ((1, 1), (0.5, 0.5), 0.0, "unique"),
        ((True,), (1.0,), 0.0, "integers"),
        ((60_000,), (1.0,), 0.0, "outside"),
        ((1,), (float("nan"),), 0.0, "finite"),
        ((1,), (-1.0,), 0.0, "non-negative"),
        ((1,), (0.5,), 0.0, "sum"),
        ((1,), (1.0,), 2.0, "value"),
    ],
)
def test_position_evaluation_validation(
    indices: tuple[int, ...],
    priors: tuple[float, ...],
    value: float,
    message: str,
) -> None:
    with pytest.raises(NeuralInferenceError, match=message):
        PositionEvaluation(indices, priors, value, Empire.A)


def test_position_evaluation_metadata_and_aliases() -> None:
    evaluation = PositionEvaluation((1,), (1.0,), 0.0, Empire.A)
    assert evaluation.legal_action_indices == (1,)
    assert evaluation.prior_probabilities == (1.0,)
    assert evaluation.evaluation_perspective is Empire.A
    with pytest.raises(NeuralInferenceError, match="state encoding"):
        PositionEvaluation((1,), (1.0,), 0.0, Empire.A, state_encoding_version=2)
    with pytest.raises(NeuralInferenceError, match="action-space"):
        PositionEvaluation((1,), (1.0,), 0.0, Empire.A, action_space_version=2)
    with pytest.raises(NeuralInferenceError, match="immutable tuple"):
        PositionEvaluation((1,), [1.0], 0.0, Empire.A)  # type: ignore[arg-type]
    with pytest.raises(NeuralInferenceError, match="positive integer"):
        PositionEvaluation((1,), (1.0,), 0.0, Empire.A, model_architecture_version=True)
