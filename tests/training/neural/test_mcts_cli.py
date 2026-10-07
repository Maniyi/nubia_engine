from __future__ import annotations

import json
from pathlib import Path

import pytest

from nubia_training.neural import (
    ModelConfig,
    PolicyValueNetwork,
    TrainingConfig,
    save_checkpoint,
)
from nubia_training.neural.cli import main


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    model = PolicyValueNetwork(
        ModelConfig(
            trunk_channels=8,
            residual_blocks=1,
            policy_embedding_dim=2,
            value_hidden_dim=8,
            normalization_groups=2,
        )
    )
    save_checkpoint(
        path,
        model,
        TrainingConfig(),
        dataset_id="cli-smoke",
        dataset_fingerprint="0" * 64,
        completed_epoch=0,
        optimizer_step=0,
        examples_seen=0,
        latest_metrics={},
    )
    return path


def test_search_cli_is_bounded_and_writes_no_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _checkpoint(tmp_path)
    before = set(tmp_path.iterdir())
    assert main(["search", str(checkpoint), "--simulations", "2"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["simulations"] == 2
    assert output["action_index"] >= 0
    assert set(tmp_path.iterdir()) == before


def test_search_cli_rejects_invalid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main(["search", str(checkpoint), "--simulations", "0"])
    assert raised.value.code == 2
    assert "simulations" in capsys.readouterr().err


def test_benchmark_safety_limit_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "benchmark",
                str(checkpoint),
                "--simulations",
                "1",
                "--max-plies",
                "1",
            ]
        )
    assert raised.value.code == 2
    assert "without an engine result" in capsys.readouterr().err
