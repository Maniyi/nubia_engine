from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from nubia_engine import (
    BrainwashAvailability,
    Empire,
    GameState,
    Piece,
    PieceType,
    Square,
)
from nubia_training import build_dataset
from nubia_training.generation import agent_spec, generate_game
from nubia_training.neural import ModelConfig


# Existing training tests import these root helpers as ``from conftest``. Pytest
# places this nested configuration on sys.path during full-suite collection, so
# keep equivalent helpers available rather than shadowing that public test setup.
def piece(
    piece_id: str,
    piece_type: PieceType,
    empire: Empire = Empire.A,
    *,
    original_empire: Empire | None = None,
    brainwash: BrainwashAvailability = BrainwashAvailability.AVAILABLE,
) -> Piece:
    power = brainwash if piece_type is PieceType.WEST_AFRICAN_MYSTIC else None
    return Piece(
        piece_id,
        piece_type,
        empire if original_empire is None else original_empire,
        empire,
        power,
    )


def state_with(
    placements: Iterable[tuple[Square, Piece]],
    side: Empire = Empire.A,
    *,
    quiet: int = 0,
    ply: int = 0,
) -> GameState:
    board: list[Piece | None] = [None] * 100
    for square, placed_piece in placements:
        board[square.row * 10 + square.column] = placed_piece
    return GameState(tuple(board), side, quiet, ply)


@pytest.fixture(scope="session")
def neural_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("neural-dataset")
    record = generate_game(
        agent_spec("random", seed=1),
        agent_spec("random", seed=2),
        first_player=Empire.A,
        max_plies=100,
    )
    assert record.status == "completed"
    build_dataset((record,), root, shard_size=7)
    return root


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(
        trunk_channels=8,
        residual_blocks=1,
        policy_embedding_dim=2,
        value_hidden_dim=8,
        normalization_groups=2,
    )
