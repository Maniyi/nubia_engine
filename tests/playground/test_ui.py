from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from nubia_engine import Square
from nubia_playground.app import PlaygroundApp
from nubia_playground.board_view import BoardGeometry
from nubia_playground.controller import GameController, SetupConfig


@pytest.mark.parametrize("square", [Square(0, 0), Square(4, 7), Square(9, 9)])
def test_board_pixel_coordinate_round_trip(square: Square) -> None:
    geometry = BoardGeometry(31, 57, 61)
    assert geometry.pixel_to_square(geometry.square_center(square)) == square


@pytest.mark.parametrize("point", [(30, 57), (31, 56), (641, 57), (31, 667)])
def test_clicks_outside_board_are_ignored(point: tuple[int, int]) -> None:
    assert BoardGeometry(31, 57, 61).pixel_to_square(point) is None


def test_return_to_setup_clears_active_game() -> None:
    app = PlaygroundApp()
    try:
        app.controller = GameController(SetupConfig())
        app._to_setup()
        assert app.controller is None
    finally:
        app.running = False


def test_invalid_setup_value_is_visible_instead_of_crashing() -> None:
    app = PlaygroundApp()
    try:
        app.setup.values["delay"] = "oops"
        app._start()
        assert app.controller is None
        assert app.setup.error is not None and "Invalid setup" in app.setup.error
    finally:
        app.running = False


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "nubia_playground", "--smoke-test"],
        [str(Path(sys.executable).with_name("nubia-playground")), "--smoke-test"],
    ],
)
def test_entry_points_start_and_close_headlessly(command: list[str]) -> None:
    environment = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def test_existing_packages_do_not_import_pygame() -> None:
    code = "import sys, nubia_engine, nubia_ai; assert 'pygame' not in sys.modules"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_browsing_checkpoint_path_does_not_import_torch() -> None:
    code = r"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import nubia_playground.app as playground_app

with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_name(checkpoint.name + ".sha256.json").write_text("{}")
    playground_app.select_checkpoint_file = lambda initial_directory: str(checkpoint)
    app = playground_app.PlaygroundApp()
    app._browse_checkpoint()
    assert app.setup.checkpoint == str(checkpoint.resolve())
    assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
