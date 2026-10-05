from __future__ import annotations

from pathlib import Path

import nubia_ai
import nubia_engine


def test_both_packages_import() -> None:
    assert nubia_ai.RandomAgent.__module__ == "nubia_ai.random_agent"
    assert nubia_engine.GameState.__module__ == "nubia_engine.models"


def test_engine_never_imports_ai() -> None:
    engine_root = Path(nubia_engine.__file__).parent
    assert all("nubia_ai" not in path.read_text() for path in engine_root.glob("*.py"))


def test_ai_reuses_engine_models_and_rules() -> None:
    ai_root = Path(nubia_ai.__file__).parent
    source = "\n".join(path.read_text() for path in ai_root.glob("*.py"))
    assert "from nubia_engine import" in source
    assert "class GameState" not in source
    assert "class Action:" not in source
    assert "def legal_actions(" not in source
    assert "def apply_action(" not in source
