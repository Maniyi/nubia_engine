from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import nubia_training


def _imports(root: Path) -> set[str]:
    imported: set[str] = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        imported.update(
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        )
    return imported


def test_only_generation_specific_training_code_imports_ai() -> None:
    root = Path(nubia_training.__file__).parent
    imported = _imports(root)
    assert any(name.startswith("nubia_engine") for name in imported)
    assert not any(name.startswith("nubia_playground") for name in imported)
    ai_files: set[str] = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        direct = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("nubia_ai")
        }
        if direct:
            ai_files.add(path.name)
    assert ai_files == {"generation.py"}


def test_existing_packages_do_not_depend_on_training() -> None:
    for package in ("nubia_engine", "nubia_ai", "nubia_playground"):
        assert all(
            not name.startswith("nubia_training")
            for name in _imports(Path("src") / package)
        )


def test_existing_imports_do_not_load_numpy_or_training() -> None:
    code = (
        "import sys, nubia_engine, nubia_ai, nubia_playground; "
        "assert 'numpy' not in sys.modules; "
        "assert 'nubia_training' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_training_import_does_not_initialize_ai_agents() -> None:
    code = "import sys, nubia_training; assert 'nubia_ai' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
