from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("package", ["nubia_engine", "nubia_ai"])
def test_existing_packages_do_not_depend_on_playground(package: str) -> None:
    root = Path("src") / package
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        )
        assert all(not name.startswith("nubia_playground") for name in imported)
