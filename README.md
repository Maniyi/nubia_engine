# NUBIA Engine

This repository will contain the rules engine for **NUBIA: AFRICAN CHESS**.
The authoritative game specification is [`docs/NUBIA_RULES.md`](docs/NUBIA_RULES.md),
with confirmed engine interpretations in
[`docs/ENGINE_DECISIONS.md`](docs/ENGINE_DECISIONS.md).

## Milestone 1

The current implementation is an immutable domain foundation: the 10 x 10 board,
owner-relative coordinates, LAND/SEA terrain, resource mines, pieces, and the
complete deterministic starting position. It does not yet implement gameplay.

## Development setup

Python 3.11 or later is required.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the development checks with:

```sh
python -m pytest
python -m pytest --cov=nubia_engine --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
```

## Deferred work

Movement and capture generation, legal-action validation, High Chief switching,
Peasant directional movement, GBESELE, Brainwash actions and re-brainwashing,
victory and scoring, repetition and no-progress draws, move notation, CLI
gameplay, AI/search/training, APIs, website integration, persistence, and
multiplayer are intentionally deferred.

