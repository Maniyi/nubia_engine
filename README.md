# NUBIA Engine

This repository will contain the rules engine for **NUBIA: AFRICAN CHESS**.
The authoritative game specification is [`docs/NUBIA_RULES.md`](docs/NUBIA_RULES.md),
with confirmed engine interpretations in
[`docs/ENGINE_DECISIONS.md`](docs/ENGINE_DECISIONS.md).

## Milestone 2

The engine provides the immutable Milestone 1 domain foundation plus deterministic
ordinary movement, capture, High Chief switching, Peasant orientation and special
horizontal movement, Imperion palace movement, and validated immutable state
transitions. Public action generation is ordered by source square, destination
square, then action kind.

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

GBESELE, Brainwash actions and re-brainwashing, victory and scoring, repetition
and no-progress draws, terminal-state enforcement, move notation, CLI gameplay,
AI/search/training, APIs, website integration, persistence, and multiplayer are
intentionally deferred.
