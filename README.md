# NUBIA Engine

This repository will contain the rules engine for **NUBIA: AFRICAN CHESS**.
The authoritative game specification is [`docs/NUBIA_RULES.md`](docs/NUBIA_RULES.md),
with confirmed engine interpretations in
[`docs/ENGINE_DECISIONS.md`](docs/ENGINE_DECISIONS.md).

## Milestone 3

The engine provides immutable rules state plus deterministic ordinary movement,
capture, High Chief switching, Peasant movement, Imperion palace movement and
validated immutable transitions. It also implements Imperion GBESELE with
non-recursive ordinary-capture defence and mandatory simultaneous targets, along
with each Mystic's independent one-use Brainwash and re-brainwashing power.
Public generation includes ordinary and special actions in stable canonical order.

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

Victory (including mine victory immediately caused by Brainwash), scoring,
repetition and no-progress draws, terminal-state enforcement, result models, move
notation, CLI gameplay, perft, AI/search/training, APIs, website integration,
persistence, and multiplayer are intentionally deferred to later milestones.
