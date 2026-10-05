# NUBIA Engine

This repository will contain the rules engine for **NUBIA: AFRICAN CHESS**.
The authoritative game specification is [`docs/NUBIA_RULES.md`](docs/NUBIA_RULES.md),
with confirmed engine interpretations in
[`docs/ENGINE_DECISIONS.md`](docs/ENGINE_DECISIONS.md).

## Milestone 6

The engine provides immutable rules state plus deterministic ordinary movement,
capture, High Chief switching, Peasant movement, Imperion palace movement and
validated immutable transitions. It also implements Imperion GBESELE with
non-recursive ordinary-capture defence and mandatory simultaneous targets, along
with each Mystic's independent one-use Brainwash and re-brainwashing power.
Public generation includes ordinary and special actions in stable canonical order.
The post-action adjudication layer adds Peasant mine victory, no-Peasant officer
scoring, identity-independent threefold repetition, the 40-ply no-progress draw,
immutable global results, and terminal-state enforcement.
Milestone 5 adds deterministic engine display notation, a portable fixed-view
board renderer, immutable action replay, perft verification, and a two-human CLI.
Milestone 6 adds a separate `nubia_ai` package containing seeded random and
deterministic one-ply heuristic baselines, explainable evaluation, immutable
agent matches, reproducible matchup summaries, and a noninteractive benchmark.
Dependency direction is strictly `nubia_ai` to `nubia_engine`; the rules engine
does not know about agents.

Engine display notation is intentionally human-readable output, not a new
official notation standard and not a parseable serialization format.

Play locally with Empire A or B moving first:

```sh
python -m nubia_engine --first-player A
nubia --first-player B
```

Run a small reproducible baseline experiment with:

```sh
python -m nubia_ai --agent-a heuristic --agent-b random --games 10 --seed 42 --swap-sides
nubia-ai --agent-a random --agent-b random --games 2 --seed 42
```

These agents are transparent baselines, not claims of strategic strength. The
heuristic agent evaluates only the immediate successor and does not model a
reply; the random agent samples uniformly from legal actions.

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

Notation parsing, serialization/save files, minimax, alpha-beta pruning, iterative
deepening, transposition tables, time controls, MCTS, self-play or reinforcement
learning, neural networks, PyTorch/GPU training, human-versus-AI or graphical UI,
APIs, website integration, persistence, multiplayer, and deployment are deferred.
