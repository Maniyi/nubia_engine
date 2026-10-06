# NUBIA Engine

This repository will contain the rules engine for **NUBIA: AFRICAN CHESS**.
The authoritative game specification is [`docs/NUBIA_RULES.md`](docs/NUBIA_RULES.md),
with confirmed engine interpretations in
[`docs/ENGINE_DECISIONS.md`](docs/ENGINE_DECISIONS.md).

## Milestone 9

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
Milestone 7 adds deterministic depth-limited minimax, optional alpha-beta
pruning, immutable search diagnostics, and principal variations. Plain minimax
remains available as a correctness reference, and both modes preserve the
engine's canonical action order for deterministic ties.
Milestone 8 adds iterative deepening through a required maximum depth, optional
cumulative node and wall-clock budgets, last-completed-iteration decisions, and
per-iteration plus cumulative telemetry. An interrupted partial iteration is
never presented as complete. If depth one cannot finish, the iterative agent
returns the first action in canonical engine order as an explicitly unevaluated
fallback.
Dependency direction is strictly `nubia_ai` to `nubia_engine`; the rules engine
does not know about agents.
Milestone 9 adds an optional Pygame playground in the separate
`nubia_playground` package. It depends on the public AI and engine APIs; neither
existing package imports the playground or Pygame.

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
python -m nubia_ai --agent-a minimax --depth-a 2 --agent-b heuristic --games 2 --seed 42 --swap-sides
python -m nubia_ai --agent-a iterative --max-depth-a 3 --node-limit-a 5000 --agent-b heuristic --games 2 --seed 42 --swap-sides
python -m nubia_ai --agent-a iterative --max-depth-a 4 --time-ms-a 100 --agent-b random --games 2 --seed 42
```

These agents are transparent baselines, not claims of strategic strength. The
heuristic agent evaluates only the immediate successor and does not model a
reply; the random agent samples uniformly from legal actions. Minimax uses
alpha-beta by default; pass `--no-alpha-beta` for the complete-tree reference
mode. `--depth-a` and `--depth-b` are ignored for non-minimax agents.
Iterative agents use `--max-depth-a`/`--max-depth-b` and optionally
`--time-ms-a`/`--time-ms-b` and `--node-limit-a`/`--node-limit-b`. Milliseconds
are converted to seconds only at the CLI boundary. Node-limited and unlimited
searches are deterministic; wall-clock-limited completed depth can vary with
machine speed and system load. If both budgets expire at the same cooperative
check, the node limit is reported first.

## Local AI playground

The playground is a development and inspection UI, not a production interface.
Install its optional dependency alongside the development tools:

```sh
python -m pip install -e ".[dev,ui]"
```

Launch it through either entry point:

```sh
python -m nubia_playground
nubia-playground
```

The setup screen supports Human versus Human, Human versus Agent, and Agent
versus Agent. Either empire can move first; in Human-versus-Agent mode the human
can control either side. Agent choices are Random (with a seed), Heuristic,
fixed-depth Minimax, and Iterative Minimax with a maximum depth plus either a
node or millisecond budget. Defaults are intentionally conservative.

During play, click one of the current empire's pieces and then a highlighted
destination. Clicking the selected square or pressing Escape clears the
selection. When multiple actions share a destination, choose the exact action
in the side panel. GBESELE is confirmed there with its complete engine-provided
target set; Brainwash and re-brainwashing choices identify their target and
resulting allegiance. The board uses the documented fixed orientation with
Empire B at the top and Empire A at the bottom.

Pieces use the supplied NUBIA icons by default. Use the visible **Display**
control during a game to switch freely between Icons and the original Letters;
this changes presentation only, never game state. Both empires' Peasants share
one Peasant icon. If an icon cannot be loaded, that piece falls back to its
letter and the side panel reports the problem. The artwork has no rules meaning.

Agent-versus-Agent games begin paused. Use Resume/Pause for automatic play and
Step to apply exactly one action while paused. Restart keeps the same setup;
Setup returns to configuration; Quit or the window close control exits. Status,
controller assignments, search configuration, move history, latest action,
quiet-ply count, and terminal results remain visible during a game.

Current limitations are deliberate: there are no saved games, undo/redo,
network play, animations, audio, additional art assets, or production-grade
accessibility controls. Search telemetry is not displayed because the shared
Agent protocol returns only an action. The playground is optional and is not
needed to import or use `nubia_engine` or `nubia_ai`.

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
python -m pytest --cov=nubia_engine --cov=nubia_ai --cov=nubia_playground --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
```

## Deferred work

Notation parsing, serialization/save files, transposition tables,
repetition-aware cache keys, move ordering, Zobrist hashing, quiescence search,
aspiration windows, parallel search, MCTS, self-play or reinforcement learning,
neural networks, PyTorch/GPU training, APIs, website integration, persistence,
multiplayer, production UI work, and deployment are deferred.
