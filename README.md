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

## Machine-learning representation foundation

Milestone 10 adds the optional `nubia_training` package. It converts immutable
engine states and legal actions into deterministic, versioned NumPy structures;
it does not include a model, training loop, dataset writer, self-play system, or
learned agent. Its dependency direction is strictly `nubia_training` to
`nubia_engine`. The engine, classical AI, and playground do not import the
training package or NumPy.

Install the training dependency with the development tools:

```sh
python -m pip install -e ".[dev,training]"
```

`STATE_ENCODING_VERSION`, `ACTION_SPACE_VERSION`, and
`TRAINING_EXAMPLE_VERSION` are independent compatibility contracts, all at
version 1. `STATE_ENCODING_SPEC` and `ACTION_SPACE_SPEC` expose immutable
metadata rather than deriving compatibility from the package version.

### State observation

`encode_state(state, perspective=...)` defaults to the side to move. Empire A
uses the engine's canonical fixed board; Empire B is rotated 180 degrees, using
`(row, column) -> (9 - row, 9 - column)`. Current and original allegiance are
then labelled relative to the requested perspective as self or opponent.

The `float32` spatial tensor has shape `(21, 10, 10)` and this exact channel
order:

```text
self_imperion
self_queen
self_north_central_war_chief
self_east_african_high_chief
self_south_african_advisor
self_west_african_mystic
self_peasant
opponent_imperion
opponent_queen
opponent_north_central_war_chief
opponent_east_african_high_chief
opponent_south_african_advisor
opponent_west_african_mystic
opponent_peasant
original_self
original_opponent
mystic_power_available
land
sea
self_resource
opponent_resource
```

The first 14 planes encode one-hot piece type and current allegiance. The next
two mark every piece's original allegiance, the Mystic plane marks surviving
Mystics whose Brainwash power is available, and the last four describe terrain
and mine ownership. Piece identifiers, notation, artwork, and display data are
excluded.

The read-only `float32` global vector has this exact order and formulas:

```text
side_to_move_is_self                 1 if side_to_move == perspective, else 0
quiet_ply_progress                   min(quiet_ply_count, 40) / 40
current_repetition_count_normalized  min(current-position occurrences, 3) / 3
is_terminal                          1 if a result is present, else 0
terminal_outcome_for_self            +1 win, -1 loss, 0 draw or non-terminal
```

The terminal flag disambiguates a draw from a non-terminal state. Encoded arrays
are copied into immutable storage. A SHA-256 representation fingerprint covers
the versions, perspective, shapes, dtypes, and bytes; it is an integrity aid,
not an engine position key.

### Actions, masks, and repetition context

The version-1 action-kind order is `MOVE`, `CAPTURE`, `SWITCH`, `GBESELE`,
`BRAINWASH`, `REBRAINWASH`. Each kind owns a block of 10,000 entries and:

```text
index = kind_offset + normalized_source_square * 100 + normalized_destination_square
```

The offsets are `0`, `10000`, `20000`, `30000`, `40000`, and `50000`, so the
fixed action space contains 60,000 entries. Squares use the same perspective
rotation as the tensor. Piece IDs, captured IDs, and special-action target IDs
are not indexed. `index_to_action` searches the current engine-generated legal
actions and returns the exact action with its current captured ID or complete
GBESELE/Brainwash target tuple. Missing, stale, forged, ambiguous, and colliding
actions fail explicitly.

The dense legal mask is a read-only 60,000-element Boolean array. Legal indices
are also retained as read-only `uint16` values in engine action order. Terminal
states have an empty mask. The action-aligned repetition vector is read-only
`uint8`: a legal slot contains the resulting position's prior occurrence count,
clipped to 0, 1, or 2, and illegal slots contain zero. A value of 2 therefore
marks an action that would ordinarily create a third occurrence. The engine's
post-action adjudication still controls higher-priority mine victory and officer
scoring.

This immediate feature cannot encode the unbounded repetition map. Deeper tree
search must carry the complete authoritative `GameState.position_history`; the
fixed observation is not a substitute for engine state.

### Training examples and memory

`TrainingExample` stores an immutable `EncodedState`, a sparse `uint16` list of
unique legal policy indices, matching read-only `float32` probabilities, and a
finite value in `[-1, 1]`. Sparse policy mass must be non-negative, finite, and
sum to 1 within `1e-6`. Multi-action distributions are supported. Values are
always from the stored perspective: `+1` win, `0` draw, and `-1` loss. A dense
read-only `float32` policy is materialized only by `dense_policy()`.

The fixed per-observation storage is:

```text
spatial tensor       (21, 10, 10) float32    8,400 bytes
global vector        (5,) float32               20 bytes
legal mask           (60,000,) bool          60,000 bytes
repetition vector    (60,000,) uint8         60,000 bytes
fixed total                                  128,420 bytes
legal indices        (N,) uint16                 2N bytes
dense policy         (60,000,) float32       240,000 bytes (only on request)
```

Excluding Python object overhead, future model activations, sparse legal-index
storage, and optional dense policies, batches of 32, 64, and 128 observations
use 4,109,440 bytes (3.92 MiB), 8,218,880 bytes (7.84 MiB), and 16,437,760 bytes
(15.68 MiB), respectively. The standard initial state has 40 legal actions, so
its complete encoded arrays occupy 128,500 bytes including its 80-byte legal
index array.

## Persistent game records and training datasets

Milestone 11 adds a deterministic local data pipeline; no external raw-game
dataset is required. Generated files belong under `artifacts/nubia_training/`
(which is ignored precisely by Git) or another explicitly supplied directory.
Nothing is generated during installation or import.

Each game is stored as canonical UTF-8 JSON at `games/<game-id>.json`. The
record contains independent schema versions, the standard setup and first
player, public agent configurations and deterministic random seeds, zero-based
ply records, genuine engine terminal data or a stable abort reason, and encoded
state fingerprints. An action's fixed Milestone 10 index is authoritative;
notation is diagnostic. Canonical JSON sorts keys, uses compact separators and
one trailing LF. SHA-256 covers canonical content without the fingerprint or
ID; the game ID is the first 24 hexadecimal characters of that digest. Writes
are atomic and identical rewrites are no-ops, while conflicts fail.

Generation supports Random, Heuristic, Minimax, and Iterative Minimax agents.
Random agents receive explicit per-game seeds. Deterministic agents retain
engine-order tie breaking. Iterative generation accepts node budgets only:
wall-clock limits are deliberately rejected because their completed depth can
vary by machine and load. Series generation can alternate the first player and
swap participant sides. Safety limits and agent failures create aborted games,
never fabricated draws.

Replay reconstructs the standard initial state and resolves every stored action
index against current engine-generated legal actions. It checks actor, action
kind, notation, pre/post fingerprints, terminal result, reasons, officer totals,
ply count, and final fingerprint. Completed games convert to one example per
pre-action state. The acting empire is the perspective, the selected action is
a one-hot behavioral-cloning policy, and the value is +1/0/-1 for eventual
win/draw/loss. These choices are observations of the generating agents, not
optimal or expert policies. Aborted games contribute no examples.

Datasets use deterministic compressed NPZ shards and a canonical
`manifest.json`. Shards store `float32` spatial `[N,21,10,10]` and global
`[N,5]` arrays; ragged legal and policy `uint16` indices with `int64` offsets;
legal-aligned `uint8` repetition counts; `float32` policy probabilities and
values; Unicode perspectives, source game IDs, agent identities, and SHA-256
fingerprints; plus all representation/schema versions. They contain no object
arrays, dense 60,000-entry masks, dense repetition vectors, or Pickle. The
manifest records ordered source games and shards, SHA-256 file checksums,
counts, outcomes, agents, ply statistics, build configuration, and skipped
abort reasons. Validation verifies the manifest digest, versions, source games
when supplied, every checksum, shard structure, counts, and reconstructed
example fingerprints.

A complete small local workflow is:

```sh
python -m nubia_training generate-games \
  --output-dir artifacts/nubia_training --games 2 --base-seed 42 \
  --first-player alternate --swap-sides --agent-a random --agent-b heuristic
python -m nubia_training validate-games \
  --games-dir artifacts/nubia_training/games
python -m nubia_training build-dataset \
  --games-dir artifacts/nubia_training/games \
  --output-dir artifacts/nubia_training/dataset --shard-size 256
python -m nubia_training validate-dataset \
  --dataset-dir artifacts/nubia_training/dataset \
  --games-dir artifacts/nubia_training/games
python -m nubia_training inspect-dataset \
  --dataset-dir artifacts/nubia_training/dataset
```

The console command `nubia-training` provides the same subcommands. A small
weak-agent validation corpus proves integrity and reproducibility; it is not a
large, diverse, strategically strong, or production-ready training dataset.

## Small policy-value neural network

Milestone 12 adds an optional PyTorch subpackage without changing the NumPy-only
training tools. Install the neural and development extras with:

```sh
python -m pip install -e ".[dev,training,neural]"
```

Importing `nubia_training` does not import PyTorch; PyTorch is loaded only by
`nubia_training.neural`. The version-1 model consumes `float32` spatial tensors
of shape `[batch,21,10,10]` and global features of shape `[batch,5]`. Global
features are projected and broadcast into the convolutional trunk. The default
trunk has 64 channels, four GroupNorm residual blocks, 16-dimensional policy
embeddings, and a 128-unit value hidden layer. The value output is one `tanh`
scalar in `[-1,1]` per example, from the stored perspective: `+1` win, `0` draw,
and `-1` loss.

The policy head does not flatten the board into a massive dense output layer.
For every action kind and square, separate learned source and destination
embeddings are produced by 1x1 convolutions. Their scaled dot products form
`[batch,6,100,100]`, then flatten exactly as:

```text
kind_offset + normalized_source * 100 + normalized_destination
```

The main forward method returns raw `[batch,60000]` logits. Legal masking is a
separate stable operation: illegal logits become negative infinity before the
legal softmax, receive exactly zero probability, and cannot affect the
normalizer. Sparse cross-entropy gathers only stored target indices while its
normalizer covers only legal actions; no dense 60,000-entry target is created.
The value objective is mean squared error, with explicit policy and value
weights.

`ShardDataset` validates the manifest and all shards before training, preserves
manifest order, loads with `allow_pickle=False`, and retains only a bounded
shard cache. It materializes a dense legal mask only for the current batch.
Sparse policy targets, legal-aligned repetition counts, perspective, game ID,
ply index, and agent identity remain available in the batch API.

Inspect the default architecture or run a deliberately tiny CPU smoke job:

```sh
python -m nubia_training.neural inspect-model
python -m nubia_training.neural train \
  artifacts/nubia_training/dataset /tmp/nubia-neural-smoke \
  --device cpu --epochs 1 --max-steps 2 --batch-size 2 \
  --trunk-channels 8 --residual-blocks 1 \
  --policy-embedding-dim 2 --value-hidden-dim 8 \
  --normalization-groups 2
python -m nubia_training.neural inspect-checkpoint \
  /tmp/nubia-neural-smoke/checkpoint.pt
```

The equivalent console command is `nubia-neural`. Devices are `cpu`, `mps`,
`cuda`, and `auto`; `auto` prefers CUDA, then MPS, then CPU. CPU is the required
Mac development path. A later Windows machine may explicitly select CUDA, but
this milestone contains no CUDA-specific kernels, mixed precision, or GPU
requirement. The CLI defaults to at most ten optimizer steps, so an explicit
configuration is required for anything beyond a bounded local check.

Checkpoints store complete model and training configurations, all representation
and schema versions, CPU model tensors, optional AdamW state, dataset identity
and fingerprint, progress counters, seed, and latest losses. A canonical JSON
sidecar records the metadata and SHA-256 of the checkpoint. Loading verifies the
checksum before using PyTorch's safe `weights_only` mode and rejects version,
architecture, representation, or requested-dataset mismatches. Existing
checkpoint paths are never overwritten.

This is an untrained model architecture. A decreasing smoke-test loss only
proves that forward, backward, optimizer, checkpoint, and resume paths work; it
is not evidence of game-playing strength. Parameter and batch-memory inspection
estimates exclude framework/runtime overhead and some intermediate activations.

## Neural inference and deterministic PUCT search

Milestone 13 adds checkpoint-backed position evaluation, a direct neural-policy
agent, and a deterministic PUCT MCTS agent. A position evaluator accepts only a
non-terminal immutable `GameState` and returns legal action indices and aligned
normalized priors in engine order, plus a value in `[-1,1]`. The perspective is
always the state's side to move. The production evaluator uses the existing
safe, version-checked checkpoint loader, explicit device selection,
`torch.inference_mode()`, and evaluation mode. CPU is the required validation
path; importing the base `nubia_training` package remains free of PyTorch.

Selection uses the following score for an edge from state `s` through action
`a`:

```text
Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

`Q` is the edge's mean value from the selecting parent player's perspective,
`P` is its prior, `N(s)` is the parent visit count, and `N(s,a)` is the edge
visit count. Unvisited edges have `Q = 0`. Evaluator and terminal values begin
from the leaf side-to-move perspective and are negated exactly once per backed-up
edge. All ties use original engine legal-action order.

Each search creates a fresh tree containing complete `GameState` values,
including repetition history. The root training target is the normalized raw
visit-count distribution over legal actions, not the neural prior. Temperature
zero chooses the highest visit count with engine-order ties. Positive
temperature samples proportional to `visits ** (1 / temperature)` with an
explicit local seed. Optional root-only Dirichlet noise is also locally seeded,
disabled by default, and never changes value predictions or deeper priors.

`NeuralPolicyAgent` chooses the greatest legal prior directly, while `MCTSAgent`
runs a new search per move and retains its latest immutable search result for
diagnostics. Both satisfy the unchanged structural `nubia_ai.Agent` protocol.
An initialized or smoke-trained checkpoint is still strategically weak; these
agents demonstrate correct inference and search, not playing strength.

Run a small CPU search or bounded comparison with:

```sh
python -m nubia_training.neural search /tmp/nubia-neural-smoke/checkpoint.pt \
  --device cpu --simulations 8 --max-depth 32 --first-player A
python -m nubia_training.neural benchmark /tmp/nubia-neural-smoke/checkpoint.pt \
  --device cpu --simulations 2 --games 1 --opponent random --max-plies 100
```

Safety-limit events remain errors rather than fabricated draws. Tree reuse,
batched or parallel leaf evaluation, persistent self-play policy records,
MCTS-guided training, and GPU optimization remain deferred.

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

Notation parsing, sustained model training, transposition tables,
repetition-aware cache keys, move ordering, Zobrist hashing, quiescence search,
aspiration windows, parallel search, self-play or reinforcement learning,
production PyTorch/GPU training, MCTS tree reuse, APIs, website integration,
persistence, multiplayer, production UI work, and deployment are deferred.
