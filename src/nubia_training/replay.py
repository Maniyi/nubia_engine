"""Authoritative engine replay and raw-game conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nubia_engine import (
    Empire,
    GameState,
    Outcome,
    apply_action,
    create_initial_state,
    format_action,
)
from nubia_training.action_space import index_to_action
from nubia_training.encoding import encode_state
from nubia_training.errors import ReplayValidationError
from nubia_training.examples import TrainingExample
from nubia_training.records import RawGameRecord


@dataclass(frozen=True, slots=True)
class ReplayReport:
    game_id: str
    states: tuple[GameState, ...]
    final_state: GameState
    total_plies: int


@dataclass(frozen=True, slots=True)
class ProvenancedExample:
    game_id: str
    ply_index: int
    acting_empire: Empire
    agent_identity: str
    example: TrainingExample


def _failure(
    record: RawGameRecord, ply: int | None, field: str, detail: str
) -> ReplayValidationError:
    location = "final" if ply is None else f"ply {ply}"
    return ReplayValidationError(
        f"game {record.game_id} {location}: {field} mismatch ({detail})"
    )


def replay_game(record: RawGameRecord) -> ReplayReport:
    """Replay every indexed action, checking each recorded invariant."""

    record.validate()
    state = create_initial_state(record.first_player)
    states = [state]
    for expected_index, ply in enumerate(record.plies):
        try:
            if ply.ply_index != expected_index:
                raise _failure(record, expected_index, "ply_index", str(ply.ply_index))
            if state.side_to_move is not ply.acting_empire:
                raise _failure(
                    record, expected_index, "acting_empire", ply.acting_empire.value
                )
            pre = encode_state(state, perspective=ply.acting_empire).fingerprint()
            if pre != ply.pre_state_fingerprint:
                raise _failure(record, expected_index, "pre_state_fingerprint", pre)
            action = index_to_action(
                state, ply.action_index, perspective=ply.acting_empire
            )
            if action.kind is not ply.action_kind:
                raise _failure(record, expected_index, "action_kind", action.kind.value)
            notation = format_action(state, action)
            if notation != ply.notation:
                raise _failure(record, expected_index, "notation", notation)
            state = apply_action(state, action)
            post = encode_state(state, perspective=ply.acting_empire).fingerprint()
            if post != ply.post_state_fingerprint:
                raise _failure(record, expected_index, "post_state_fingerprint", post)
            states.append(state)
        except ReplayValidationError:
            raise
        except Exception as error:
            raise _failure(
                record, expected_index, "action_index", str(error)
            ) from error
    if len(record.plies) != record.total_plies:
        raise _failure(record, None, "total_plies", str(len(record.plies)))
    if (record.status == "completed") != (state.result is not None):
        raise _failure(record, None, "completion_status", record.status)
    if state.result != record.final_result:
        raise _failure(record, None, "final_result", repr(state.result))
    final_fingerprint = encode_state(state).fingerprint()
    if final_fingerprint != record.final_state_fingerprint:
        raise _failure(record, None, "final_state_fingerprint", final_fingerprint)
    return ReplayReport(record.game_id, tuple(states), state, len(record.plies))


def game_to_examples(record: RawGameRecord) -> tuple[ProvenancedExample, ...]:
    """Convert a completed validated game to one-hot behavioral targets."""

    report = replay_game(record)
    if record.status == "aborted":
        return ()
    if record.final_result is None:
        raise ReplayValidationError(f"game {record.game_id} has no terminal result")
    output: list[ProvenancedExample] = []
    for ply, state in zip(record.plies, report.states[:-1], strict=True):
        encoded = encode_state(state, perspective=ply.acting_empire)
        result = record.final_result
        value = (
            0.0
            if result.outcome is Outcome.DRAW
            else 1.0
            if result.winner is ply.acting_empire
            else -1.0
        )
        example = TrainingExample(
            encoded,
            np.asarray((ply.action_index,), dtype=np.uint16),
            np.asarray((1.0,), dtype=np.float32),
            value,
        )
        spec = record.agent_a if ply.acting_empire is Empire.A else record.agent_b
        output.append(
            ProvenancedExample(
                record.game_id, ply.ply_index, ply.acting_empire, spec.name, example
            )
        )
    return tuple(output)
