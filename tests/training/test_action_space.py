from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    ActionTarget,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    create_initial_state,
    legal_actions,
    legal_actions_from,
)
from nubia_training import (
    ACTION_KIND_BLOCK_SIZE,
    ACTION_KIND_OFFSETS,
    ACTION_KIND_ORDER,
    ACTION_SPACE_SIZE,
    ActionIndexCollisionError,
    IllegalOrStaleActionError,
    InvalidActionIndexError,
    UnresolvedActionIndexError,
    action_to_index,
    index_to_action,
    legal_action_indices,
    legal_action_mask,
)
from nubia_training import action_space as action_space_module


def test_explicit_kind_order_offsets_and_total_size() -> None:
    assert ACTION_KIND_ORDER == (
        ActionKind.MOVE,
        ActionKind.CAPTURE,
        ActionKind.SWITCH,
        ActionKind.GBESELE,
        ActionKind.BRAINWASH,
        ActionKind.REBRAINWASH,
    )
    assert ACTION_KIND_BLOCK_SIZE == 10_000
    assert (
        tuple(
            (kind, ordinal * 10_000) for ordinal, kind in enumerate(ACTION_KIND_ORDER)
        )
        == ACTION_KIND_OFFSETS
    )
    assert ACTION_SPACE_SIZE == 60_000


@pytest.mark.parametrize("index", [True, False, -1, 60_000, 1.5, "1"])
def test_invalid_and_boolean_indices_are_rejected(index: object) -> None:
    with pytest.raises(InvalidActionIndexError):
        index_to_action(create_initial_state(Empire.A), index)  # type: ignore[arg-type]


def test_every_initial_action_round_trips_and_mask_is_stable() -> None:
    state = create_initial_state(Empire.A)
    actions = legal_actions(state)
    indices = legal_action_indices(state)
    assert len(indices) == len(set(indices)) == len(actions)
    assert tuple(action_to_index(state, action) for action in actions) == indices
    assert tuple(index_to_action(state, index) for index in indices) == actions
    first = legal_action_mask(state)
    second = legal_action_mask(state)
    assert first.dtype == np.bool_
    assert first.shape == (ACTION_SPACE_SIZE,)
    assert int(first.sum()) == len(actions)
    assert first.tobytes() == second.tobytes()
    assert not first.flags.writeable


def test_ordinary_capture_and_switch_round_trip() -> None:
    cases = (
        state_with(
            [
                (Square(4, 4), piece("q", PieceType.QUEEN)),
                (Square(4, 6), piece("target", PieceType.PEASANT, Empire.B)),
                (Square(7, 0), piece("p", PieceType.PEASANT)),
            ]
        ),
        state_with([(Square(4, 4), piece("c", PieceType.EAST_AFRICAN_HIGH_CHIEF))]),
    )
    for state, kind in zip(cases, (ActionKind.CAPTURE, ActionKind.SWITCH), strict=True):
        action = next(action for action in legal_actions(state) if action.kind is kind)
        assert index_to_action(state, action_to_index(state, action)) == action


def test_all_special_action_kinds_round_trip_with_exact_targets() -> None:
    gbesele = state_with(
        [
            (Square(9, 4), piece("i", PieceType.IMPERION)),
            (Square(7, 4), piece("one", PieceType.PEASANT, Empire.B)),
            (Square(9, 1), piece("two", PieceType.PEASANT, Empire.B)),
        ]
    )
    brainwash = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
        ]
    )
    rebrainwash = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (
                Square(2, 4),
                piece(
                    "q",
                    PieceType.QUEEN,
                    Empire.B,
                    original_empire=Empire.A,
                ),
            ),
        ]
    )
    for state, kind in (
        (gbesele, ActionKind.GBESELE),
        (brainwash, ActionKind.BRAINWASH),
        (rebrainwash, ActionKind.REBRAINWASH),
    ):
        action = next(item for item in legal_actions(state) if item.kind is kind)
        resolved = index_to_action(state, action_to_index(state, action))
        assert resolved == action
        assert resolved.targets == action.targets


def test_perspective_b_rotates_source_and_destination() -> None:
    state = state_with(
        [(Square(2, 3), piece("q", PieceType.QUEEN, Empire.B))], Empire.B
    )
    action = next(
        action
        for action in legal_actions_from(state, Square(2, 3))
        if action.destination == Square(2, 4)
    )
    index = action_to_index(state, action, perspective=Empire.B)
    expected = 0 * 10_000 + 76 * 100 + 75
    assert index == expected
    assert index_to_action(state, index, perspective=Empire.B) == action


def test_dynamic_ids_do_not_affect_structural_index() -> None:
    first = state_with(
        [
            (Square(4, 4), piece("q1", PieceType.QUEEN)),
            (Square(4, 6), piece("target1", PieceType.PEASANT, Empire.B)),
            (Square(7, 0), piece("p1", PieceType.PEASANT)),
        ]
    )
    second = state_with(
        [
            (Square(4, 4), piece("q2", PieceType.QUEEN)),
            (Square(4, 6), piece("target2", PieceType.PEASANT, Empire.B)),
            (Square(7, 0), piece("p2", PieceType.PEASANT)),
        ]
    )
    first_capture = next(
        action for action in legal_actions(first) if action.kind is ActionKind.CAPTURE
    )
    second_capture = next(
        action for action in legal_actions(second) if action.kind is ActionKind.CAPTURE
    )
    assert first_capture.captured_piece_id != second_capture.captured_piece_id
    assert action_to_index(first, first_capture) == action_to_index(
        second, second_capture
    )


def test_target_ids_do_not_affect_special_structural_index() -> None:
    states = tuple(
        state_with(
            [
                (Square(4, 4), piece(f"m-{suffix}", PieceType.WEST_AFRICAN_MYSTIC)),
                (
                    Square(2, 4),
                    piece(f"q-{suffix}", PieceType.QUEEN, Empire.B),
                ),
            ]
        )
        for suffix in ("a", "b")
    )
    actions = tuple(
        next(
            action
            for action in legal_actions(state)
            if action.kind is ActionKind.BRAINWASH
        )
        for state in states
    )
    assert actions[0].target_piece_id != actions[1].target_piece_id
    assert action_to_index(states[0], actions[0]) == action_to_index(
        states[1], actions[1]
    )


def test_forged_stale_and_unresolved_actions_fail_cleanly() -> None:
    state = create_initial_state(Empire.A)
    legal = legal_actions(state)[0]
    forged = replace(legal, piece_id="stale")
    with pytest.raises(IllegalOrStaleActionError):
        action_to_index(state, forged)
    with pytest.raises(UnresolvedActionIndexError):
        index_to_action(state, 0)


def test_terminal_mask_and_indices_are_empty() -> None:
    live = create_initial_state(Empire.A)
    terminal = replace(
        live,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
    )
    assert not legal_action_mask(terminal).any()
    assert legal_action_indices(terminal) == ()


def test_collision_detection_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    first = Action("one", Square(4, 4), Square(4, 5), ActionKind.MOVE)
    second = Action("two", Square(4, 4), Square(4, 5), ActionKind.MOVE)
    monkeypatch.setattr(
        action_space_module, "legal_actions", lambda _state: (first, second)
    )
    with pytest.raises(ActionIndexCollisionError):
        legal_action_indices(create_initial_state(Empire.A))


def test_forged_special_target_identity_is_rejected() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
        ]
    )
    action = next(
        item for item in legal_actions(state) if item.kind is ActionKind.BRAINWASH
    )
    forged = replace(
        action,
        targets=(ActionTarget(action.destination, "stale"),),
    )
    with pytest.raises(IllegalOrStaleActionError):
        action_to_index(state, forged)


def test_public_action_functions_validate_state_action_and_perspective() -> None:
    state = create_initial_state(Empire.A)
    action = legal_actions(state)[0]
    with pytest.raises(TypeError):
        legal_action_indices("state")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        legal_action_indices(state, perspective="A")  # type: ignore[arg-type]
    with pytest.raises(IllegalOrStaleActionError):
        action_to_index(state, "action")  # type: ignore[arg-type]
    assert action_to_index(state, action, perspective=Empire.A) >= 0


def test_action_to_index_detects_a_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    state = create_initial_state(Empire.A)
    action = legal_actions(state)[0]
    colliding = replace(action, piece_id="another-current-id")
    monkeypatch.setattr(
        action_space_module,
        "legal_actions",
        lambda _state: (action, colliding),
    )
    with pytest.raises(ActionIndexCollisionError):
        action_to_index(state, action)
