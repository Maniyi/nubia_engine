import pytest
from conftest import piece, state_with

from nubia_engine import (
    ActionKind,
    Empire,
    PieceType,
    ReplayError,
    ResultReason,
    Square,
    legal_actions_from,
    replay_actions,
)


def test_empty_and_one_action_replays_are_immutable_and_equal() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    empty = replay_actions(state, [])
    assert empty.initial_state is empty.final_state is state
    assert empty.actions == () and empty.states == (state,)
    assert empty.applied_actions == empty.actions
    assert empty.state_sequence == empty.states
    action = legal_actions_from(state, source)[0]
    first = replay_actions(state, [action])
    assert first == replay_actions(state, [action])
    assert first.states == (state, first.final_state)
    assert state.piece_at(source) is not None


def test_replay_supports_capture_and_special_action() -> None:
    source, target = Square(4, 4), Square(4, 6)
    capture_state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (target, piece("enemy", PieceType.PEASANT, Empire.B)),
            (Square(2, 0), piece("p", PieceType.PEASANT, Empire.B)),
        ]
    )
    capture = next(
        action
        for action in legal_actions_from(capture_state, source)
        if action.kind is ActionKind.CAPTURE
    )
    captured_square = replay_actions(capture_state, [capture]).final_state.piece_at(
        target
    )
    assert captured_square is not None and captured_square.id == "q"

    mystic_state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    brainwash = next(
        action
        for action in legal_actions_from(mystic_state, source)
        if action.kind is ActionKind.BRAINWASH
    )
    converted = replay_actions(mystic_state, [brainwash]).final_state.piece_at(
        Square(2, 4)
    )
    assert converted is not None and converted.current_empire is Empire.A

    imperion = Square.from_notation("A:p5")
    gbesele_state = state_with(
        [
            (imperion, piece("i", PieceType.IMPERION)),
            (
                Square.from_notation("A:c5"),
                piece("enemy", PieceType.PEASANT, Empire.B),
            ),
            (Square(2, 0), piece("survivor", PieceType.PEASANT, Empire.B)),
        ]
    )
    gbesele = next(
        action
        for action in legal_actions_from(gbesele_state, imperion)
        if action.kind is ActionKind.GBESELE
    )
    assert (
        replay_actions(gbesele_state, [gbesele]).final_state.piece_at(
            Square.from_notation("A:c5")
        )
        is None
    )

    restore_state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (
                Square(2, 4),
                piece(
                    "converted",
                    PieceType.QUEEN,
                    Empire.B,
                    original_empire=Empire.A,
                ),
            ),
        ]
    )
    restore = next(
        action
        for action in legal_actions_from(restore_state, source)
        if action.kind is ActionKind.REBRAINWASH
    )
    restored = replay_actions(restore_state, [restore]).final_state.piece_at(
        Square(2, 4)
    )
    assert restored is not None and restored.current_empire is Empire.A


def test_replay_wraps_illegal_stale_and_post_terminal_actions_with_index() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    action = legal_actions_from(state, source)[0]
    with pytest.raises(ReplayError, match="index 1") as caught:
        replay_actions(state, [action, action])
    assert caught.value.action_index == 1
    assert caught.value.__cause__ is caught.value.cause

    mine_source = Square(4, Square.from_notation("B:o6").column)
    terminal_state = state_with(
        [
            (mine_source, piece("p", PieceType.PEASANT)),
            (Square(8, 0), piece("q", PieceType.QUEEN)),
        ]
    )
    win = next(
        action
        for action in legal_actions_from(terminal_state, mine_source)
        if action.destination == Square.from_notation("B:o6")
    )
    later = legal_actions_from(terminal_state, Square(8, 0))[0]
    with pytest.raises(ReplayError, match="index 1"):
        replay_actions(terminal_state, [win, later])


def test_replay_reaches_repetition_and_no_progress_draws() -> None:
    a_start, a_out = Square(8, 2), Square(8, 3)
    b_start, b_out = Square(1, 7), Square(1, 6)
    initial = state_with(
        [
            (a_start, piece("a-q", PieceType.QUEEN)),
            (b_start, piece("b-q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("a-p", PieceType.PEASANT)),
        ]
    )
    current = initial
    actions = []
    for _ in range(2):
        for source, destination in (
            (a_start, a_out),
            (b_start, b_out),
            (a_out, a_start),
            (b_out, b_start),
        ):
            action = next(
                item
                for item in legal_actions_from(current, source)
                if item.kind is ActionKind.MOVE and item.destination == destination
            )
            actions.append(action)
            current = replay_actions(current, [action]).final_state
    repetition = replay_actions(initial, actions).final_state.result
    assert repetition is not None
    assert repetition.reasons == (ResultReason.THREEFOLD_REPETITION,)

    quiet = state_with(
        [
            (Square(4, 4), piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ],
        quiet=39,
    )
    quiet_action = legal_actions_from(quiet, Square(4, 4))[0]
    no_progress = replay_actions(quiet, [quiet_action]).final_state.result
    assert no_progress is not None
    assert no_progress.reasons == (ResultReason.NO_PROGRESS_40_PLIES,)


def test_replay_validates_public_arguments() -> None:
    with pytest.raises(TypeError):
        replay_actions("state", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        replay_actions(state_with([]), 1)  # type: ignore[arg-type]
