from dataclasses import replace

from conftest import piece, state_with

from nubia_engine import (
    ActionKind,
    Empire,
    GameState,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    apply_action,
    create_initial_state,
    legal_actions_from,
    position_key,
)


def _move(state: GameState, source: Square, destination: Square) -> GameState:
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.destination == destination and action.kind is ActionKind.MOVE
    )
    return apply_action(state, action)


def test_initial_state_records_its_position_once_deterministically() -> None:
    for first in Empire:
        state = create_initial_state(first)
        assert state.position_history == (position_key(state),)
        assert state.position_history == create_initial_state(first).position_history


def test_legacy_state_seeds_current_position_before_first_transition() -> None:
    source, destination = Square(4, 4), Square(4, 5)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ]
    )
    result = _move(state, source, destination)
    assert result.position_history == (position_key(state), position_key(result))
    assert state.position_history == ()


def test_real_reversible_sequence_draws_on_third_occurrence() -> None:
    a_start, a_out = Square(8, 2), Square(8, 3)
    b_start, b_out = Square(1, 7), Square(1, 6)
    state = state_with(
        [
            (a_start, piece("a-q", PieceType.QUEEN)),
            (b_start, piece("b-q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("a-p", PieceType.PEASANT)),
        ]
    )
    for cycle in range(2):
        state = _move(state, a_start, a_out)
        state = _move(state, b_start, b_out)
        state = _move(state, a_out, a_start)
        state = _move(state, b_out, b_start)
        if cycle == 0:
            assert state.result is None
    assert state.result is not None
    assert state.result.reasons == (ResultReason.THREEFOLD_REPETITION,)
    assert state.position_history.count(position_key(state)) == 3


def test_quiet_move_and_switch_reach_no_progress_draw() -> None:
    cases = (
        (PieceType.QUEEN, ActionKind.MOVE),
        (PieceType.EAST_AFRICAN_HIGH_CHIEF, ActionKind.SWITCH),
    )
    for piece_type, kind in cases:
        source = Square(4, 4)
        state = state_with(
            [
                (source, piece("actor", piece_type)),
                (Square(7, 0), piece("p", PieceType.PEASANT)),
            ],
            quiet=39,
        )
        action = next(
            action
            for action in legal_actions_from(state, source)
            if action.kind is kind
        )
        result = apply_action(state, action)
        assert result.quiet_ply_count == 40
        assert result.result is not None
        assert result.result.reasons == (ResultReason.NO_PROGRESS_40_PLIES,)


def test_progress_actions_reset_counter_without_no_progress_draw() -> None:
    source, target = Square(4, 4), Square(4, 6)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (target, piece("target", PieceType.PEASANT, Empire.B)),
            (Square(2, 0), piece("survivor", PieceType.PEASANT, Empire.B)),
        ],
        quiet=39,
    )
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.CAPTURE
    )
    result = apply_action(state, action)
    assert result.quiet_ply_count == 0
    assert result.result is None


def test_simultaneous_repetition_and_no_progress_keeps_both_reasons() -> None:
    source, destination = Square(4, 4), Square(4, 5)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
        ],
        quiet=39,
    )
    board = list(state.board)
    board[source.row * 10 + source.column] = None
    board[destination.row * 10 + destination.column] = piece("q", PieceType.QUEEN)
    expected = replace(
        state,
        board=tuple(board),
        side_to_move=Empire.B,
        quiet_ply_count=40,
        ply_number=1,
    )
    key = position_key(expected)
    seeded = replace(state, position_history=(key, key))
    result = _move(seeded, source, destination)
    assert result.result is not None
    assert result.result.outcome is Outcome.DRAW
    assert result.result.reasons == (
        ResultReason.THREEFOLD_REPETITION,
        ResultReason.NO_PROGRESS_40_PLIES,
    )
