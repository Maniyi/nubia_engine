from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    Empire,
    GameState,
    PieceType,
    Square,
    apply_action,
    format_action,
    legal_actions_from,
)


def _action(
    state: GameState,
    source: Square,
    kind: ActionKind,
    destination: Square | None = None,
) -> Action:
    return next(
        action
        for action in legal_actions_from(state, source)
        if action.kind is kind
        and (destination is None or action.destination == destination)
    )


def test_formats_move_capture_and_switch_without_ids() -> None:
    source = Square.from_notation("A:b4")
    move_state = state_with([(source, piece("secret-queen-id", PieceType.QUEEN))])
    move = _action(move_state, source, ActionKind.MOVE, Square.from_notation("A:b3"))
    assert format_action(move_state, move) == "Q A:b4-A:b3"

    target = Square.from_notation("A:b7")
    capture_state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (target, piece("hidden", PieceType.PEASANT, Empire.B)),
        ]
    )
    capture = _action(capture_state, source, ActionKind.CAPTURE, target)
    assert format_action(capture_state, capture) == "Q A:b4xA:b7"
    assert "hidden" not in format_action(capture_state, capture)

    chief_state = state_with(
        [(source, piece("chief", PieceType.EAST_AFRICAN_HIGH_CHIEF))]
    )
    switch = _action(
        chief_state, source, ActionKind.SWITCH, Square.from_notation("A:b5")
    )
    assert format_action(chief_state, switch) == "C A:b4-A:b5 (switch)"


def test_formats_all_special_actions_and_canonical_gbesele_targets() -> None:
    imperion_square = Square.from_notation("A:p5")
    gbesele_state = state_with(
        [
            (imperion_square, piece("i", PieceType.IMPERION)),
            (Square.from_notation("A:p2"), piece("one", PieceType.PEASANT, Empire.B)),
            (Square.from_notation("A:c5"), piece("two", PieceType.PEASANT, Empire.B)),
        ]
    )
    gbesele = _action(gbesele_state, imperion_square, ActionKind.GBESELE)
    assert format_action(gbesele_state, gbesele) == "I A:p5 ⚡ [A:c5, A:p2]"

    mystic_square = Square(4, 4)
    target = Square(2, 4)
    brainwash_state = state_with(
        [
            (mystic_square, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (target, piece("target", PieceType.QUEEN, Empire.B)),
        ]
    )
    brainwash = _action(brainwash_state, mystic_square, ActionKind.BRAINWASH)
    assert format_action(brainwash_state, brainwash).endswith("🌀 B:c6 (brainwash)")

    restore_state = replace(
        brainwash_state,
        board=tuple(
            replace(placed, original_empire=Empire.A)
            if placed is not None and placed.id == "target"
            else placed
            for placed in brainwash_state.board
        ),
    )
    restore = _action(restore_state, mystic_square, ActionKind.REBRAINWASH)
    assert format_action(restore_state, restore).endswith("🌀 B:c6 (restore)")


def test_converted_piece_uses_actual_type_and_historical_action_is_stable() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (
                source,
                piece("converted", PieceType.QUEEN, Empire.A, original_empire=Empire.B),
            )
        ]
    )
    action = legal_actions_from(state, source)[0]
    before = format_action(state, action)
    apply_action(state, action)
    assert before.startswith("Q ")
    assert format_action(state, action) == before


def test_formatter_rejects_mismatched_state_data() -> None:
    source = Square(4, 4)
    state = state_with([(source, piece("q", PieceType.QUEEN))])
    action = legal_actions_from(state, source)[0]
    with pytest.raises(ValueError, match="actor"):
        format_action(state_with([]), action)
    with pytest.raises(TypeError):
        format_action("state", action)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        format_action(state, "action")  # type: ignore[arg-type]

    capture_state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (Square(4, 6), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    capture = _action(capture_state, source, ActionKind.CAPTURE)
    with pytest.raises(ValueError, match="captured"):
        format_action(state, capture)

    mystic_state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    brainwash = _action(mystic_state, source, ActionKind.BRAINWASH)
    with pytest.raises(ValueError, match="target"):
        format_action(
            state_with([(source, piece("m", PieceType.WEST_AFRICAN_MYSTIC))]),
            brainwash,
        )
