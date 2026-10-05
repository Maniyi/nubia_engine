from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    BrainwashAvailability,
    Empire,
    GameState,
    PieceType,
    Square,
    legal_actions,
    legal_actions_from,
)


def _specials(state: GameState, source: Square) -> list[Action]:
    return [
        action
        for action in legal_actions_from(state, source)
        if action.kind in (ActionKind.BRAINWASH, ActionKind.REBRAINWASH)
    ]


def test_brainwash_row_column_and_diagonal_geometry_is_exactly_two() -> None:
    source = Square(4, 4)
    targets = (Square(2, 4), Square(4, 6), Square(6, 6))
    state = state_with(
        [
            (source, piece("mystic", PieceType.WEST_AFRICAN_MYSTIC)),
            *[
                (square, piece(f"target-{index}", PieceType.PEASANT, Empire.B))
                for index, square in enumerate(targets)
            ],
            (Square(3, 3), piece("near", PieceType.PEASANT, Empire.B)),
            (Square(7, 4), piece("far", PieceType.PEASANT, Empire.B)),
        ]
    )
    actions = _specials(state, source)
    assert {action.destination for action in actions} == set(targets)
    assert all(action.destination != source for action in actions)
    assert all(action.target_piece_id is not None for action in actions)


def test_midpoint_friendly_target_and_imperion_block_brainwash() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("mystic", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(3, 4), piece("midpoint", PieceType.PEASANT)),
            (Square(2, 4), piece("blocked", PieceType.PEASANT, Empire.B)),
            (Square(4, 6), piece("friendly", PieceType.PEASANT)),
            (Square(6, 6), piece("imperion", PieceType.IMPERION, Empire.B)),
        ]
    )
    assert _specials(state, source) == []


def test_spent_mystic_has_no_special_action() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (
                source,
                piece(
                    "mystic",
                    PieceType.WEST_AFRICAN_MYSTIC,
                    brainwash=BrainwashAvailability.SPENT,
                ),
            ),
            (Square(2, 4), piece("target", PieceType.PEASANT, Empire.B)),
        ]
    )
    assert _specials(state, source) == []


def test_brainwash_is_separate_from_ordinary_capture_and_movement() -> None:
    source = Square(4, 4)
    target = Square(2, 4)
    state = state_with(
        [
            (source, piece("mystic", PieceType.WEST_AFRICAN_MYSTIC)),
            (target, piece("target", PieceType.PEASANT, Empire.B)),
        ]
    )
    actions = legal_actions_from(state, source)
    at_target = [action for action in actions if action.destination == target]
    assert {action.kind for action in at_target} == {
        ActionKind.CAPTURE,
        ActionKind.BRAINWASH,
    }
    brainwash = next(
        action for action in at_target if action.kind is ActionKind.BRAINWASH
    )
    assert brainwash.captured_piece_id is None
    assert brainwash.source == source


def test_converted_target_generates_rebrainwash_only_for_original_side() -> None:
    source = Square(4, 4)
    converted = piece("target", PieceType.QUEEN, Empire.B, original_empire=Empire.A)
    state = state_with(
        [
            (source, piece("mystic", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), converted),
        ]
    )
    specials = _specials(state, source)
    assert len(specials) == 1
    assert specials[0].kind is ActionKind.REBRAINWASH


def test_specials_use_both_public_generators_deterministically() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("mystic", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("target", PieceType.PEASANT, Empire.B)),
        ]
    )
    first = legal_actions(state)
    assert first == legal_actions(state)
    assert set(_specials(state, source)) <= set(first)
    assert len(first) == len(set(first))
