from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    ActionTarget,
    BrainwashAvailability,
    Empire,
    GameState,
    IllegalActionError,
    PieceType,
    Square,
    apply_action,
    legal_actions_from,
)


def _action(state: GameState, source: Square, kind: ActionKind) -> Action:
    return next(
        action for action in legal_actions_from(state, source) if action.kind is kind
    )


def test_gbesele_removes_all_targets_as_one_immutable_transition() -> None:
    source = Square(9, 4)
    targets = (Square(7, 4), Square(9, 1))
    imperion = piece("imperion", PieceType.IMPERION)
    survivor = piece("survivor", PieceType.PEASANT)
    state = state_with(
        [
            (source, imperion),
            (targets[0], piece("target-1", PieceType.PEASANT, Empire.B)),
            (targets[1], piece("target-2", PieceType.PEASANT, Empire.B)),
            (Square(6, 9), survivor),
        ],
        quiet=8,
        ply=12,
    )
    action = _action(state, source, ActionKind.GBESELE)
    result = apply_action(state, action)
    assert result.piece_at(source) is imperion
    assert all(result.piece_at(target) is None for target in targets)
    assert result.piece_at(Square(6, 9)) is survivor
    assert all(state.piece_at(target) is not None for target in targets)
    assert result.side_to_move is Empire.B
    assert result.ply_number == 13
    assert result.quiet_ply_count == 0


@pytest.mark.parametrize("mode", ["subset", "extra", "stale"])
def test_forged_gbesele_target_sets_are_rejected(mode: str) -> None:
    source = Square(9, 4)
    state = state_with(
        [
            (source, piece("imperion", PieceType.IMPERION)),
            (Square(7, 4), piece("one", PieceType.PEASANT, Empire.B)),
            (Square(9, 1), piece("two", PieceType.PEASANT, Empire.B)),
            (Square(5, 8), piece("other", PieceType.PEASANT, Empire.B)),
        ]
    )
    legal = _action(state, source, ActionKind.GBESELE)
    if mode == "subset":
        targets = legal.targets[:-1]
    elif mode == "extra":
        targets = tuple(
            sorted(
                (*legal.targets, ActionTarget(Square(5, 9), "extra")),
                key=lambda target: target.square,
            )
        )
    else:
        targets = tuple(
            ActionTarget(target.square, "stale" if index == 0 else target.piece_id)
            for index, target in enumerate(legal.targets)
        )
    forged = Action(
        legal.piece_id,
        source,
        source,
        ActionKind.GBESELE,
        targets=targets,
    )
    with pytest.raises(IllegalActionError):
        apply_action(state, forged)


def test_gbesele_is_rejected_when_none_is_currently_available() -> None:
    source = Square(9, 4)
    state = state_with([(source, piece("imperion", PieceType.IMPERION))])
    forged = Action(
        "imperion",
        source,
        source,
        ActionKind.GBESELE,
        targets=(ActionTarget(Square(7, 4), "missing"),),
    )
    with pytest.raises(IllegalActionError):
        apply_action(state, forged)


def test_brainwash_changes_only_rules_relevant_allegiance_and_power() -> None:
    source = Square(4, 4)
    target_square = Square(2, 4)
    mystic = piece("mystic", PieceType.WEST_AFRICAN_MYSTIC)
    target = piece("target", PieceType.QUEEN, Empire.B)
    state = state_with([(source, mystic), (target_square, target)], quiet=6, ply=3)
    action = _action(state, source, ActionKind.BRAINWASH)
    result = apply_action(state, action)
    changed_target = result.piece_at(target_square)
    changed_mystic = result.piece_at(source)
    assert changed_target is not None and changed_mystic is not None
    assert changed_target.id == target.id
    assert changed_target.piece_type is target.piece_type
    assert changed_target.original_empire is Empire.B
    assert changed_target.current_empire is Empire.A
    assert changed_target.is_converted
    assert changed_mystic.brainwash is BrainwashAvailability.SPENT
    assert len(result.pieces) == len(state.pieces)
    assert state.piece_at(target_square) is target
    assert state.piece_at(source) is mystic
    assert result.side_to_move is Empire.B
    assert result.ply_number == 4
    assert result.quiet_ply_count == 0


@pytest.mark.parametrize(
    "target_power",
    [BrainwashAvailability.AVAILABLE, BrainwashAvailability.SPENT],
)
def test_brainwashed_mystic_preserves_its_own_power(
    target_power: BrainwashAvailability,
) -> None:
    source = Square(4, 4)
    target_square = Square(2, 4)
    state = state_with(
        [
            (source, piece("actor", PieceType.WEST_AFRICAN_MYSTIC)),
            (
                target_square,
                piece(
                    "target",
                    PieceType.WEST_AFRICAN_MYSTIC,
                    Empire.B,
                    brainwash=target_power,
                ),
            ),
        ]
    )
    result = apply_action(state, _action(state, source, ActionKind.BRAINWASH))
    target = result.piece_at(target_square)
    assert target is not None
    assert target.brainwash is target_power


def test_unused_converted_mystic_can_act_but_spent_one_cannot() -> None:
    source = Square(4, 4)
    target_square = Square(2, 4)
    opposing_square = Square(2, 6)
    available = piece(
        "converted",
        PieceType.WEST_AFRICAN_MYSTIC,
        Empire.A,
        original_empire=Empire.B,
    )
    next_state = state_with(
        [
            (source, available),
            (opposing_square, piece("target", PieceType.PEASANT, Empire.B)),
        ],
        Empire.A,
    )
    assert any(
        action.kind is ActionKind.BRAINWASH
        for action in legal_actions_from(next_state, source)
    )
    spent_state = state_with(
        [
            (
                source,
                replace(available, brainwash=BrainwashAvailability.SPENT),
            ),
            (target_square, piece("target", PieceType.PEASANT, Empire.B)),
        ]
    )
    assert all(
        action.kind not in (ActionKind.BRAINWASH, ActionKind.REBRAINWASH)
        for action in legal_actions_from(spent_state, source)
    )


def test_rebrainwash_restores_original_allegiance_and_preserves_target_power() -> None:
    source = Square(4, 4)
    target_square = Square(2, 4)
    target = piece(
        "target",
        PieceType.WEST_AFRICAN_MYSTIC,
        Empire.B,
        original_empire=Empire.A,
        brainwash=BrainwashAvailability.SPENT,
    )
    state = state_with(
        [
            (source, piece("actor", PieceType.WEST_AFRICAN_MYSTIC)),
            (target_square, target),
        ],
        quiet=5,
    )
    action = _action(state, source, ActionKind.REBRAINWASH)
    result = apply_action(state, action)
    restored = result.piece_at(target_square)
    actor = result.piece_at(source)
    assert restored is not None and actor is not None
    assert restored.current_empire is restored.original_empire is Empire.A
    assert not restored.is_converted
    assert restored.brainwash is BrainwashAvailability.SPENT
    assert actor.brainwash is BrainwashAvailability.SPENT
    assert result.quiet_ply_count == 0


@pytest.mark.parametrize("kind", [ActionKind.BRAINWASH, ActionKind.REBRAINWASH])
def test_forged_or_stale_brainwash_action_is_rejected(kind: ActionKind) -> None:
    source = Square(4, 4)
    target_square = Square(2, 4)
    target = piece(
        "target",
        PieceType.QUEEN,
        Empire.B,
        original_empire=Empire.A if kind is ActionKind.REBRAINWASH else Empire.B,
    )
    state = state_with(
        [
            (source, piece("actor", PieceType.WEST_AFRICAN_MYSTIC)),
            (target_square, target),
        ]
    )
    wrong_kind = (
        ActionKind.REBRAINWASH if kind is ActionKind.BRAINWASH else ActionKind.BRAINWASH
    )
    forged_kind = Action(
        "actor",
        source,
        target_square,
        wrong_kind,
        targets=(ActionTarget(target_square, "target"),),
    )
    stale_id = Action(
        "actor",
        source,
        target_square,
        kind,
        targets=(ActionTarget(target_square, "stale"),),
    )
    with pytest.raises(IllegalActionError):
        apply_action(state, forged_kind)
    with pytest.raises(IllegalActionError):
        apply_action(state, stale_id)


def test_converted_and_restored_peasant_orientation_uses_existing_generation() -> None:
    actor_square = Square(4, 4)
    target_square = Square(2, 4)
    converted = piece("peasant", PieceType.PEASANT, Empire.B, original_empire=Empire.A)
    state = state_with(
        [
            (actor_square, piece("actor", PieceType.WEST_AFRICAN_MYSTIC)),
            (target_square, converted),
        ]
    )
    result = apply_action(state, _action(state, actor_square, ActionKind.REBRAINWASH))
    actions = legal_actions_from(replace(result, side_to_move=Empire.A), target_square)
    destinations = {action.destination for action in actions}
    assert Square(1, 4) in destinations
    assert Square(3, 4) not in destinations
    assert Square(2, 2) not in destinations


def test_brainwashed_peasant_reverses_forward_capture_and_row_eligibility() -> None:
    actor_square = Square(4, 4)
    target_square = Square(2, 4)
    state = state_with(
        [
            (
                actor_square,
                piece("actor", PieceType.WEST_AFRICAN_MYSTIC, Empire.B),
            ),
            (target_square, piece("peasant", PieceType.PEASANT, Empire.A)),
            (Square(3, 3), piece("enemy", PieceType.QUEEN, Empire.A)),
        ],
        Empire.B,
    )
    result = apply_action(state, _action(state, actor_square, ActionKind.BRAINWASH))
    active = replace(result, side_to_move=Empire.B)
    actions = legal_actions_from(active, target_square)
    by_destination = {action.destination: action for action in actions}
    assert Square(3, 4) in by_destination
    assert Square(1, 4) not in by_destination
    assert by_destination[Square(3, 3)].kind is ActionKind.CAPTURE
    assert {Square(2, 2), Square(2, 6)} <= set(by_destination)
