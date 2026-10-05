import pytest

from nubia_engine import Action, ActionKind, ActionTarget, Square


def test_special_action_targets_are_immutable_values() -> None:
    target = ActionTarget(Square(2, 2), "target")
    action = Action(
        "mystic",
        Square(4, 4),
        Square(2, 2),
        ActionKind.BRAINWASH,
        targets=(target,),
    )
    assert action == Action(
        "mystic",
        Square(4, 4),
        Square(2, 2),
        ActionKind.BRAINWASH,
        targets=(target,),
    )
    assert action.target_piece_id == "target"


@pytest.mark.parametrize(
    "targets",
    [
        (ActionTarget(Square(2, 2), "same"), ActionTarget(Square(3, 3), "same")),
        (ActionTarget(Square(2, 2), "one"), ActionTarget(Square(2, 2), "two")),
    ],
)
def test_special_action_rejects_duplicate_target_ids_or_squares(
    targets: tuple[ActionTarget, ...],
) -> None:
    with pytest.raises(ValueError, match="unique"):
        Action(
            "imperion",
            Square(9, 4),
            Square(9, 4),
            ActionKind.GBESELE,
            targets=targets,
        )


def test_gbesele_requires_canonical_complete_shape() -> None:
    with pytest.raises(ValueError, match="canonical"):
        Action(
            "imperion",
            Square(9, 4),
            Square(9, 4),
            ActionKind.GBESELE,
            targets=(
                ActionTarget(Square(8, 4), "later"),
                ActionTarget(Square(7, 4), "earlier"),
            ),
        )
    with pytest.raises(ValueError, match="source"):
        Action(
            "imperion",
            Square(9, 4),
            Square(8, 4),
            ActionKind.GBESELE,
            targets=(ActionTarget(Square(7, 4), "target"),),
        )


def test_brainwash_requires_exactly_its_destination_target() -> None:
    with pytest.raises(ValueError, match="exactly"):
        Action(
            "mystic",
            Square(4, 4),
            Square(2, 4),
            ActionKind.BRAINWASH,
            targets=(ActionTarget(Square(4, 2), "target"),),
        )
    with pytest.raises(ValueError, match="identify"):
        Action(
            "mystic",
            Square(4, 4),
            Square(2, 4),
            ActionKind.BRAINWASH,
        )


def test_special_target_and_action_reject_malformed_values() -> None:
    with pytest.raises(TypeError):
        ActionTarget("square", "target")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ActionTarget(Square(2, 2), "")
    with pytest.raises(TypeError):
        Action(
            "mystic",
            Square(4, 4),
            Square(2, 4),
            ActionKind.BRAINWASH,
            targets=[ActionTarget(Square(2, 4), "target")],  # type: ignore[arg-type]
        )


def test_ordinary_action_cannot_carry_special_targets() -> None:
    with pytest.raises(ValueError, match="ordinary"):
        Action(
            "queen",
            Square(4, 4),
            Square(4, 5),
            ActionKind.MOVE,
            targets=(ActionTarget(Square(4, 5), "target"),),
        )
