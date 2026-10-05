from dataclasses import FrozenInstanceError

import pytest
from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    Empire,
    PieceType,
    Square,
    legal_actions,
    legal_actions_from,
)


def test_action_is_immutable_and_compares_by_value() -> None:
    first = Action("q", Square(4, 4), Square(4, 5), ActionKind.MOVE)
    assert first == Action("q", Square(4, 4), Square(4, 5), ActionKind.MOVE)
    with pytest.raises(FrozenInstanceError):
        first.piece_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"piece_id": ""},
        {"source": Square(4, 5)},
        {"kind": "MOVE"},
        {"source": "A:b1"},
        {"destination": "A:b2"},
    ],
)
def test_action_rejects_malformed_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "piece_id": "q",
        "source": Square(4, 4),
        "destination": Square(4, 5),
        "kind": ActionKind.MOVE,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        Action(**values)  # type: ignore[arg-type]


def test_action_capture_identifier_invariants() -> None:
    with pytest.raises(ValueError):
        Action("q", Square(4, 4), Square(4, 5), ActionKind.CAPTURE)
    with pytest.raises(ValueError):
        Action("q", Square(4, 4), Square(4, 5), ActionKind.MOVE, "enemy")


def test_generation_is_immutable_unique_deterministic_and_sorted() -> None:
    state = state_with(
        [
            (Square(7, 7), piece("q2", PieceType.QUEEN)),
            (Square(2, 2), piece("q1", PieceType.QUEEN)),
        ]
    )
    first = legal_actions(state)
    assert isinstance(first, tuple)
    assert first == legal_actions(state)
    assert len(first) == len(set(first))
    keys = [
        (
            action.source.row * 10 + action.source.column,
            action.destination.row * 10 + action.destination.column,
            action.kind.value,
        )
        for action in first
    ]
    assert keys == sorted(keys)


def test_only_active_empire_generates_and_empty_source_is_empty() -> None:
    own = piece("own", PieceType.QUEEN)
    opposing = piece("other", PieceType.QUEEN, Empire.B)
    state = state_with([(Square(2, 2), own), (Square(7, 7), opposing)])
    assert legal_actions_from(state, Square(2, 2))
    assert legal_actions_from(state, Square(7, 7)) == ()
    assert legal_actions_from(state, Square(0, 0)) == ()
    assert all(action.piece_id == "own" for action in legal_actions(state))


def test_generation_validates_public_arguments() -> None:
    state = state_with([])
    with pytest.raises(TypeError):
        legal_actions("state")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        legal_actions_from(state, "A:p1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        legal_actions_from("state", Square(0, 0))  # type: ignore[arg-type]


def test_no_generated_capture_targets_an_imperion() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("q", PieceType.QUEEN)),
            (Square(4, 7), piece("i", PieceType.IMPERION, Empire.B)),
        ]
    )
    actions = legal_actions_from(state, Square(4, 4))
    assert Square(4, 7) not in {action.destination for action in actions}
    assert Square(4, 8) not in {action.destination for action in actions}
