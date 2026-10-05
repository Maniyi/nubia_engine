from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_ai import DEFAULT_WEIGHTS, EvaluationWeights, evaluate_state
from nubia_engine import (
    BrainwashAvailability,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
)


def _terminal(winner: Empire | None):  # type: ignore[no-untyped-def]
    state = state_with([(Square(7, 0), piece("p", PieceType.PEASANT))])
    result = (
        GameResult(Outcome.WIN, winner, (ResultReason.MINE_VICTORY,))
        if winner is not None
        else GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,))
    )
    return replace(state, result=result)


@pytest.mark.parametrize(
    ("winner", "expected"),
    [
        (Empire.A, DEFAULT_WEIGHTS.terminal),
        (Empire.B, -DEFAULT_WEIGHTS.terminal),
        (None, 0),
    ],
)
def test_terminal_scores_dominate_and_suppress_features(
    winner: Empire | None, expected: int
) -> None:
    score = evaluate_state(_terminal(winner), Empire.A)
    assert score.total == score.terminal == expected
    assert score.officer_material == score.peasant_presence == 0
    assert abs(score.total) > 100_000 or score.total == 0


def test_officer_material_and_mystic_power_use_official_values() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("q", PieceType.QUEEN)),
            (
                Square(5, 4),
                piece(
                    "m",
                    PieceType.WEST_AFRICAN_MYSTIC,
                    Empire.B,
                    brainwash=BrainwashAvailability.SPENT,
                ),
            ),
        ]
    )
    weights = replace(DEFAULT_WEIGHTS, officer_material=1)
    assert evaluate_state(state, Empire.A, weights).officer_material == 6
    unused = replace(
        state,
        board=tuple(
            replace(p, brainwash=BrainwashAvailability.AVAILABLE)
            if p is not None and p.id == "m"
            else p
            for p in state.board
        ),
    )
    assert evaluate_state(unused, Empire.A, weights).officer_material == 2


def test_current_allegiance_controls_material_and_peasant_features() -> None:
    converted = state_with(
        [
            (
                Square(4, 4),
                piece("q", PieceType.QUEEN, Empire.A, original_empire=Empire.B),
            ),
            (
                Square(7, 0),
                piece("p", PieceType.PEASANT, Empire.A, original_empire=Empire.B),
            ),
        ]
    )
    score = evaluate_state(converted, Empire.A)
    assert score.officer_material > 0
    assert score.peasant_presence > 0
    assert score.peasant_progress > 0


def test_peasant_progress_rewards_closer_opposing_resources() -> None:
    far = state_with([(Square.from_notation("A:c2"), piece("p", PieceType.PEASANT))])
    near = state_with([(Square.from_notation("B:b2"), piece("p", PieceType.PEASANT))])
    assert (
        evaluate_state(near, Empire.A).peasant_progress
        > evaluate_state(far, Empire.A).peasant_progress
    )


def test_mobility_has_perspective_sign_and_symmetry(small_state) -> None:  # type: ignore[no-untyped-def]
    a = evaluate_state(small_state, Empire.A)
    b = evaluate_state(small_state, Empire.B)
    assert a.mobility > 0 and b.mobility < 0
    assert a.total == -b.total
    assert a.total == (
        a.officer_material + a.peasant_presence + a.peasant_progress + a.mobility
    )
    assert evaluate_state(small_state, Empire.A) == a


def test_evaluation_validates_inputs_and_weight_bound(small_state) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TypeError):
        evaluate_state("state", Empire.A)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_state(small_state, "A")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EvaluationWeights(mobility=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="terminal"):
        EvaluationWeights(terminal=1)
