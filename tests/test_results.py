from dataclasses import FrozenInstanceError

import pytest

from nubia_engine import Empire, GameResult, Outcome, ResultReason


def test_result_reasons_are_unique_and_canonical() -> None:
    result = GameResult(
        Outcome.DRAW,
        None,
        (
            ResultReason.NO_PROGRESS_40_PLIES,
            ResultReason.THREEFOLD_REPETITION,
            ResultReason.THREEFOLD_REPETITION,
        ),
    )
    assert result.reasons == (
        ResultReason.THREEFOLD_REPETITION,
        ResultReason.NO_PROGRESS_40_PLIES,
    )
    with pytest.raises(FrozenInstanceError):
        result.winner = Empire.A  # type: ignore[misc]


@pytest.mark.parametrize(
    "result",
    [
        lambda: GameResult(Outcome.WIN, None, (ResultReason.MINE_VICTORY,)),
        lambda: GameResult(
            Outcome.DRAW, Empire.A, (ResultReason.THREEFOLD_REPETITION,)
        ),
        lambda: GameResult(Outcome.DRAW, None, ()),
        lambda: GameResult(Outcome.WIN, Empire.A, (ResultReason.THREEFOLD_REPETITION,)),
        lambda: GameResult(
            Outcome.WIN,
            Empire.A,
            (ResultReason.MINE_VICTORY,),
            (1, 0),
        ),
        lambda: GameResult(
            Outcome.WIN,
            Empire.A,
            (ResultReason.NO_PEASANTS_SCORING,),
            (1, 2),
        ),
    ],
)
def test_invalid_result_combinations_are_rejected(result: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        result()  # type: ignore[operator]


def test_scoring_totals_are_exposed_in_empire_order() -> None:
    result = GameResult(
        Outcome.DRAW,
        None,
        (ResultReason.NO_PEASANTS_SCORING,),
        (8, 8),
    )
    assert result.scores == result.officer_score_totals == (8, 8)
    assert result.score_a == result.score_b == 8
