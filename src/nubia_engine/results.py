"""Immutable global game results and their invariants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nubia_engine.enums import Empire


class Outcome(Enum):
    """The global outcome of a completed game."""

    WIN = "win"
    DRAW = "draw"


class ResultReason(Enum):
    """A rules condition responsible for a terminal result."""

    MINE_VICTORY = "mine_victory"
    NO_PEASANTS_SCORING = "no_peasants_scoring"
    THREEFOLD_REPETITION = "threefold_repetition"
    NO_PROGRESS_40_PLIES = "no_progress_40_plies"


_REASON_ORDER = {reason: index for index, reason in enumerate(ResultReason)}
_DRAW_REASONS = {
    ResultReason.THREEFOLD_REPETITION,
    ResultReason.NO_PROGRESS_40_PLIES,
}


@dataclass(frozen=True, slots=True)
class GameResult:
    """A validated, immutable result shared by both players.

    ``scores`` is ``(Empire A total, Empire B total)`` and is present exactly
    when the result was decided by no-Peasant officer scoring.
    """

    outcome: Outcome
    winner: Empire | None
    reasons: tuple[ResultReason, ...]
    scores: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            raise TypeError("outcome must be an Outcome")
        if self.winner is not None and not isinstance(self.winner, Empire):
            raise TypeError("winner must be an Empire or None")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("reasons must be a non-empty tuple")
        if any(not isinstance(reason, ResultReason) for reason in self.reasons):
            raise TypeError("reasons must contain only ResultReason values")

        canonical = tuple(sorted(set(self.reasons), key=_REASON_ORDER.__getitem__))
        object.__setattr__(self, "reasons", canonical)

        if self.outcome is Outcome.WIN and self.winner is None:
            raise ValueError("a win must identify exactly one winner")
        if self.outcome is Outcome.DRAW and self.winner is not None:
            raise ValueError("a draw cannot identify a winner")

        scoring = ResultReason.NO_PEASANTS_SCORING in canonical
        if scoring:
            if canonical != (ResultReason.NO_PEASANTS_SCORING,):
                raise ValueError("scoring suppresses all lower-priority reasons")
            if (
                not isinstance(self.scores, tuple)
                or len(self.scores) != 2
                or any(
                    isinstance(score, bool) or not isinstance(score, int) or score < 0
                    for score in self.scores
                )
            ):
                raise ValueError("scoring results require two non-negative totals")
            score_a, score_b = self.scores
            expected_winner = (
                Empire.A
                if score_a > score_b
                else Empire.B
                if score_b > score_a
                else None
            )
            expected_outcome = (
                Outcome.WIN if expected_winner is not None else Outcome.DRAW
            )
            if (
                self.outcome is not expected_outcome
                or self.winner is not expected_winner
            ):
                raise ValueError("scoring outcome must agree with its totals")
        elif self.scores is not None:
            raise ValueError("score totals are only valid for officer scoring")

        if ResultReason.MINE_VICTORY in canonical:
            if (
                canonical != (ResultReason.MINE_VICTORY,)
                or self.outcome is not Outcome.WIN
            ):
                raise ValueError("mine victory must be a single winning reason")
        elif not scoring and (
            self.outcome is not Outcome.DRAW or not set(canonical) <= _DRAW_REASONS
        ):
            raise ValueError("repetition and no-progress reasons must be draws")

    @property
    def score_a(self) -> int | None:
        """Empire A's officer total, when scoring decided the result."""

        return None if self.scores is None else self.scores[0]

    @property
    def score_b(self) -> int | None:
        """Empire B's officer total, when scoring decided the result."""

        return None if self.scores is None else self.scores[1]

    @property
    def officer_score_totals(self) -> tuple[int, int] | None:
        """Both officer totals in canonical Empire A, Empire B order."""

        return self.scores
