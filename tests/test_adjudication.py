from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    BrainwashAvailability,
    Empire,
    GameState,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    adjudicate,
    apply_action,
    legal_actions_from,
    officer_scores,
    position_key,
)


def _action_to(state: GameState, source: Square, destination: Square) -> Action:
    return next(
        action
        for action in legal_actions_from(state, source)
        if action.destination == destination
    )


def test_move_and_capture_onto_opposing_mine_win_immediately() -> None:
    mine = Square.from_notation("B:o6")
    for source, target in ((Square(4, mine.column), None), (Square(4, 3), mine)):
        placements = [(source, piece("p", PieceType.PEASANT))]
        if target is not None:
            placements.append((mine, piece("target", PieceType.QUEEN, Empire.B)))
        state = state_with(placements)
        result = apply_action(state, _action_to(state, source, mine))
        assert result.result is not None
        assert result.result.outcome is Outcome.WIN
        assert result.result.winner is Empire.A
        assert result.result.reasons == (ResultReason.MINE_VICTORY,)


def test_brainwash_on_mine_uses_resulting_current_allegiance() -> None:
    source = Square(5, 4)
    mine = Square.from_notation("B:o6")
    state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (mine, piece("p", PieceType.PEASANT, Empire.B)),
        ]
    )
    action = next(
        action
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.BRAINWASH
    )
    result = apply_action(state, action)
    assert result.result is not None
    assert result.result.winner is Empire.A
    assert result.result.reasons == (ResultReason.MINE_VICTORY,)


def test_own_mine_and_non_peasant_do_not_win() -> None:
    own_mine = Square.from_notation("A:o2")
    peasant_state = state_with([(own_mine, piece("p", PieceType.PEASANT))])
    officer_state = state_with([(own_mine, piece("q", PieceType.QUEEN, Empire.B))])
    assert adjudicate(peasant_state) is None
    scoring = adjudicate(officer_state)
    assert scoring is not None
    assert scoring.reasons == (ResultReason.NO_PEASANTS_SCORING,)


@pytest.mark.parametrize(
    ("piece_type", "power", "value"),
    [
        (PieceType.NORTH_CENTRAL_WAR_CHIEF, None, 3),
        (PieceType.EAST_AFRICAN_HIGH_CHIEF, None, 5),
        (PieceType.SOUTH_AFRICAN_ADVISOR, None, 5),
        (PieceType.QUEEN, None, 9),
        (PieceType.WEST_AFRICAN_MYSTIC, BrainwashAvailability.AVAILABLE, 7),
        (PieceType.WEST_AFRICAN_MYSTIC, BrainwashAvailability.SPENT, 3),
        (PieceType.IMPERION, None, 0),
        (PieceType.PEASANT, None, 0),
    ],
)
def test_every_officer_score_value(
    piece_type: PieceType,
    power: BrainwashAvailability | None,
    value: int,
) -> None:
    placed = (
        piece("x", piece_type)
        if power is None
        else piece("x", piece_type, brainwash=power)
    )
    state = state_with([(Square(4, 4), placed)])
    assert officer_scores(state) == (value, 0)


def test_scoring_waits_for_all_peasants_and_uses_current_empire() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("p", PieceType.PEASANT)),
            (
                Square(5, 5),
                piece("q", PieceType.QUEEN, Empire.B, original_empire=Empire.A),
            ),
        ]
    )
    assert adjudicate(state) is None
    without_peasant = replace(
        state,
        board=tuple(
            None
            if placed is not None and placed.piece_type is PieceType.PEASANT
            else placed
            for placed in state.board
        ),
    )
    result = adjudicate(without_peasant)
    assert result is not None
    assert result.winner is Empire.B
    assert result.scores == (0, 9)


def test_final_peasant_capture_triggers_scoring_before_draws() -> None:
    source, target = Square(4, 4), Square(4, 6)
    state = state_with(
        [
            (source, piece("q", PieceType.QUEEN)),
            (target, piece("p", PieceType.PEASANT, Empire.B)),
        ],
        quiet=39,
    )
    result = apply_action(state, _action_to(state, source, target))
    assert result.result is not None
    assert result.result.reasons == (ResultReason.NO_PEASANTS_SCORING,)
    assert result.result.scores == (9, 0)


def test_equal_officer_scores_draw() -> None:
    state = state_with(
        [
            (Square(4, 4), piece("a", PieceType.QUEEN)),
            (Square(5, 5), piece("b", PieceType.QUEEN, Empire.B)),
        ]
    )
    result = adjudicate(state)
    assert result is not None
    assert result.outcome is Outcome.DRAW
    assert result.scores == (9, 9)


def test_mine_victory_suppresses_scoring_and_draws() -> None:
    mine = Square.from_notation("B:o6")
    state = state_with([(mine, piece("p", PieceType.PEASANT))], quiet=40)
    key = position_key(state)
    state = replace(state, position_history=(key, key, key))
    result = adjudicate(state)
    assert result is not None
    assert result.reasons == (ResultReason.MINE_VICTORY,)
