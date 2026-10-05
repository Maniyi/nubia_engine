from __future__ import annotations

from dataclasses import replace

from conftest import piece, state_with

from nubia_ai import DEFAULT_WEIGHTS, HeuristicAgent
from nubia_engine import ActionKind, Empire, GameState, PieceType, Square, legal_actions


def test_immediate_mine_victory_is_selected() -> None:
    mine = Square.from_notation("B:o2")
    state = state_with(
        [
            (Square(4, mine.column), piece("p", PieceType.PEASANT)),
            (Square(7, 8), piece("other", PieceType.PEASANT)),
            (Square(2, 8), piece("enemy", PieceType.PEASANT, Empire.B)),
        ]
    )
    chosen = HeuristicAgent(Empire.A).choose_action(state)
    assert chosen.destination == mine


def test_rankings_follow_engine_order_and_ties_choose_first() -> None:
    state = state_with(
        [
            (Square(9, 4), piece("i", PieceType.IMPERION)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    weights = replace(
        DEFAULT_WEIGHTS,
        officer_material=0,
        peasant_presence=0,
        peasant_progress=0,
        mobility=0,
    )
    agent = HeuristicAgent(Empire.A, weights=weights)
    ranked = agent.rank_actions(state)
    assert tuple(item.action for item in ranked) == legal_actions(state)
    assert len({item.score for item in ranked}) == 1
    assert agent.choose_action(state) == legal_actions(state)[0]


def test_agent_returns_highest_scored_legal_action_without_mutation(
    small_state: GameState,
) -> None:
    before = small_state
    agent = HeuristicAgent(Empire.A)
    ranked = agent.rank_actions(small_state)
    chosen = agent.choose_action(small_state)
    assert chosen in legal_actions(small_state)
    assert next(item for item in ranked if item.action == chosen).score == max(
        item.score for item in ranked
    )
    assert small_state is before


def test_special_action_is_ranked_and_can_be_selected() -> None:
    source = Square(4, 4)
    state = state_with(
        [
            (source, piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("p", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    ranked = HeuristicAgent(Empire.A).rank_actions(state)
    assert any(item.action.kind is ActionKind.BRAINWASH for item in ranked)
    assert HeuristicAgent(Empire.A).choose_action(state).kind is ActionKind.BRAINWASH
