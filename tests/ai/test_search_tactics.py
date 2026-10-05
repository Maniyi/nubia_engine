from __future__ import annotations

from conftest import piece, state_with

from nubia_ai import HeuristicAgent, MinimaxAgent, SearchConfig
from nubia_engine import (
    Action,
    Empire,
    GameState,
    Outcome,
    PieceType,
    Square,
    apply_action,
    legal_actions,
)


def _mine_threat_state() -> GameState:
    # B's "threat" Peasant on A:b2 wins next by advancing to A:o2. A's Advisor
    # on A:o1 can occupy that mine without capturing the threat. The tempting
    # A Peasant has a more attractive one-ply forward move, so only reply-aware
    # search values the quiet Advisor defense correctly.
    return state_with(
        [
            (Square(6, 0), piece("block", PieceType.SOUTH_AFRICAN_ADVISOR)),
            (Square(5, 4), piece("bait", PieceType.PEASANT)),
            (Square(5, 1), piece("threat", PieceType.PEASANT, Empire.B)),
            (Square(2, 9), piece("other-b", PieceType.PEASANT, Empire.B)),
        ]
    )


def test_depth_two_avoids_a_documented_immediate_mine_loss() -> None:
    state = _mine_threat_state()
    shallow = HeuristicAgent(Empire.A).choose_action(state)
    assert shallow.piece_id == "bait"
    after_shallow = apply_action(state, shallow)

    def wins(state: GameState, action: Action) -> bool:
        result = apply_action(state, action).result
        return result is not None and result.outcome is Outcome.WIN

    winning_replies = [
        action for action in legal_actions(after_shallow) if wins(after_shallow, action)
    ]
    assert any(action.piece_id == "threat" for action in winning_replies)

    deep = MinimaxAgent(Empire.A, SearchConfig(2)).search(state)
    assert deep.action.piece_id == "block"
    assert deep.action.destination == Square.from_notation("A:o2")
    after_defense = apply_action(state, deep.action)
    assert all(not wins(after_defense, reply) for reply in legal_actions(after_defense))


def test_threat_reply_appears_in_the_shallow_actions_searched_by_depth_two() -> None:
    state = _mine_threat_state()
    shallow = HeuristicAgent(Empire.A).choose_action(state)
    after_shallow = apply_action(state, shallow)
    mine = Square.from_notation("A:o2")
    assert any(
        reply.piece_id == "threat" and reply.destination == mine
        for reply in legal_actions(after_shallow)
    )
