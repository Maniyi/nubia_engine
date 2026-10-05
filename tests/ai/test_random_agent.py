from __future__ import annotations

import random

import pytest

from nubia_ai import RandomAgent
from nubia_engine import Empire, GameState, apply_action, legal_actions


def test_random_action_is_legal_and_state_is_unchanged(small_state: GameState) -> None:
    before = small_state
    action = RandomAgent(Empire.A, seed=7).choose_action(small_state)
    assert action in legal_actions(small_state)
    assert small_state is before


def test_equal_seeds_reproduce_a_state_sequence(small_state: GameState) -> None:
    first = RandomAgent(Empire.A, seed=42)
    second = RandomAgent(Empire.A, seed=42)
    states = [small_state]
    chosen_first = []
    chosen_second = []
    for state in states:
        chosen_first.append(first.choose_action(state))
        chosen_second.append(second.choose_action(state))
    assert chosen_first == chosen_second


def test_different_seeds_cover_different_choices_without_flakiness(
    small_state: GameState,
) -> None:
    decisions = {
        RandomAgent(Empire.A, seed=seed).choose_action(small_state)
        for seed in range(20)
    }
    assert len(decisions) > 1


def test_injected_rng_and_seed_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either"):
        RandomAgent(Empire.A, seed=1, rng=random.Random(1))
    with pytest.raises(TypeError):
        RandomAgent(Empire.A, rng=object())  # type: ignore[arg-type]


def test_rng_state_advances_without_global_random(small_state: GameState) -> None:
    global_state = random.getstate()
    agent = RandomAgent(Empire.A, seed=3)
    _ = [agent.choose_action(small_state) for _ in range(4)]
    assert random.getstate() == global_state


def test_same_agents_follow_same_alternating_states(small_state: GameState) -> None:
    action = RandomAgent(Empire.A, seed=8).choose_action(small_state)
    successor = apply_action(small_state, action)
    assert successor.side_to_move is Empire.B
