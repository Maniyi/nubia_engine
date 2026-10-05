"""Seeded uniform-random baseline agent."""

from __future__ import annotations

import random

from nubia_ai.agent import available_actions_for
from nubia_engine import Action, Empire, GameState


class RandomAgent:
    """A stateful, reproducible uniform-random baseline policy."""

    def __init__(
        self,
        empire: Empire,
        *,
        seed: int | None = None,
        rng: random.Random | None = None,
        name: str = "RandomAgent",
    ) -> None:
        if not isinstance(empire, Empire):
            raise TypeError("empire must be an Empire")
        if rng is not None and seed is not None:
            raise ValueError("provide either seed or rng, not both")
        if rng is not None and not isinstance(rng, random.Random):
            raise TypeError("rng must be a random.Random")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.empire = empire
        self.name = name
        self._rng = rng if rng is not None else random.Random(seed)

    def choose_action(self, state: GameState) -> Action:
        """Choose uniformly from the engine's current legal-action tuple."""

        actions = available_actions_for(self, state)
        return actions[self._rng.randrange(len(actions))]
