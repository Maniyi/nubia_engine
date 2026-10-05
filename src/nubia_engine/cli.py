"""Testable two-human terminal interface for NUBIA."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from nubia_engine.enums import Empire
from nubia_engine.models import GameState
from nubia_engine.move_generation import legal_actions
from nubia_engine.notation import format_action
from nubia_engine.rendering import render_board, render_result
from nubia_engine.setup import create_initial_state
from nubia_engine.transitions import apply_action

Input = Callable[[str], str]
Output = Callable[[str], None]


def play_game(
    initial_state: GameState,
    *,
    input_fn: Input = input,
    output_fn: Output = print,
) -> GameState:
    """Run a two-human session and return its final or voluntarily quit state."""

    if not isinstance(initial_state, GameState):
        raise TypeError("initial_state must be a GameState")
    state = initial_state
    while True:
        output_fn(render_board(state))
        if state.result is not None:
            output_fn(render_result(state))
            return state
        actions = legal_actions(state)
        output_fn(f"Legal actions for Empire {state.side_to_move.value}:")
        for number, action in enumerate(actions, start=1):
            output_fn(f"{number:>3}. {format_action(state, action)}")
        command = input_fn("Select action number (h=help, q=quit): ").strip()
        if command.lower() == "q":
            output_fn("Game quit; no result recorded.")
            return state
        if command.lower() == "h":
            output_fn("Enter a listed number to move, h for help, or q to quit.")
            continue
        try:
            selection = int(command)
        except ValueError:
            output_fn("Invalid input. Enter a number, h, or q.")
            continue
        if not 1 <= selection <= len(actions):
            output_fn(f"Selection out of range. Choose 1-{len(actions)}.")
            continue
        state = apply_action(state, actions[selection - 1])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""

    parser = argparse.ArgumentParser(description="Play two-human NUBIA")
    parser.add_argument(
        "--first-player",
        choices=[empire.value for empire in Empire],
        default=Empire.A.value,
        help="first empire to move (default: A)",
    )
    args = parser.parse_args(argv)
    play_game(create_initial_state(Empire(args.first_player)))
    return 0
