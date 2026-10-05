from dataclasses import replace

import pytest
from conftest import state_with

from nubia_engine import Empire, GameResult, Outcome, ResultReason, create_initial_state
from nubia_engine.cli import main, play_game


def _session(commands: list[str], state=None):  # type: ignore[no-untyped-def]
    prompts: list[str] = []
    output: list[str] = []
    iterator = iter(commands)

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(iterator)

    final = play_game(
        state or create_initial_state(Empire.A), input_fn=read, output_fn=output.append
    )
    return final, prompts, output


def test_immediate_quit_shows_board_and_stable_numbered_menu() -> None:
    state = create_initial_state(Empire.A)
    final, prompts, output = _session(["q"], state)
    assert final is state and final.result is None
    assert "NUBIA board" in output[0]
    assert output[1] == "Legal actions for Empire A:"
    assert output[2].startswith("  1. ")
    assert len(prompts) == 1
    assert output[-1] == "Game quit; no result recorded."


@pytest.mark.parametrize(
    ("commands", "message"),
    [
        (["h", "q"], "Enter a listed number"),
        (["wat", "q"], "Invalid input"),
        (["999", "q"], "out of range"),
    ],
)
def test_help_invalid_and_out_of_range_do_not_change_state(
    commands: list[str], message: str
) -> None:
    state = create_initial_state(Empire.B)
    final, _prompts, output = _session(commands, state)
    assert final is state
    assert any(message in line for line in output)
    assert output.count("Legal actions for Empire B:") == 2


def test_one_valid_move_then_quit_renders_resulting_board() -> None:
    initial = create_initial_state(Empire.A)
    final, prompts, output = _session(["1", "q"], initial)
    assert final.ply_number == 1 and final.side_to_move is Empire.B
    assert len(prompts) == 2
    assert sum("NUBIA board" in line for line in output) == 2


def test_terminal_result_is_displayed_without_prompt() -> None:
    terminal = replace(
        state_with([]),
        result=GameResult(
            Outcome.DRAW, None, (ResultReason.NO_PEASANTS_SCORING,), (0, 0)
        ),
    )
    final, prompts, output = _session([], terminal)
    assert final is terminal and prompts == []
    assert any("officer totals A=0, B=0" in line for line in output)


def test_main_accepts_both_explicit_first_players(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Empire] = []
    monkeypatch.setattr(
        "nubia_engine.cli.play_game", lambda state: seen.append(state.side_to_move)
    )
    assert main(["--first-player", "A"]) == 0
    assert main(["--first-player", "B"]) == 0
    assert seen == [Empire.A, Empire.B]


def test_play_game_validates_initial_state() -> None:
    with pytest.raises(TypeError):
        play_game("state")  # type: ignore[arg-type]
