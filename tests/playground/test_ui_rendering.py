from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest
from conftest import piece, state_with

from nubia_engine import (
    ActionKind,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    legal_actions,
)
from nubia_playground.app import PlaygroundApp, build_parser, main
from nubia_playground.controller import (
    AgentKind,
    GameController,
    GameMode,
    SetupConfig,
)
from nubia_playground.panels import Button, draw_lines


@pytest.fixture
def app() -> Iterator[PlaygroundApp]:
    instance = PlaygroundApp()
    yield instance
    instance.running = False
    pygame.quit()


def test_setup_draws_every_contextual_agent_configuration(app: PlaygroundApp) -> None:
    app._draw()
    assert app.buttons

    app.setup.mode = GameMode.HUMAN_VS_HUMAN
    app._draw()
    app._cycle_mode()
    assert app.setup.mode is GameMode.HUMAN_VS_AGENT

    app.setup.human = Empire.B
    for kind in AgentKind:
        app.setup.kinds[Empire.A] = kind
        app._draw()
    app._cycle_kind(Empire.A)
    app._toggle_budget(Empire.A)
    assert app.setup.budget_kind[Empire.A] == "Time (ms)"

    app.setup.mode = GameMode.AGENT_VS_AGENT
    app.setup.kinds[Empire.A] = AgentKind.ITERATIVE
    app.setup.kinds[Empire.B] = AgentKind.MINIMAX
    app.setup.error = "visible problem"
    app._draw()


def test_setup_keyboard_editing_and_valid_start(app: PlaygroundApp) -> None:
    app.setup.active_field = "delay"
    app.setup.values["delay"] = "500"
    app._key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode=""))
    app._key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, unicode="0"))
    app._key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode="x"))
    assert app.setup.values["delay"] == "500"
    app._start()
    assert app.controller is not None

    action = legal_actions(app.controller.state)[0]
    app.controller.select_square(action.source)
    app._key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode=""))
    assert app.controller.selected_source is None


def test_hidden_irrelevant_numeric_fields_do_not_block_setup(
    app: PlaygroundApp,
) -> None:
    app.setup.kinds[Empire.A] = AgentKind.HEURISTIC
    app.setup.values["seed_A"] = "not used"
    app.setup.values["depth_A"] = "not used"
    app.setup.values["max_A"] = "not used"
    app.setup.values["budget_A"] = "not used"
    app.setup.mode = GameMode.AGENT_VS_AGENT
    app._start()
    assert app.controller is not None


def test_game_draws_modes_selections_errors_results_and_sizes(
    app: PlaygroundApp,
) -> None:
    for size in ((1024, 768), (1440, 900)):
        app.screen = pygame.display.set_mode(size)
        app.controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
        action = legal_actions(app.controller.state)[0]
        app.controller.select_square(action.source)
        app._draw()
        assert app._geometry().pixel_to_square(
            app._geometry().square_center(action.source)
        )

    app.controller = GameController(SetupConfig(mode=GameMode.AGENT_VS_AGENT))
    app.controller.error = "agent failed"
    app._draw()
    app.controller.toggle_pause()
    app._draw()

    app.controller.state = replace(
        app.controller.state,
        result=GameResult(
            Outcome.DRAW,
            None,
            (ResultReason.NO_PROGRESS_40_PLIES,),
        ),
    )
    app._draw()


def test_special_action_labels_show_targets_and_engine_result(
    app: PlaygroundApp,
) -> None:
    state = state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("ap", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    app.controller = GameController(
        SetupConfig(mode=GameMode.HUMAN_VS_HUMAN), state=state
    )
    app.controller.select_square(Square(4, 4))
    app.controller.select_square(Square(2, 4))
    brainwash = next(
        action
        for action in app.controller.action_choices
        if action.kind is ActionKind.BRAINWASH
    )
    label = app._action_label(brainwash)
    assert "targets:" in label and "result: Empire A" in label
    app._draw()


def test_agent_future_applies_one_action_and_controls_reset(app: PlaygroundApp) -> None:
    app.controller = GameController(SetupConfig(mode=GameMode.AGENT_VS_AGENT))
    app._request_agent(force=True)
    assert app.controller.thinking
    assert app.agent_future is not None
    app.agent_future.result(timeout=5)
    app._update_agent()
    assert app.controller.state.ply_number == 1
    assert app.controller.paused

    app._restart()
    assert app.controller.state.ply_number == 0
    app._step()
    assert app.agent_future is not None
    app.agent_future.result(timeout=5)
    app._update_agent()

    app._to_setup()
    assert app.controller is None
    app._quit()
    assert not app.running


def test_resumed_agent_game_requests_automatic_move(app: PlaygroundApp) -> None:
    app.controller = GameController(SetupConfig(mode=GameMode.AGENT_VS_AGENT))
    app.controller.toggle_pause()
    app.last_move_time = time.monotonic() - 1
    app._update_agent()
    assert app.agent_future is not None
    app.agent_future.result(timeout=5)
    app._update_agent()
    assert app.controller.state.ply_number == 1


def test_event_loop_handles_board_click_button_click_and_quit(
    app: PlaygroundApp,
) -> None:
    app.controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    source = legal_actions(app.controller.state)[0].source
    pygame.event.post(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=app._geometry().square_center(source),
        )
    )
    app._events()
    assert app.controller.selected_source == source

    called: list[bool] = []
    app.buttons = [Button("Do", pygame.Rect(0, 0, 50, 50), lambda: called.append(True))]
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(10, 10))
    )
    app._events()
    assert called == [True]

    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app._events()
    assert not app.running


def test_widget_enabled_disabled_and_line_drawing(app: PlaygroundApp) -> None:
    calls: list[str] = []
    enabled = Button("Enabled", pygame.Rect(0, 0, 100, 30), lambda: calls.append("yes"))
    disabled = Button(
        "Disabled", pygame.Rect(0, 40, 100, 30), lambda: calls.append("no"), False
    )
    enabled.draw(app.screen, app.small_font)
    disabled.draw(app.screen, app.small_font)
    assert enabled.handle((5, 5))
    assert not disabled.handle((5, 45))
    assert calls == ["yes"]
    assert draw_lines(app.screen, ["one", "two"], (5, 80), app.small_font) > 80


def test_bounded_run_and_public_main_close_cleanly() -> None:
    assert build_parser().parse_args(["--smoke-test"]).smoke_test
    instance = PlaygroundApp()
    assert instance.run(max_frames=1) == 0
    assert main(["--smoke-test"]) == 0
