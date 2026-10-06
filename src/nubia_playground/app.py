"""Responsive local Pygame application for NUBIA."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

try:
    import pygame
except ImportError:  # pragma: no cover - exercised only without the UI extra
    pygame = None  # type: ignore[assignment]

from nubia_engine import (
    Action,
    ActionKind,
    Empire,
    apply_action,
    format_action,
    render_result,
)
from nubia_playground import theme
from nubia_playground.board_view import BoardGeometry, DisplayMode, draw_board
from nubia_playground.controller import (
    AgentKind,
    AgentSettings,
    GameController,
    GameMode,
    SetupConfig,
    SetupValidationError,
)
from nubia_playground.panels import Button, draw_lines
from nubia_playground.piece_assets import PIECE_ASSETS, PieceIconStore


@dataclass(slots=True)
class SetupScreen:
    """Mutable text-entry state which produces an immutable SetupConfig."""

    mode: GameMode = GameMode.HUMAN_VS_AGENT
    first: Empire = Empire.A
    human: Empire = Empire.A
    kinds: dict[Empire, AgentKind] = field(
        default_factory=lambda: {
            Empire.A: AgentKind.RANDOM,
            Empire.B: AgentKind.RANDOM,
        }
    )
    budget_kind: dict[Empire, str] = field(
        default_factory=lambda: {Empire.A: "Nodes", Empire.B: "Nodes"}
    )
    values: dict[str, str] = field(
        default_factory=lambda: {
            "seed_A": "0",
            "seed_B": "0",
            "depth_A": "1",
            "depth_B": "1",
            "max_A": "2",
            "max_B": "2",
            "budget_A": "2000",
            "budget_B": "2000",
            "delay": "500",
        }
    )
    active_field: str | None = None
    error: str | None = None

    def parse(self) -> SetupConfig:
        try:
            delay = int(self.values["delay"])
            settings: dict[Empire, AgentSettings] = {}
            for empire in Empire:
                suffix = empire.value
                kind = self.kinds[empire]
                seed = (
                    int(self.values[f"seed_{suffix}"])
                    if kind is AgentKind.RANDOM
                    else 0
                )
                depth = (
                    int(self.values[f"depth_{suffix}"])
                    if kind is AgentKind.MINIMAX
                    else 1
                )
                maximum = (
                    int(self.values[f"max_{suffix}"])
                    if kind is AgentKind.ITERATIVE
                    else 2
                )
                budget = (
                    int(self.values[f"budget_{suffix}"])
                    if kind is AgentKind.ITERATIVE
                    else 2_000
                )
                settings[empire] = AgentSettings(
                    kind=kind,
                    seed=seed,
                    depth=depth,
                    max_depth=maximum,
                    node_limit=(
                        budget
                        if kind is not AgentKind.ITERATIVE
                        or self.budget_kind[empire] == "Nodes"
                        else None
                    ),
                    time_limit_ms=(
                        budget
                        if kind is AgentKind.ITERATIVE
                        and self.budget_kind[empire] == "Time (ms)"
                        else None
                    ),
                )
            return SetupConfig(
                self.mode,
                self.first,
                self.human,
                settings[Empire.A],
                settings[Empire.B],
                delay,
            )
        except (ValueError, SetupValidationError) as error:
            raise SetupValidationError(f"Invalid setup: {error}") from error


class PlaygroundApp:
    """The event loop and view state; game rules remain in public engine APIs."""

    def __init__(self) -> None:
        if pygame is None:
            raise RuntimeError(
                "Pygame is required. Install the UI extra with "
                'pip install -e ".[dev,ui]"'
            )
        pygame.init()
        pygame.display.set_caption("NUBIA Local AI Playground")
        self.screen = pygame.display.set_mode(theme.WINDOW_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 40)
        self.setup = SetupScreen()
        self.controller: GameController | None = None
        self.buttons: list[Button] = []
        self.running = True
        self.agent_future: Future[Action] | None = None
        self.agent_snapshot: Any = None
        self.last_move_time = 0.0
        self.display_mode = DisplayMode.ICONS
        self.icon_store = PieceIconStore()

    def run(self, *, max_frames: int | None = None) -> int:
        frames = 0
        try:
            while self.running and (max_frames is None or frames < max_frames):
                self._events()
                self._update_agent()
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
                frames += 1
        finally:
            pygame.quit()
        return 0

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._key(event)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if any(button.handle(event.pos) for button in reversed(self.buttons)):
                    continue
                if self.controller is not None:
                    square = self._geometry().pixel_to_square(event.pos)
                    if square is not None:
                        self.controller.select_square(square)

    def _key(self, event: pygame.event.Event) -> None:
        if self.controller is not None:
            if event.key == pygame.K_ESCAPE:
                self.controller.clear_selection()
            return
        field_name = self.setup.active_field
        if field_name is None:
            return
        if event.key == pygame.K_BACKSPACE:
            self.setup.values[field_name] = self.setup.values[field_name][:-1]
        elif event.unicode and (event.unicode.isdigit() or event.unicode == "-"):
            self.setup.values[field_name] += event.unicode

    def _geometry(self) -> BoardGeometry:
        width, height = self.screen.get_size()
        size = max(48, min(66, (height - 130) // 10, (width - 390) // 10))
        return BoardGeometry(34, 62, size)

    def _draw(self) -> None:
        self.screen.fill(theme.BACKGROUND)
        self.buttons = []
        if self.controller is None:
            self._draw_setup()
        else:
            self._draw_game()

    def _button(
        self,
        label: str,
        rect: tuple[int, int, int, int],
        callback: Any,
        *,
        enabled: bool = True,
    ) -> None:
        button = Button(label, pygame.Rect(rect), callback, enabled)
        button.draw(self.screen, self.small_font)
        self.buttons.append(button)

    def _cycle_mode(self) -> None:
        modes = list(GameMode)
        self.setup.mode = modes[(modes.index(self.setup.mode) + 1) % len(modes)]

    def _cycle_kind(self, empire: Empire) -> None:
        kinds = list(AgentKind)
        current = self.setup.kinds[empire]
        self.setup.kinds[empire] = kinds[(kinds.index(current) + 1) % len(kinds)]

    def _draw_setup(self) -> None:
        self.screen.blit(
            self.title_font.render("NUBIA · Local AI Playground", True, theme.TEXT),
            (50, 34),
        )
        draw_lines(
            self.screen,
            [
                "A local inspection UI; the immutable engine remains authoritative.",
                "Click a value to change it. Click a number field, then type to edit.",
            ],
            (52, 82),
            self.small_font,
            color=theme.MUTED,
        )
        self._button(self.setup.mode.value, (52, 135, 240, 38), self._cycle_mode)
        self._button(
            f"First: Empire {self.setup.first.value}",
            (310, 135, 180, 38),
            lambda: setattr(self.setup, "first", self._other(self.setup.first)),
        )
        if self.setup.mode is GameMode.HUMAN_VS_AGENT:
            self._button(
                f"Human: Empire {self.setup.human.value}",
                (508, 135, 190, 38),
                lambda: setattr(self.setup, "human", self._other(self.setup.human)),
            )
        y = 210
        for empire in Empire:
            controlled = self.setup.mode is GameMode.AGENT_VS_AGENT or (
                self.setup.mode is GameMode.HUMAN_VS_AGENT
                and empire is not self.setup.human
            )
            if not controlled:
                continue
            self.screen.blit(
                self.font.render(f"Empire {empire.value} agent", True, theme.TEXT),
                (52, y),
            )
            self._button(
                self.setup.kinds[empire].value,
                (210, y - 6, 210, 34),
                lambda side=empire: self._cycle_kind(side),
            )
            kind = self.setup.kinds[empire]
            if kind is AgentKind.RANDOM:
                self._field("Seed", f"seed_{empire.value}", 448, y)
            elif kind is AgentKind.MINIMAX:
                self._field("Depth", f"depth_{empire.value}", 448, y)
            elif kind is AgentKind.ITERATIVE:
                self._field("Max depth", f"max_{empire.value}", 448, y)
                self._button(
                    self.setup.budget_kind[empire],
                    (680, y - 6, 105, 34),
                    lambda side=empire: self._toggle_budget(side),
                )
                self._field("Budget", f"budget_{empire.value}", 802, y)
            y += 62
        self._field("Auto delay (ms)", "delay", 52, max(y + 15, 370))
        self._button("Start game", (52, max(y + 75, 430), 190, 44), self._start)
        self._button("Quit", (260, max(y + 75, 430), 110, 44), self._quit)
        if self.setup.error:
            draw_lines(
                self.screen,
                [self.setup.error],
                (52, max(y + 135, 490)),
                self.small_font,
                color=theme.ERROR,
            )

    def _field(self, label: str, name: str, x: int, y: int) -> None:
        self.screen.blit(self.small_font.render(label, True, theme.MUTED), (x, y - 1))
        rect = (x + 98, y - 7, 92, 34)
        self._button(
            self.setup.values[name] or " ",
            rect,
            lambda: setattr(self.setup, "active_field", name),
        )
        if self.setup.active_field == name:
            pygame.draw.rect(
                self.screen, theme.SELECTED, pygame.Rect(rect), 2, border_radius=5
            )

    def _toggle_budget(self, empire: Empire) -> None:
        self.setup.budget_kind[empire] = (
            "Time (ms)" if self.setup.budget_kind[empire] == "Nodes" else "Nodes"
        )

    @staticmethod
    def _other(empire: Empire) -> Empire:
        return Empire.B if empire is Empire.A else Empire.A

    def _start(self) -> None:
        try:
            self.controller = GameController(self.setup.parse())
            self.setup.error = None
            self.last_move_time = time.monotonic()
        except SetupValidationError as error:
            self.setup.error = str(error)

    def _quit(self) -> None:
        self.running = False

    def _draw_game(self) -> None:
        controller = self.controller
        assert controller is not None
        geometry = self._geometry()
        draw_board(
            self.screen,
            geometry,
            controller,
            self.font,
            self.small_font,
            display_mode=self.display_mode,
            icon_store=self.icon_store,
        )
        panel_x = geometry.left + geometry.cell_size * 10 + 24
        panel_width = max(320, self.screen.get_width() - panel_x - 20)
        pygame.draw.rect(
            self.screen,
            theme.PANEL,
            (panel_x, 24, panel_width, self.screen.get_height() - 48),
            border_radius=8,
        )
        state = controller.state
        status = (
            "Thinking"
            if controller.thinking
            else "Paused"
            if controller.paused
            else "Ready"
        )
        y = draw_lines(
            self.screen,
            [
                f"Empire {state.side_to_move.value} to move · {status}",
                f"A: {controller.controller_name(Empire.A)}",
                f"B: {controller.controller_name(Empire.B)}",
                f"Ply {state.ply_number} · quiet {state.quiet_ply_count}",
                f"Last: {controller.latest_action}",
                render_result(state),
            ],
            (panel_x + 14, 38),
            self.small_font,
        )
        for empire in Empire:
            if empire in controller.agents:
                y = draw_lines(
                    self.screen,
                    [
                        f"{empire.value} config: "
                        f"{controller.settings_for(empire).description()}"
                    ],
                    (panel_x + 14, y),
                    self.small_font,
                    color=theme.MUTED,
                )
        y += 8
        if controller.error:
            y = draw_lines(
                self.screen,
                [controller.error],
                (panel_x + 14, y),
                self.small_font,
                color=theme.ERROR,
            )
        selected = (
            state.piece_at(controller.selected_source)
            if controller.selected_source is not None
            else None
        )
        if selected is not None:
            y = draw_lines(
                self.screen,
                [f"Piece: {PIECE_ASSETS[selected.piece_type].display_name}"],
                (panel_x + 14, y),
                self.small_font,
                color=theme.MUTED,
            )
        if self.icon_store.failures:
            failed = ", ".join(
                PIECE_ASSETS[piece_type].display_name
                for piece_type in self.icon_store.failures
            )
            y = draw_lines(
                self.screen,
                [f"Icon fallback: {failed}"],
                (panel_x + 14, y),
                self.small_font,
                color=theme.ERROR,
            )
        self._button("Restart", (panel_x + 14, y + 5, 88, 32), self._restart)
        self._button("Setup", (panel_x + 110, y + 5, 78, 32), self._to_setup)
        self._button("Quit", (panel_x + 196, y + 5, 64, 32), self._quit)
        y += 48
        self._button(
            f"Display: {self.display_mode.value}",
            (panel_x + 14, y, 166, 32),
            self._toggle_display_mode,
        )
        y += 44
        if controller.config.mode is GameMode.AGENT_VS_AGENT:
            self._button(
                "Resume" if controller.paused else "Pause",
                (panel_x + 14, y, 88, 32),
                controller.toggle_pause,
            )
            self._button(
                "Step",
                (panel_x + 110, y, 70, 32),
                self._step,
                enabled=controller.paused
                and not controller.thinking
                and state.result is None,
            )
            y += 44
        y = self._draw_action_choices(panel_x + 14, y, panel_width - 28)
        y += 8
        self.screen.blit(
            self.small_font.render("Move history", True, theme.MUTED), (panel_x + 14, y)
        )
        y += 20
        available = max(3, (self.screen.get_height() - y - 42) // 19)
        for index, record in list(enumerate(controller.history, start=1))[-available:]:
            text = self.small_font.render(
                f"{index}. {record.notation}", True, theme.TEXT
            )
            self.screen.blit(text, (panel_x + 14, y))
            y += 19

    def _draw_action_choices(self, x: int, y: int, width: int) -> int:
        controller = self.controller
        assert controller is not None
        choices = controller.action_choices
        if controller.selected_source is not None:
            self.screen.blit(
                self.small_font.render(
                    f"Selected {controller.selected_source} · "
                    f"legal actions {len(controller.selected_actions)}",
                    True,
                    theme.MUTED,
                ),
                (x, y),
            )
            y += 22
        for action in choices[:4]:
            label = self._action_label(action)
            self._button(
                label,
                (x, y, width, 30),
                lambda chosen=action: controller.submit_human_action(chosen),
            )
            y += 34
        return y

    def _action_label(self, action: Action) -> str:
        controller = self.controller
        assert controller is not None
        label = format_action(controller.state, action)
        if action.targets:
            targets = ", ".join(str(target.square) for target in action.targets)
            label += f" · targets: {targets}"
        if action.kind in (ActionKind.BRAINWASH, ActionKind.REBRAINWASH):
            successor = apply_action(controller.state, action)
            changed = successor.piece_at(action.destination)
            if changed is not None:
                label += f" · result: Empire {changed.current_empire.value}"
        return label

    def _restart(self) -> None:
        assert self.controller is not None
        self.agent_future = None
        self.controller.restart()
        self.last_move_time = time.monotonic()

    def _toggle_display_mode(self) -> None:
        self.display_mode = (
            DisplayMode.LETTERS
            if self.display_mode is DisplayMode.ICONS
            else DisplayMode.ICONS
        )

    def _to_setup(self) -> None:
        self.agent_future = None
        self.controller = None

    def _step(self) -> None:
        if self.controller is not None and self.controller.paused:
            self._request_agent(force=True)

    def _request_agent(self, *, force: bool = False) -> None:
        controller = self.controller
        if (
            controller is None
            or controller.thinking
            or controller.state.result is not None
        ):
            return
        if controller.paused and not force:
            return
        snapshot = controller.agent_snapshot()
        if snapshot is None:
            return
        state, agent = snapshot
        controller.thinking = True
        self.agent_snapshot = state
        future: Future[Action] = Future()

        def choose() -> None:
            try:
                future.set_result(agent.choose_action(state))
            except Exception as error:
                future.set_exception(error)

        self.agent_future = future
        Thread(target=choose, name="nubia-agent", daemon=True).start()

    def _update_agent(self) -> None:
        controller = self.controller
        if controller is None:
            return
        if self.agent_future is not None and self.agent_future.done():
            future = self.agent_future
            self.agent_future = None
            controller.thinking = False
            try:
                action = future.result()
            except Exception as error:
                controller.error = f"Agent error: {error}"
                controller.paused = True
            else:
                controller.apply_agent_choice(self.agent_snapshot, action)
                self.last_move_time = time.monotonic()
        if self.agent_future is None and controller.current_agent is not None:
            elapsed_ms = (time.monotonic() - self.last_move_time) * 1000
            if elapsed_ms >= controller.config.move_delay_ms:
                self._request_agent()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local NUBIA Pygame playground."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="initialize and cleanly close after two frames (for headless validation)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if pygame is None:
        print(
            'Pygame is required; install with: pip install -e ".[dev,ui]"',
            file=sys.stderr,
        )
        return 1
    return PlaygroundApp().run(max_frames=2 if args.smoke_test else None)
