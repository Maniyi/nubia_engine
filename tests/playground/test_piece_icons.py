from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest
from conftest import piece, state_with

from nubia_engine import Empire, PieceType, Square, legal_actions
from nubia_playground import theme
from nubia_playground.app import PlaygroundApp
from nubia_playground.board_view import BoardGeometry, DisplayMode, draw_board
from nubia_playground.controller import GameController, GameMode, SetupConfig
from nubia_playground.piece_assets import (
    PIECE_ASSETS,
    PieceIconStore,
    piece_asset_resource,
)


def _rgb(surface: pygame.Surface, position: tuple[int, int]) -> tuple[int, int, int]:
    color = surface.get_at(position)
    return color.r, color.g, color.b


@pytest.fixture
def app() -> Iterator[PlaygroundApp]:
    instance = PlaygroundApp()
    yield instance
    instance.running = False
    pygame.quit()


def test_mapping_covers_every_piece_type_with_seven_distinct_assets() -> None:
    assert set(PIECE_ASSETS) == set(PieceType)
    filenames = [asset.filename for asset in PIECE_ASSETS.values()]
    assert len(filenames) == len(set(filenames)) == 7


def test_both_empires_peasants_resolve_to_the_shared_visual() -> None:
    a_peasant = piece("ap", PieceType.PEASANT, Empire.A)
    b_peasant = piece("bp", PieceType.PEASANT, Empire.B)
    assert a_peasant.piece_type is b_peasant.piece_type is PieceType.PEASANT
    assert PIECE_ASSETS[a_peasant.piece_type].filename == "peasant.png"
    assert piece_asset_resource(a_peasant.piece_type) == piece_asset_resource(
        b_peasant.piece_type
    )


def test_every_packaged_icon_opens_and_contains_transparency() -> None:
    pygame.init()
    store = PieceIconStore()
    for piece_type in PieceType:
        resource = piece_asset_resource(piece_type)
        assert resource.is_file()
        image = store.source(piece_type)
        assert image is not None
        assert image.get_bitsize() == 32
        opaque = pygame.mask.from_surface(image, 254).count()
        assert opaque < image.get_width() * image.get_height()
    assert store.source_cache_size == 7


def test_resource_lookup_does_not_depend_on_process_working_directory(
    tmp_path: Path,
) -> None:
    code = """
from nubia_engine import PieceType
from nubia_playground.piece_assets import piece_asset_resource
for piece_type in PieceType:
    resource = piece_asset_resource(piece_type)
    assert resource.is_file()
    assert resource.read_bytes().startswith(b'\\x89PNG\\r\\n\\x1a\\n')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def test_icon_mode_is_default_and_visible_control_switches_without_game_mutation(
    app: PlaygroundApp,
) -> None:
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    app.controller = controller
    action = legal_actions(controller.state)[0]
    assert controller.select_square(action.source)
    before = (
        controller.state,
        controller.selected_source,
        controller.selected_actions,
        tuple(controller.history),
        controller.paused,
        controller.thinking,
    )

    assert app.display_mode is DisplayMode.ICONS
    app._draw()
    display_button = next(
        button for button in app.buttons if button.label == "Display: Icons"
    )
    assert display_button.handle(display_button.rect.center)
    assert app.display_mode.value == "Letters"
    assert before == (
        controller.state,
        controller.selected_source,
        controller.selected_actions,
        tuple(controller.history),
        controller.paused,
        controller.thinking,
    )

    app._draw()
    display_button = next(
        button for button in app.buttons if button.label == "Display: Letters"
    )
    assert display_button.handle(display_button.rect.center)
    assert app.display_mode is DisplayMode.ICONS
    assert before == (
        controller.state,
        controller.selected_source,
        controller.selected_actions,
        tuple(controller.history),
        controller.paused,
        controller.thinking,
    )


def test_scaled_icons_preserve_aspect_ratio_fit_circle_and_use_bounded_cache() -> None:
    pygame.init()
    store = PieceIconStore()
    source = store.source(PieceType.IMPERION)
    assert source is not None
    for bounds in ((30, 30), (42, 42), (54, 54), (30, 30)):
        icon = store.scaled(PieceType.IMPERION, bounds)
        assert icon is not None
        assert icon.get_width() <= bounds[0]
        assert icon.get_height() <= bounds[1]
        assert icon.get_width() / icon.get_height() == pytest.approx(
            source.get_width() / source.get_height(), rel=0.03
        )
        visible = icon.get_bounding_rect(min_alpha=1)
        circle_radius = bounds[0] / 0.68 * 0.46
        visible_corner_distance = max(
            ((x - icon.get_width() / 2) ** 2 + (y - icon.get_height() / 2) ** 2) ** 0.5
            for x in (visible.left, visible.right)
            for y in (visible.top, visible.bottom)
        )
        assert visible_corner_distance <= circle_radius
    assert store.source_cache_size == 1
    assert store.scaled_cache_size == 1


def test_source_and_scaled_surfaces_are_cached() -> None:
    pygame.init()
    store = PieceIconStore()
    assert store.source(PieceType.QUEEN) is store.source(PieceType.QUEEN)
    assert store.scaled(PieceType.QUEEN, (40, 40)) is store.scaled(
        PieceType.QUEEN, (40, 40)
    )
    assert store.source_cache_size == store.scaled_cache_size == 1


def test_icons_use_allegiance_color_with_a_black_outline() -> None:
    pygame.init()
    store = PieceIconStore()
    icon_a = store.scaled(PieceType.QUEEN, (45, 45), theme.EMPIRE_A)
    icon_b = store.scaled(PieceType.QUEEN, (45, 45), theme.EMPIRE_B)
    assert icon_a is not None and icon_b is not None
    colors_a = {
        _rgb(icon_a, (x, y))
        for x in range(icon_a.get_width())
        for y in range(icon_a.get_height())
    }
    colors_b = {
        _rgb(icon_b, (x, y))
        for x in range(icon_b.get_width())
        for y in range(icon_b.get_height())
    }
    assert theme.EMPIRE_A in colors_a
    assert theme.EMPIRE_B in colors_b
    assert (0, 0, 0) in colors_a & colors_b
    assert pygame.image.tobytes(icon_a, "RGBA") != pygame.image.tobytes(icon_b, "RGBA")
    assert store.scaled_cache_size == 2


def test_letter_mode_retains_every_original_piece_symbol() -> None:
    pygame.init()
    placements = [
        (Square(index, 0), piece(str(index), piece_type))
        for index, piece_type in enumerate(PieceType)
    ]
    controller = GameController(
        SetupConfig(mode=GameMode.HUMAN_VS_HUMAN), state=state_with(placements)
    )
    seen: list[str] = []
    real_font = pygame.font.Font(None, 26)

    class SpyFont:
        def render(
            self, text: str, antialias: bool, color: tuple[int, int, int]
        ) -> pygame.Surface:
            seen.append(text)
            return real_font.render(text, antialias, color)

    draw_board(
        pygame.Surface((700, 700)),
        BoardGeometry(10, 10, 60),
        controller,
        SpyFont(),  # type: ignore[arg-type]
        pygame.font.Font(None, 20),
        display_mode=DisplayMode.LETTERS,
    )
    assert set(seen) == {"I", "Q", "W", "C", "A", "M", "P"}


def test_last_move_highlight_remains_visible_in_icon_mode() -> None:
    pygame.init()
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    action = legal_actions(controller.state)[0]
    assert controller.select_square(action.source)
    assert controller.select_square(action.destination)
    geometry = BoardGeometry(10, 10, 60)
    surface = pygame.Surface((700, 700))
    draw_board(
        surface,
        geometry,
        controller,
        pygame.font.Font(None, 26),
        pygame.font.Font(None, 20),
        display_mode=DisplayMode.ICONS,
        icon_store=PieceIconStore(),
    )
    for square in (action.source, action.destination):
        rect = pygame.Rect(geometry.square_rect(square))
        assert _rgb(surface, (rect.left + 1, rect.top + 1)) == theme.LAST_MOVE


def test_unreadable_icon_warns_once_and_draws_correct_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pygame.init()
    pygame.display.set_mode((1, 1))
    real_load = pygame.image.load

    def broken_load(*args: object, **kwargs: object) -> pygame.Surface:
        raise pygame.error("broken icon")

    monkeypatch.setattr(pygame.image, "load", broken_load)
    store = PieceIconStore()
    with pytest.warns(RuntimeWarning, match="queen.png"):
        assert store.source(PieceType.QUEEN) is None
    assert store.source(PieceType.QUEEN) is None
    monkeypatch.setattr(pygame.image, "load", real_load)

    controller = GameController(
        SetupConfig(mode=GameMode.HUMAN_VS_HUMAN),
        state=state_with([(Square(4, 4), piece("q", PieceType.QUEEN))]),
    )
    seen: list[str] = []
    real_font = pygame.font.Font(None, 26)

    class SpyFont:
        def render(
            self, text: str, antialias: bool, color: tuple[int, int, int]
        ) -> pygame.Surface:
            seen.append(text)
            return real_font.render(text, antialias, color)

    draw_board(
        pygame.Surface((700, 700)),
        BoardGeometry(10, 10, 60),
        controller,
        SpyFont(),  # type: ignore[arg-type]
        pygame.font.Font(None, 20),
        display_mode=DisplayMode.ICONS,
        icon_store=store,
    )
    assert "Q" in seen
    assert PieceType.QUEEN in store.failures

    app = PlaygroundApp()
    try:
        app.controller = controller
        app.icon_store = store
        app._draw()
    finally:
        app.running = False


def test_scaling_and_rendering_errors_disable_only_the_failed_icon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pygame.init()
    pygame.display.set_mode((1, 1))
    store = PieceIconStore()
    assert store.source(PieceType.QUEEN) is not None

    def broken_scale(*args: object, **kwargs: object) -> pygame.Surface:
        raise pygame.error("cannot scale")

    monkeypatch.setattr(pygame.transform, "smoothscale", broken_scale)
    with pytest.warns(RuntimeWarning, match="could not scale"):
        assert store.scaled(PieceType.QUEEN, (40, 40)) is None
    store.disable(PieceType.QUEEN, "rendering also failed")
    assert store.source(PieceType.QUEEN) is None

    store = PieceIconStore()
    bad_icon = pygame.Surface((20, 20), pygame.SRCALPHA)
    monkeypatch.setattr(
        store, "scaled", lambda piece_type, bounds, fill_color=None: bad_icon
    )

    class FailingIconSurface(pygame.Surface):
        def blit(
            self,
            source: pygame.Surface,
            destination: Any,
            area: Any = None,
            special_flags: int = 0,
        ) -> pygame.Rect:
            if source is bad_icon:
                raise pygame.error("cannot render")
            return super().blit(source, destination, area, special_flags)

    controller = GameController(
        SetupConfig(mode=GameMode.HUMAN_VS_HUMAN),
        state=state_with([(Square(4, 4), piece("q", PieceType.QUEEN))]),
    )
    with pytest.warns(RuntimeWarning, match="could not render"):
        draw_board(
            FailingIconSurface((700, 700)),
            BoardGeometry(10, 10, 60),
            controller,
            pygame.font.Font(None, 26),
            pygame.font.Font(None, 20),
            display_mode=DisplayMode.ICONS,
            icon_store=store,
        )
    assert PieceType.QUEEN in store.failures


@pytest.mark.parametrize("display_mode", list(DisplayMode))
def test_allegiance_brainwash_mystic_selection_and_legal_overlays_remain_visible(
    display_mode: DisplayMode,
) -> None:
    pygame.init()
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    action = legal_actions(controller.state)[0]
    assert controller.select_square(action.source)
    geometry = BoardGeometry(10, 10, 60)
    surface = pygame.Surface((700, 700))
    draw_board(
        surface,
        geometry,
        controller,
        pygame.font.Font(None, 26),
        pygame.font.Font(None, 20),
        display_mode=display_mode,
        icon_store=PieceIconStore(),
    )
    selected_rect = pygame.Rect(geometry.square_rect(action.source))
    destination_rect = pygame.Rect(geometry.square_rect(action.destination))
    assert _rgb(surface, (selected_rect.left, selected_rect.top)) == theme.SELECTED
    assert any(
        _rgb(surface, (x, y)) == theme.LEGAL
        for x in range(destination_rect.left, destination_rect.right)
        for y in range(destination_rect.top, destination_rect.bottom)
    )

    special = state_with(
        [
            (
                Square(4, 4),
                piece(
                    "m",
                    PieceType.WEST_AFRICAN_MYSTIC,
                    Empire.B,
                    original_empire=Empire.A,
                ),
            )
        ]
    )
    controller = GameController(
        SetupConfig(mode=GameMode.HUMAN_VS_HUMAN), state=special
    )
    surface.fill((0, 0, 0))
    draw_board(
        surface,
        geometry,
        controller,
        pygame.font.Font(None, 26),
        pygame.font.Font(None, 20),
        display_mode=display_mode,
        icon_store=PieceIconStore(),
    )
    rect = pygame.Rect(geometry.square_rect(Square(4, 4)))
    radius = round(rect.width * 0.46)
    allegiance_pixel = _rgb(surface, (rect.centerx + radius - 4, rect.centery))
    assert allegiance_pixel == theme.EMPIRE_B
    assert any(
        _rgb(surface, (x, y)) == theme.SPECIAL
        for x in range(rect.centerx - radius - 2, rect.centerx + 1)
        for y in range(rect.centery - radius - 2, rect.centery + radius + 3)
    )
    assert _rgb(surface, (rect.right - 8, rect.top + 8)) == theme.SPECIAL
