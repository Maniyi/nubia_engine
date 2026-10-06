"""Fixed-orientation 10x10 Pygame board rendering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pygame

from nubia_engine import BrainwashAvailability, GameState, PieceType, Square
from nubia_playground import theme
from nubia_playground.controller import GameController
from nubia_playground.piece_assets import PieceIconStore

_SYMBOLS = {
    PieceType.IMPERION: "I",
    PieceType.QUEEN: "Q",
    PieceType.NORTH_CENTRAL_WAR_CHIEF: "W",
    PieceType.EAST_AFRICAN_HIGH_CHIEF: "C",
    PieceType.SOUTH_AFRICAN_ADVISOR: "A",
    PieceType.WEST_AFRICAN_MYSTIC: "M",
    PieceType.PEASANT: "P",
}

_PIECE_RADIUS_RATIO = 0.46
_ICON_SIZE_RATIO = 0.68


class DisplayMode(StrEnum):
    """Session-local piece rendering preferences."""

    ICONS = "Icons"
    LETTERS = "Letters"


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """Pure board/pixel coordinate conversion."""

    left: int
    top: int
    cell_size: int

    def square_rect(self, square: Square) -> tuple[int, int, int, int]:
        return (
            self.left + square.column * self.cell_size,
            self.top + square.row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def pixel_to_square(self, position: tuple[int, int]) -> Square | None:
        x, y = position
        column = (x - self.left) // self.cell_size
        row = (y - self.top) // self.cell_size
        if x < self.left or y < self.top or not (0 <= row < 10 and 0 <= column < 10):
            return None
        return Square(row, column)

    def square_center(self, square: Square) -> tuple[int, int]:
        x, y, width, height = self.square_rect(square)
        return x + width // 2, y + height // 2


def draw_board(
    surface: pygame.Surface,
    geometry: BoardGeometry,
    controller: GameController,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    *,
    display_mode: DisplayMode = DisplayMode.LETTERS,
    icon_store: PieceIconStore | None = None,
) -> None:
    """Draw the canonical fixed view: Empire B top and Empire A bottom."""

    state: GameState = controller.state
    destinations = {action.destination for action in controller.selected_actions}
    special_destinations = {
        action.destination for action in controller.selected_actions if action.targets
    }
    last_action = controller.history[-1].action if controller.history else None
    for row in range(10):
        for column in range(10):
            square = Square(row, column)
            rect = pygame.Rect(geometry.square_rect(square))
            pygame.draw.rect(
                surface,
                theme.LAND if square.terrain.value == "LAND" else theme.SEA,
                rect,
            )
            pygame.draw.rect(surface, (20, 25, 31), rect, 1)
            if last_action is not None and square in (
                last_action.source,
                last_action.destination,
            ):
                pygame.draw.rect(surface, theme.LAST_MOVE, rect, 3)
            if square.is_resource:
                pygame.draw.polygon(
                    surface,
                    theme.MINE,
                    [
                        (rect.centerx, rect.top + 5),
                        (rect.right - 5, rect.centery),
                        (rect.centerx, rect.bottom - 5),
                        (rect.left + 5, rect.centery),
                    ],
                    3,
                )
            piece = state.piece_at(square)
            if piece is not None:
                color = (
                    theme.EMPIRE_A
                    if piece.current_empire.value == "A"
                    else theme.EMPIRE_B
                )
                radius = round(rect.width * _PIECE_RADIUS_RATIO)
                pygame.draw.circle(surface, color, rect.center, radius)
                icon = None
                if display_mode is DisplayMode.ICONS and icon_store is not None:
                    icon_size = max(1, round(rect.width * _ICON_SIZE_RATIO))
                    icon = icon_store.scaled(
                        piece.piece_type, (icon_size, icon_size), color
                    )
                if icon is None:
                    label = font.render(_SYMBOLS[piece.piece_type], True, (12, 15, 19))
                    surface.blit(label, label.get_rect(center=rect.center))
                else:
                    assert icon_store is not None
                    try:
                        surface.blit(icon, icon.get_rect(center=rect.center))
                    except pygame.error as error:
                        icon_store.disable(
                            piece.piece_type,
                            f"could not render {piece.piece_type.value}: {error}",
                        )
                        label = font.render(
                            _SYMBOLS[piece.piece_type], True, (12, 15, 19)
                        )
                        surface.blit(label, label.get_rect(center=rect.center))
                if piece.is_converted:
                    pygame.draw.circle(surface, theme.SPECIAL, rect.center, radius, 3)
                    original = small_font.render(
                        piece.original_empire.value, True, theme.TEXT
                    )
                    surface.blit(original, (rect.left + 3, rect.top + 1))
                if piece.piece_type is PieceType.WEST_AFRICAN_MYSTIC:
                    power = piece.brainwash is BrainwashAvailability.AVAILABLE
                    pygame.draw.circle(
                        surface,
                        theme.SPECIAL if power else theme.MUTED,
                        (rect.right - 8, rect.top + 8),
                        5,
                    )
            if square in destinations:
                color = theme.SPECIAL if square in special_destinations else theme.LEGAL
                pygame.draw.circle(
                    surface, color, rect.center, max(5, rect.width // 9), 3
                )
            if square == controller.selected_source:
                pygame.draw.rect(surface, theme.SELECTED, rect, 4)

    caption = small_font.render(
        "Fixed view: B top · A bottom · columns follow Empire A paths 1-10",
        True,
        theme.MUTED,
    )
    surface.blit(caption, (geometry.left, geometry.top + geometry.cell_size * 10 + 6))
