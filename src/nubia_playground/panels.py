"""Small reusable Pygame panel widgets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame

from nubia_playground import theme


@dataclass(slots=True)
class Button:
    label: str
    rect: pygame.Rect
    callback: Callable[[], None]
    enabled: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        color = theme.BUTTON_HOVER if hovered and self.enabled else theme.BUTTON
        if not self.enabled:
            color = theme.PANEL_ALT
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        text = font.render(
            self.label, True, theme.TEXT if self.enabled else theme.MUTED
        )
        surface.blit(text, text.get_rect(center=self.rect.center))

    def handle(self, position: tuple[int, int]) -> bool:
        if self.enabled and self.rect.collidepoint(position):
            self.callback()
            return True
        return False


def draw_lines(
    surface: pygame.Surface,
    lines: list[str],
    position: tuple[int, int],
    font: pygame.font.Font,
    *,
    color: tuple[int, int, int] = theme.TEXT,
    spacing: int = 22,
) -> int:
    x, y = position
    for line in lines:
        surface.blit(font.render(line, True, color), (x, y))
        y += spacing
    return y
