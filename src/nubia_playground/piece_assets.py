"""Package-resource-aware piece icon metadata and loading."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Final

import pygame

from nubia_engine import PieceType


@dataclass(frozen=True, slots=True)
class PieceAsset:
    """The presentation metadata for one engine piece type."""

    filename: str
    display_name: str


PIECE_ASSETS: Final = MappingProxyType(
    {
        PieceType.IMPERION: PieceAsset("Imperion.png", "Imperion"),
        PieceType.QUEEN: PieceAsset("queen.png", "Queen"),
        PieceType.NORTH_CENTRAL_WAR_CHIEF: PieceAsset(
            "warchief.png", "North-Central War Chief"
        ),
        PieceType.EAST_AFRICAN_HIGH_CHIEF: PieceAsset(
            "high_chief.png", "East African High Chief"
        ),
        PieceType.SOUTH_AFRICAN_ADVISOR: PieceAsset(
            "advisor.png", "South African Advisor"
        ),
        PieceType.WEST_AFRICAN_MYSTIC: PieceAsset("mystic.png", "West African Mystic"),
        PieceType.PEASANT: PieceAsset("peasant.png", "Peasant"),
    }
)


def piece_asset_resource(piece_type: PieceType) -> resources.abc.Traversable:
    """Return the package resource for ``piece_type`` without using the CWD."""

    asset = PIECE_ASSETS[piece_type]
    return resources.files("nubia_playground").joinpath(
        "assets", "pieces", asset.filename
    )


class PieceIconStore:
    """Load each source once and retain at most one scaled size per piece type."""

    def __init__(self) -> None:
        self._sources: dict[PieceType, pygame.Surface | None] = {}
        self._scaled: dict[
            tuple[PieceType, tuple[int, int, int] | None],
            tuple[tuple[int, int], pygame.Surface],
        ] = {}
        self.failures: dict[PieceType, str] = {}

    def source(self, piece_type: PieceType) -> pygame.Surface | None:
        """Return the cached source image, recording a diagnosable load failure."""

        if piece_type in self._sources:
            return self._sources[piece_type]
        asset = PIECE_ASSETS[piece_type]
        resource = piece_asset_resource(piece_type)
        try:
            with resource.open("rb") as stream:
                image = pygame.image.load(stream, asset.filename)
        except (OSError, pygame.error) as error:
            self._record_failure(
                piece_type, f"could not load {asset.filename}: {error}"
            )
            image = None
        self._sources[piece_type] = image
        return image

    def scaled(
        self,
        piece_type: PieceType,
        bounds: tuple[int, int],
        fill_color: tuple[int, int, int] | None = None,
    ) -> pygame.Surface | None:
        """Return an aspect-preserving, optionally allegiance-colored icon."""

        cache_key = piece_type, fill_color
        cached = self._scaled.get(cache_key)
        if cached is not None and cached[0] == bounds:
            return cached[1]
        source = self.source(piece_type)
        if source is None:
            return None
        width, height = source.get_size()
        scale = min(bounds[0] / width, bounds[1] / height)
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        try:
            icon = pygame.transform.smoothscale(source, scaled_size)
            if fill_color is not None:
                icon = self._color_with_outline(icon, fill_color)
        except pygame.error as error:
            self._record_failure(
                piece_type,
                f"could not scale {PIECE_ASSETS[piece_type].filename}: {error}",
            )
            return None
        self._scaled[cache_key] = (bounds, icon)
        return icon

    @staticmethod
    def _color_with_outline(
        icon: pygame.Surface, fill_color: tuple[int, int, int]
    ) -> pygame.Surface:
        """Color broad interiors while preserving fine black artwork details."""

        mask = pygame.mask.from_surface(icon, threshold=8)
        outline = mask.copy()
        outline_width = max(1, round(min(icon.get_size()) * 0.035))
        for x_offset in range(-outline_width, outline_width + 1):
            for y_offset in range(-outline_width, outline_width + 1):
                if (
                    x_offset * x_offset + y_offset * y_offset
                    <= outline_width * outline_width
                ):
                    outline.draw(mask, (x_offset, y_offset))
        colored = outline.to_surface(setcolor=(0, 0, 0, 255), unsetcolor=(0, 0, 0, 0))
        interior = PieceIconStore._eroded_mask(mask)
        fill = interior.to_surface(setcolor=(*fill_color, 255), unsetcolor=(0, 0, 0, 0))
        colored.blit(fill, (0, 0))
        return colored

    @staticmethod
    def _eroded_mask(mask: pygame.Mask) -> pygame.Mask:
        """Keep only pixels surrounded by artwork, leaving thin details black."""

        width, height = mask.get_size()
        interior = pygame.Mask((width, height))
        neighbors = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if all(mask.get_at((x + dx, y + dy)) for dx, dy in neighbors):
                    interior.set_at((x, y))
        return interior

    @property
    def source_cache_size(self) -> int:
        """Number of attempted source loads, including cached failures."""

        return len(self._sources)

    @property
    def scaled_cache_size(self) -> int:
        """Retained surfaces: at most one size per piece type and color."""

        return len(self._scaled)

    def disable(self, piece_type: PieceType, message: str) -> None:
        """Disable a runtime-unrenderable icon and retain its diagnostic."""

        self._sources[piece_type] = None
        for cache_key in tuple(self._scaled):
            if cache_key[0] is piece_type:
                del self._scaled[cache_key]
        self._record_failure(piece_type, message)

    def _record_failure(self, piece_type: PieceType, message: str) -> None:
        if piece_type not in self.failures:
            self.failures[piece_type] = message
            warnings.warn(message, RuntimeWarning, stacklevel=2)
