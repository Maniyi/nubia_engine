"""Canonical fixed-board coordinates and square properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from nubia_engine.enums import Empire, RowKind, Terrain

BOARD_SIZE = 10

_ROW_DETAILS: tuple[tuple[Empire, str, RowKind], ...] = (
    (Empire.B, "p", RowKind.PALACE),
    (Empire.B, "m", RowKind.MARKET),
    (Empire.B, "c", RowKind.TOWN_CENTER),
    (Empire.B, "o", RowKind.OUTSKIRTS),
    (Empire.B, "b", RowKind.BORDER),
    (Empire.A, "b", RowKind.BORDER),
    (Empire.A, "o", RowKind.OUTSKIRTS),
    (Empire.A, "c", RowKind.TOWN_CENTER),
    (Empire.A, "m", RowKind.MARKET),
    (Empire.A, "p", RowKind.PALACE),
)

_ROW_BY_OWNER_AND_SYMBOL = {
    (empire, symbol): row
    for row, (empire, symbol, _row_kind) in enumerate(_ROW_DETAILS)
}


@dataclass(frozen=True, slots=True, order=True)
class Square:
    """A square in fixed-board coordinates, with rows and columns numbered 0-9."""

    row: int
    column: int
    _NOTATION_ERROR: ClassVar[str] = "notation must look like A:o2 or B:p10"

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or not isinstance(self.row, int):
            raise TypeError("row must be an integer")
        if isinstance(self.column, bool) or not isinstance(self.column, int):
            raise TypeError("column must be an integer")
        if not 0 <= self.row < BOARD_SIZE:
            raise ValueError("row must be between 0 and 9")
        if not 0 <= self.column < BOARD_SIZE:
            raise ValueError("column must be between 0 and 9")

    @classmethod
    def from_notation(cls, notation: str) -> Square:
        """Convert owner-relative notation such as ``A:o2`` to a square."""

        if not isinstance(notation, str):
            raise TypeError("notation must be a string")
        if len(notation) not in (4, 5) or notation[1:2] != ":":
            raise ValueError(cls._NOTATION_ERROR)

        try:
            empire = Empire(notation[0])
        except ValueError as error:
            raise ValueError(
                f"invalid empire in {notation!r}; expected A or B"
            ) from error

        row_symbol = notation[2]
        try:
            row = _ROW_BY_OWNER_AND_SYMBOL[(empire, row_symbol)]
        except KeyError as error:
            raise ValueError(
                f"invalid row symbol in {notation!r}; expected p, m, c, o, or b"
            ) from error

        path_text = notation[3:]
        if not path_text.isascii() or not path_text.isdecimal():
            raise ValueError(f"invalid path in {notation!r}; expected 1 through 10")
        if len(path_text) > 1 and path_text.startswith("0"):
            raise ValueError(f"invalid path in {notation!r}; expected canonical digits")
        path = int(path_text)
        if not 1 <= path <= BOARD_SIZE:
            raise ValueError(f"path must be between 1 and 10 in {notation!r}")

        column = path - 1 if empire is Empire.A else BOARD_SIZE - path
        return cls(row=row, column=column)

    @property
    def half(self) -> Empire:
        """The empire whose half contains this square."""

        return _ROW_DETAILS[self.row][0]

    @property
    def row_kind(self) -> RowKind:
        """The owner-relative kind of this square's row."""

        return _ROW_DETAILS[self.row][2]

    @property
    def path(self) -> int:
        """The path number from the viewpoint of this square's half."""

        return self.column + 1 if self.half is Empire.A else BOARD_SIZE - self.column

    @property
    def terrain(self) -> Terrain:
        """The square's alternating LAND or SEA terrain."""

        return Terrain.SEA if (self.row + self.column) % 2 == 1 else Terrain.LAND

    @property
    def is_resource(self) -> bool:
        """Whether this square contains a natural-resource mine."""

        return self.row_kind is RowKind.OUTSKIRTS and self.path % 2 == 0

    def to_notation(self) -> str:
        """Return this square's unique owner-relative notation."""

        empire, row_symbol, _row_kind = _ROW_DETAILS[self.row]
        return f"{empire.value}:{row_symbol}{self.path}"

    def __str__(self) -> str:
        return self.to_notation()


ALL_SQUARES: tuple[Square, ...] = tuple(
    Square(row, column) for row in range(BOARD_SIZE) for column in range(BOARD_SIZE)
)

RESOURCE_SQUARES: tuple[Square, ...] = tuple(
    square for square in ALL_SQUARES if square.is_resource
)
