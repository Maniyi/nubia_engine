from dataclasses import FrozenInstanceError

import pytest

from nubia_engine import ALL_SQUARES, Empire, RowKind, Square


@pytest.mark.parametrize(
    ("notation", "row", "column"),
    [
        ("A:p1", 9, 0),
        ("A:p10", 9, 9),
        ("A:b1", 5, 0),
        ("B:p1", 0, 9),
        ("B:p10", 0, 0),
        ("B:b1", 4, 9),
    ],
)
def test_required_coordinate_examples(notation: str, row: int, column: int) -> None:
    assert Square.from_notation(notation) == Square(row, column)


def test_all_squares_are_unique_and_round_trip() -> None:
    assert len(ALL_SQUARES) == 100
    assert len(set(ALL_SQUARES)) == 100
    assert all(
        Square.from_notation(square.to_notation()) == square for square in ALL_SQUARES
    )


def test_path_directions_are_opposite() -> None:
    assert Square.from_notation("A:p1").column == 0
    assert Square.from_notation("A:p10").column == 9
    assert Square.from_notation("B:p1").column == 9
    assert Square.from_notation("B:p10").column == 0


@pytest.mark.parametrize(
    ("row", "half", "kind"),
    [
        (0, Empire.B, RowKind.PALACE),
        (1, Empire.B, RowKind.MARKET),
        (2, Empire.B, RowKind.TOWN_CENTER),
        (3, Empire.B, RowKind.OUTSKIRTS),
        (4, Empire.B, RowKind.BORDER),
        (5, Empire.A, RowKind.BORDER),
        (6, Empire.A, RowKind.OUTSKIRTS),
        (7, Empire.A, RowKind.TOWN_CENTER),
        (8, Empire.A, RowKind.MARKET),
        (9, Empire.A, RowKind.PALACE),
    ],
)
def test_row_mapping(row: int, half: Empire, kind: RowKind) -> None:
    square = Square(row, 0)
    assert square.half is half
    assert square.row_kind is kind


@pytest.mark.parametrize(
    "notation",
    [
        "",
        "A:o",
        "A:o2x",
        "A-o2",
        "C:o2",
        "a:o2",
        "A:z2",
        "A:O2",
        "A:o0",
        "A:o11",
        "A:o-1",
        "A:o01",
        "A:o٢",
    ],
)
def test_invalid_notation_is_rejected(notation: str) -> None:
    with pytest.raises(ValueError):
        Square.from_notation(notation)


def test_non_string_notation_is_rejected() -> None:
    with pytest.raises(TypeError):
        Square.from_notation(2)  # type: ignore[arg-type]


@pytest.mark.parametrize(("row", "column"), [(-1, 0), (10, 0), (0, -1), (0, 10)])
def test_out_of_range_fixed_coordinates_are_rejected(row: int, column: int) -> None:
    with pytest.raises(ValueError):
        Square(row, column)


def test_non_integer_fixed_coordinates_are_rejected() -> None:
    with pytest.raises(TypeError):
        Square(True, 0)
    with pytest.raises(TypeError):
        Square(0, 1.5)  # type: ignore[arg-type]


def test_square_is_immutable() -> None:
    square = Square(0, 0)
    with pytest.raises(FrozenInstanceError):
        square.row = 1  # type: ignore[misc]
