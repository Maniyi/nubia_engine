from nubia_engine import RESOURCE_SQUARES, Empire, Square


def test_resource_square_count_and_uniqueness() -> None:
    assert len(RESOURCE_SQUARES) == 10
    assert len(set(RESOURCE_SQUARES)) == 10


def test_each_half_has_five_resource_squares() -> None:
    assert sum(square.half is Empire.A for square in RESOURCE_SQUARES) == 5
    assert sum(square.half is Empire.B for square in RESOURCE_SQUARES) == 5


def test_exact_resource_notations() -> None:
    assert {square.to_notation() for square in RESOURCE_SQUARES} == {
        "A:o2",
        "A:o4",
        "A:o6",
        "A:o8",
        "A:o10",
        "B:o2",
        "B:o4",
        "B:o6",
        "B:o8",
        "B:o10",
    }


def test_b_resource_columns_use_reversed_path_orientation() -> None:
    assert [Square.from_notation(f"B:o{path}").column for path in range(2, 11, 2)] == [
        8,
        6,
        4,
        2,
        0,
    ]


def test_resource_accessor() -> None:
    assert Square.from_notation("A:o2").is_resource
    assert not Square.from_notation("A:o1").is_resource
    assert not Square.from_notation("A:c2").is_resource
