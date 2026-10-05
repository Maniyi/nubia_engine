from nubia_engine import ALL_SQUARES, Square, Terrain


def test_terrain_is_evenly_split() -> None:
    assert sum(square.terrain is Terrain.LAND for square in ALL_SQUARES) == 50
    assert sum(square.terrain is Terrain.SEA for square in ALL_SQUARES) == 50


def test_each_empires_palace_path_one_is_sea() -> None:
    assert Square.from_notation("A:p1").terrain is Terrain.SEA
    assert Square.from_notation("B:p1").terrain is Terrain.SEA


def test_horizontal_neighbors_alternate() -> None:
    for row in range(10):
        for column in range(9):
            assert Square(row, column).terrain is not Square(row, column + 1).terrain


def test_vertical_neighbors_alternate_including_border_boundary() -> None:
    for row in range(9):
        for column in range(10):
            assert Square(row, column).terrain is not Square(row + 1, column).terrain
    for column in range(10):
        assert Square(4, column).terrain is not Square(5, column).terrain
