from conftest import piece, state_with

from nubia_engine import (
    Action,
    ActionKind,
    Empire,
    GameState,
    PieceType,
    Square,
    legal_actions_from,
)


def _gbesele(state: GameState, source: Square) -> list[Action]:
    return [
        action
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.GBESELE
    ]


def test_one_gbesele_contains_all_row_column_and_diagonal_targets() -> None:
    source = Square(9, 4)
    targets = (Square(9, 1), Square(7, 4), Square(6, 1))
    state = state_with(
        [
            (source, piece("imperion", PieceType.IMPERION)),
            *[
                (square, piece(f"target-{index}", PieceType.PEASANT, Empire.B))
                for index, square in enumerate(targets)
            ],
        ]
    )
    actions = _gbesele(state, source)
    assert len(actions) == 1
    assert tuple(target.square for target in actions[0].targets) == tuple(
        sorted(targets)
    )
    assert actions[0].destination == source
    assert actions == _gbesele(state, source)


def test_only_first_occupied_square_on_each_ray_is_considered() -> None:
    source = Square(9, 4)
    state = state_with(
        [
            (source, piece("imperion", PieceType.IMPERION)),
            (Square(7, 4), piece("first", PieceType.PEASANT, Empire.B)),
            (Square(6, 4), piece("behind", PieceType.PEASANT, Empire.B)),
            (Square(9, 5), piece("friendly", PieceType.PEASANT)),
            (Square(9, 7), piece("hidden", PieceType.QUEEN, Empire.B)),
        ]
    )
    action = _gbesele(state, source)[0]
    assert [(target.square, target.piece_id) for target in action.targets] == [
        (Square(7, 4), "first")
    ]


def test_defended_first_enemy_still_blocks_piece_behind() -> None:
    source = Square(9, 4)
    state = state_with(
        [
            (source, piece("imperion", PieceType.IMPERION)),
            (Square(7, 4), piece("defended", PieceType.PEASANT, Empire.B)),
            (Square(6, 4), piece("behind", PieceType.PEASANT, Empire.B)),
            (Square(5, 3), piece("war", PieceType.NORTH_CENTRAL_WAR_CHIEF, Empire.B)),
        ]
    )
    assert _gbesele(state, source) == []


def test_outside_half_opposing_imperion_and_defended_targets_are_excluded() -> None:
    source = Square(9, 4)
    state = state_with(
        [
            (source, piece("imperion", PieceType.IMPERION)),
            (Square(4, 4), piece("outside", PieceType.PEASANT, Empire.B)),
            (Square(9, 1), piece("enemy-i", PieceType.IMPERION, Empire.B)),
            (Square(7, 2), piece("defended", PieceType.PEASANT, Empire.B)),
            (Square(6, 1), piece("defender", PieceType.PEASANT, Empire.B)),
        ]
    )
    assert _gbesele(state, source) == []


def test_converted_opponent_can_be_targeted_by_current_allegiance() -> None:
    source = Square(9, 4)
    converted = piece("converted", PieceType.QUEEN, Empire.B, original_empire=Empire.A)
    state = state_with(
        [(source, piece("imperion", PieceType.IMPERION)), (Square(7, 4), converted)]
    )
    assert _gbesele(state, source)[0].targets[0].piece_id == "converted"


def test_ordinary_palace_movement_remains_separate() -> None:
    source = Square(9, 4)
    state = state_with(
        [
            (source, piece("imperion", PieceType.IMPERION)),
            (Square(7, 4), piece("target", PieceType.PEASANT, Empire.B)),
        ]
    )
    actions = legal_actions_from(state, source)
    assert any(action.kind is ActionKind.GBESELE for action in actions)
    assert any(action.kind is ActionKind.MOVE for action in actions)
