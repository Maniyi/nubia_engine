from dataclasses import replace

from conftest import piece, state_with

from nubia_engine import (
    BrainwashAvailability,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    position_key,
)


def test_key_uses_rules_state_but_excludes_metadata_and_piece_ids() -> None:
    square = Square(4, 4)
    first = state_with([(square, piece("first", PieceType.PEASANT))])
    other_id = state_with([(square, piece("other", PieceType.PEASANT))])
    metadata = replace(
        first,
        quiet_ply_count=39,
        ply_number=80,
        result=GameResult(Outcome.DRAW, None, (ResultReason.NO_PROGRESS_40_PLIES,)),
        position_history=(position_key(first),),
    )
    assert position_key(first) == position_key(other_id) == position_key(metadata)
    assert hash(position_key(first)) == hash(position_key(other_id))


def test_key_changes_for_board_side_and_allegiance_state() -> None:
    square = Square(4, 4)
    base = state_with([(square, piece("p", PieceType.PEASANT))])
    moved = state_with([(Square(4, 5), piece("p", PieceType.PEASANT))])
    next_side = replace(base, side_to_move=Empire.B)
    converted = state_with(
        [
            (
                square,
                piece("p", PieceType.PEASANT, Empire.B, original_empire=Empire.A),
            )
        ]
    )
    different_origin = state_with(
        [(square, piece("p", PieceType.PEASANT, original_empire=Empire.B))]
    )
    assert (
        len(
            {
                position_key(base),
                position_key(moved),
                position_key(next_side),
                position_key(converted),
                position_key(different_origin),
            }
        )
        == 5
    )


def test_mystic_power_is_attached_to_square_not_identity() -> None:
    left, right = Square(4, 3), Square(4, 5)
    available = piece("available", PieceType.WEST_AFRICAN_MYSTIC)
    spent = piece(
        "spent",
        PieceType.WEST_AFRICAN_MYSTIC,
        brainwash=BrainwashAvailability.SPENT,
    )
    base = state_with([(left, available), (right, spent)])
    renamed = state_with(
        [
            (left, replace(available, id="x")),
            (right, replace(spent, id="y")),
        ]
    )
    exchanged = state_with([(left, spent), (right, available)])
    assert position_key(base) == position_key(renamed)
    assert position_key(base) != position_key(exchanged)
    assert tuple(item.square_index for item in position_key(base).occupied) == tuple(
        sorted(item.square_index for item in position_key(base).occupied)
    )
