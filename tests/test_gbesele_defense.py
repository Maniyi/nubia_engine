import pytest
from conftest import piece, state_with

from nubia_engine import Empire, PieceType, Square, is_defended_for_gbesele

TARGET = Square(5, 5)


@pytest.mark.parametrize(
    ("piece_type", "source"),
    [
        (PieceType.QUEEN, Square(5, 1)),
        (PieceType.SOUTH_AFRICAN_ADVISOR, Square(1, 5)),
        (PieceType.EAST_AFRICAN_HIGH_CHIEF, Square(2, 2)),
        (PieceType.NORTH_CENTRAL_WAR_CHIEF, Square(3, 4)),
        (PieceType.WEST_AFRICAN_MYSTIC, Square(3, 5)),
        (PieceType.PEASANT, Square(4, 4)),
    ],
)
def test_each_ordinary_capturer_can_defend(
    piece_type: PieceType, source: Square
) -> None:
    state = state_with(
        [
            (TARGET, piece("protected", PieceType.PEASANT, Empire.B)),
            (source, piece("defender", piece_type, Empire.B)),
        ]
    )
    assert is_defended_for_gbesele(state, TARGET, Empire.B)


def test_imperion_switch_and_target_itself_do_not_defend() -> None:
    state = state_with(
        [
            (TARGET, piece("target", PieceType.QUEEN, Empire.B)),
            (Square(5, 3), piece("imperion", PieceType.IMPERION, Empire.B)),
            (
                Square(5, 4),
                piece("chief", PieceType.EAST_AFRICAN_HIGH_CHIEF, Empire.B),
            ),
        ]
    )
    assert not is_defended_for_gbesele(state, TARGET, Empire.B)


def test_mystic_only_defends_at_capture_distance_with_clear_midpoint() -> None:
    near = state_with(
        [
            (TARGET, piece("target", PieceType.PEASANT, Empire.B)),
            (
                Square(4, 5),
                piece("mystic", PieceType.WEST_AFRICAN_MYSTIC, Empire.B),
            ),
        ]
    )
    blocked = state_with(
        [
            (TARGET, piece("target", PieceType.PEASANT, Empire.B)),
            (
                Square(3, 5),
                piece("mystic", PieceType.WEST_AFRICAN_MYSTIC, Empire.B),
            ),
            (Square(4, 5), piece("blocker", PieceType.PEASANT, Empire.A)),
        ]
    )
    assert not is_defended_for_gbesele(near, TARGET, Empire.B)
    assert not is_defended_for_gbesele(blocked, TARGET, Empire.B)


@pytest.mark.parametrize("blocker_empire", list(Empire))
def test_any_intervening_piece_blocks_sliding_defence(
    blocker_empire: Empire,
) -> None:
    state = state_with(
        [
            (TARGET, piece("target", PieceType.PEASANT, Empire.B)),
            (Square(5, 1), piece("queen", PieceType.QUEEN, Empire.B)),
            (Square(5, 3), piece("blocker", PieceType.PEASANT, blocker_empire)),
        ]
    )
    assert not is_defended_for_gbesele(state, TARGET, Empire.B)


def test_converted_defender_uses_current_allegiance() -> None:
    converted = piece(
        "converted",
        PieceType.QUEEN,
        Empire.B,
        original_empire=Empire.A,
    )
    state = state_with(
        [
            (TARGET, piece("target", PieceType.PEASANT, Empire.B)),
            (Square(5, 1), converted),
        ]
    )
    assert is_defended_for_gbesele(state, TARGET, Empire.B)
    assert not is_defended_for_gbesele(state, TARGET, Empire.A)


def test_peasant_defence_uses_current_allegiance_direction() -> None:
    converted = piece(
        "converted",
        PieceType.PEASANT,
        Empire.B,
        original_empire=Empire.A,
    )
    state = state_with(
        [
            (TARGET, piece("target", PieceType.QUEEN, Empire.B)),
            (Square(4, 4), converted),
        ]
    )
    assert is_defended_for_gbesele(state, TARGET, Empire.B)


def test_defence_is_deterministic_and_does_not_call_complete_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nubia_engine.move_generation as move_generation

    state = state_with(
        [
            (TARGET, piece("target", PieceType.PEASANT, Empire.B)),
            (Square(3, 4), piece("war", PieceType.NORTH_CENTRAL_WAR_CHIEF, Empire.B)),
        ]
    )

    def fail(*_args: object) -> None:
        raise AssertionError("complete generation must not be used")

    monkeypatch.setattr(move_generation, "legal_actions", fail)
    assert is_defended_for_gbesele(state, TARGET, Empire.B)
    assert is_defended_for_gbesele(state, TARGET, Empire.B)


def test_defence_public_arguments_are_validated() -> None:
    state = state_with([])
    with pytest.raises(TypeError):
        is_defended_for_gbesele("state", TARGET, Empire.B)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        is_defended_for_gbesele(state, "target", Empire.B)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        is_defended_for_gbesele(state, TARGET, "B")  # type: ignore[arg-type]
