from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_engine import (
    BrainwashAvailability,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    create_initial_state,
    render_board,
)


def test_board_has_canonical_rows_fixed_width_and_all_resources() -> None:
    rendered = render_board(create_initial_state(Empire.A))
    board_lines = [
        line
        for line in rendered.splitlines()
        if line[:3]
        in {"B:p", "B:m", "B:c", "B:o", "B:b", "A:b", "A:o", "A:c", "A:m", "A:p"}
    ]
    assert [line[:3] for line in board_lines] == [
        "B:p",
        "B:m",
        "B:c",
        "B:o",
        "B:b",
        "A:b",
        "A:o",
        "A:c",
        "A:m",
        "A:p",
    ]
    assert len({len(line) for line in board_lines}) == 1
    assert sum(line.count("R~") for line in board_lines) == 10
    assert "AP" in rendered and "BP" in rendered
    assert ".=LAND ~=SEA" in rendered
    assert render_board(create_initial_state(Empire.A)) == rendered


def test_board_shows_allegiance_conversion_mystic_power_and_counters() -> None:
    state = state_with(
        [
            (
                Square(4, 4),
                piece("converted", PieceType.QUEEN, Empire.B, original_empire=Empire.A),
            ),
            (Square(5, 5), piece("unused", PieceType.WEST_AFRICAN_MYSTIC)),
            (
                Square(5, 6),
                piece(
                    "spent",
                    PieceType.WEST_AFRICAN_MYSTIC,
                    brainwash=BrainwashAvailability.SPENT,
                ),
            ),
        ],
        Empire.B,
        quiet=7,
        ply=11,
    )
    rendered = render_board(state)
    assert "BQ*" in rendered
    assert "AM+" in rendered and "AM-" in rendered
    assert "Side to move: Empire B" in rendered
    assert "Ply: 11 | Quiet plies: 7" in rendered
    occupied_mine = state_with(
        [(Square.from_notation("A:o2"), piece("q", PieceType.QUEEN))]
    )
    assert "AQ^" in render_board(occupied_mine)


def test_board_renders_mine_scoring_and_simultaneous_draw_results() -> None:
    base = state_with([])
    mine = replace(
        base, result=GameResult(Outcome.WIN, Empire.A, (ResultReason.MINE_VICTORY,))
    )
    scoring = replace(
        base,
        result=GameResult(
            Outcome.DRAW, None, (ResultReason.NO_PEASANTS_SCORING,), (9, 9)
        ),
    )
    both = replace(
        base,
        result=GameResult(
            Outcome.DRAW,
            None,
            (ResultReason.THREEFOLD_REPETITION, ResultReason.NO_PROGRESS_40_PLIES),
        ),
    )
    assert "Empire A wins; reason(s): mine victory" in render_board(mine)
    assert "officer totals A=9, B=9" in render_board(scoring)
    assert "threefold repetition, no progress 40 plies" in render_board(both)


def test_renderers_validate_state() -> None:
    from nubia_engine import render_result

    with pytest.raises(TypeError):
        render_board("state")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        render_result("state")  # type: ignore[arg-type]
