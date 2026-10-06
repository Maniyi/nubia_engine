from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import piece, state_with

from nubia_ai import (
    HeuristicAgent,
    IterativeMinimaxAgent,
    MinimaxAgent,
    RandomAgent,
)
from nubia_engine import (
    ActionKind,
    Empire,
    GameResult,
    Outcome,
    PieceType,
    ResultReason,
    Square,
    apply_action,
    format_action,
    legal_actions,
    legal_actions_from,
)
from nubia_playground.controller import (
    AgentKind,
    AgentSettings,
    GameController,
    GameMode,
    SetupConfig,
    SetupValidationError,
    create_agent,
)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (AgentKind.RANDOM, RandomAgent),
        (AgentKind.HEURISTIC, HeuristicAgent),
        (AgentKind.MINIMAX, MinimaxAgent),
        (AgentKind.ITERATIVE, IterativeMinimaxAgent),
    ],
)
def test_creation_of_every_agent_type(kind: AgentKind, expected: type[object]) -> None:
    settings = AgentSettings(kind=kind)
    assert isinstance(create_agent(Empire.A, settings), expected)


def test_agent_descriptions_and_time_budget_configuration() -> None:
    assert "seed 4" in AgentSettings(seed=4).description()
    assert AgentSettings(kind=AgentKind.HEURISTIC).description() == "Heuristic (1 ply)"
    assert "depth 3" in AgentSettings(kind=AgentKind.MINIMAX, depth=3).description()
    timed = AgentSettings(
        kind=AgentKind.ITERATIVE,
        node_limit=None,
        time_limit_ms=25,
        max_depth=4,
    )
    assert timed.description() == "Iterative (max 4, 25 ms)"
    assert isinstance(create_agent(Empire.B, timed), IterativeMinimaxAgent)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"depth": 0},
        {"max_depth": -1},
        {"node_limit": 0},
        {"kind": AgentKind.ITERATIVE, "node_limit": None, "time_limit_ms": None},
        {"kind": AgentKind.ITERATIVE, "node_limit": 4, "time_limit_ms": 5},
    ],
)
def test_setup_validation_rejects_bad_numeric_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(SetupValidationError):
        AgentSettings(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(SetupValidationError, match="delay"):
        SetupConfig(move_delay_ms=0)


def test_setup_validation_rejects_wrong_domain_types() -> None:
    with pytest.raises(SetupValidationError, match="agent type"):
        AgentSettings(kind="Random")  # type: ignore[arg-type]
    with pytest.raises(SetupValidationError, match="mode"):
        SetupConfig(mode="play")  # type: ignore[arg-type]
    with pytest.raises(SetupValidationError, match="first"):
        SetupConfig(first_player="A")  # type: ignore[arg-type]
    with pytest.raises(SetupValidationError, match="human"):
        SetupConfig(human_empire="A")  # type: ignore[arg-type]
    with pytest.raises(SetupValidationError, match="settings"):
        SetupConfig(agent_a="random")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SetupConfig"):
        GameController(object())  # type: ignore[arg-type]


def test_human_versus_human_turn_flow_uses_exact_engine_action() -> None:
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    before = controller.state
    action = legal_actions(before)[0]
    assert controller.select_square(action.source)
    selected_action = next(
        choice
        for choice in controller.selected_actions
        if choice.destination == action.destination
    )
    assert controller.select_square(action.destination)
    assert controller.state == apply_action(before, action)
    assert controller.history[0].action is selected_action
    assert controller.history[0].notation == format_action(before, action)
    assert controller.is_human_turn


def test_human_versus_agent_turn_ownership() -> None:
    config = SetupConfig(
        mode=GameMode.HUMAN_VS_AGENT,
        first_player=Empire.B,
        human_empire=Empire.A,
    )
    controller = GameController(config)
    assert not controller.is_human_turn
    assert controller.current_agent is not None
    source = legal_actions(controller.state)[0].source
    assert not controller.select_square(source)
    assert controller.state.ply_number == 0
    assert controller.step_agent()
    assert controller.is_human_turn


def test_agent_versus_agent_pause_resume_and_single_step() -> None:
    controller = GameController(SetupConfig(mode=GameMode.AGENT_VS_AGENT))
    assert controller.paused
    controller.toggle_pause()
    assert not controller.paused
    controller.toggle_pause()
    assert controller.paused
    before = controller.state
    assert controller.step_agent()
    assert controller.state.ply_number == before.ply_number + 1
    assert controller.paused


def test_legal_selection_deselection_and_invalid_square_is_immutable() -> None:
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    action = legal_actions(controller.state)[0]
    assert controller.select_square(action.source)
    before = controller.state
    assert not controller.select_square(Square(4, 4))
    assert controller.state is before
    assert not controller.select_square(action.source)
    assert controller.selected_source is None


def test_clicking_another_active_piece_replaces_selection() -> None:
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    actions = legal_actions(controller.state)
    first = actions[0].source
    second = next(action.source for action in actions if action.source != first)
    assert controller.select_square(first)
    assert controller.select_square(second)
    assert controller.selected_source == second


def _special_state() -> object:
    return state_with(
        [
            (Square(4, 4), piece("m", PieceType.WEST_AFRICAN_MYSTIC)),
            (Square(2, 4), piece("q", PieceType.QUEEN, Empire.B)),
            (Square(7, 0), piece("ap", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )


def test_multiple_actions_same_destination_require_exact_special_choice() -> None:
    state = _special_state()
    controller = GameController(
        SetupConfig(mode=GameMode.HUMAN_VS_HUMAN),
        state=state,  # type: ignore[arg-type]
    )
    source = Square(4, 4)
    destination = Square(2, 4)
    candidates = tuple(
        action
        for action in legal_actions_from(controller.state, source)
        if action.destination == destination
    )
    assert {action.kind for action in candidates} == {
        ActionKind.CAPTURE,
        ActionKind.BRAINWASH,
    }
    assert controller.select_square(source)
    assert controller.select_square(destination)
    assert controller.state is state
    assert controller.action_choices == candidates
    brainwash = next(
        action
        for action in controller.action_choices
        if action.kind is ActionKind.BRAINWASH
    )
    assert controller.submit_human_action(brainwash)
    assert controller.history[-1].action is brainwash
    assert controller.state.piece_at(destination).current_empire is Empire.A  # type: ignore[union-attr]


def test_gbesele_choice_retains_complete_engine_target_set() -> None:
    state = state_with(
        [
            (Square(7, 4), piece("i", PieceType.IMPERION)),
            (Square(6, 4), piece("p1", PieceType.PEASANT, Empire.B)),
            (Square(7, 7), piece("p2", PieceType.PEASANT, Empire.B)),
            (Square(8, 0), piece("ap", PieceType.PEASANT)),
            (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
        ]
    )
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN), state=state)
    source = Square(7, 4)
    gbesele = next(
        action
        for action in legal_actions_from(state, source)
        if action.kind is ActionKind.GBESELE
    )
    assert controller.select_square(source)
    assert controller.action_choices == (gbesele,)
    assert len(gbesele.targets) == 2
    assert controller.submit_human_action(gbesele)
    assert controller.history[-1].action is gbesele


def test_stale_selection_is_rejected_after_state_changes() -> None:
    controller = GameController(SetupConfig(mode=GameMode.HUMAN_VS_HUMAN))
    selected = legal_actions(controller.state)[0]
    assert controller.select_square(selected.source)
    other = next(
        action
        for action in legal_actions(controller.state)
        if action.source != selected.source
    )
    controller.state = apply_action(controller.state, other)
    assert not controller.submit_human_action(selected)
    assert controller.error is not None and "stale" in controller.error


def test_terminal_state_prevents_actions_and_stops_play() -> None:
    terminal = replace(
        state_with(
            [
                (Square(7, 0), piece("ap", PieceType.PEASANT)),
                (Square(2, 9), piece("bp", PieceType.PEASANT, Empire.B)),
            ]
        ),
        result=GameResult(
            Outcome.DRAW,
            None,
            (ResultReason.THREEFOLD_REPETITION,),
        ),
    )
    controller = GameController(
        SetupConfig(mode=GameMode.AGENT_VS_AGENT), state=terminal
    )
    assert controller.agent_snapshot() is None
    assert not controller.step_agent()
    assert not controller.select_square(Square(7, 0))
    assert controller.state is terminal


def test_restart_is_fresh_with_same_configuration_and_history_cleared() -> None:
    config = SetupConfig(mode=GameMode.AGENT_VS_AGENT, first_player=Empire.B)
    controller = GameController(config)
    assert controller.step_agent()
    old = controller.state
    controller.restart()
    assert controller.config is config
    assert controller.state is not old
    assert controller.state.ply_number == 0
    assert controller.state.side_to_move is Empire.B
    assert controller.history == []
    assert controller.paused


def test_seeded_random_agents_reproduce_same_state_sequence() -> None:
    config = SetupConfig(
        mode=GameMode.AGENT_VS_AGENT,
        agent_a=AgentSettings(seed=17),
        agent_b=AgentSettings(seed=29),
    )
    first = GameController(config)
    second = GameController(config)
    for _ in range(8):
        assert first.step_agent() and second.step_agent()
    assert [record.action for record in first.history] == [
        record.action for record in second.history
    ]
    assert first.state == second.state


def test_agent_exception_pauses_without_fabricating_move() -> None:
    class BrokenAgent:
        empire = Empire.A
        name = "Broken"

        def choose_action(self, state: object) -> object:
            raise RuntimeError("boom")

    controller = GameController(SetupConfig(mode=GameMode.AGENT_VS_AGENT))
    controller.agents[Empire.A] = BrokenAgent()  # type: ignore[assignment]
    before = controller.state
    assert not controller.step_agent()
    assert controller.state is before
    assert controller.paused
    assert controller.error == "Agent error: boom"


def test_stale_and_illegal_agent_results_are_rejected() -> None:
    controller = GameController(SetupConfig(mode=GameMode.AGENT_VS_AGENT))
    old_state = controller.state
    old_action = legal_actions(old_state)[0]
    assert controller.step_agent()
    assert not controller.apply_agent_choice(old_state, old_action)

    current = controller.state
    assert not controller.apply_agent_choice(current, old_action)
    assert controller.paused
    assert controller.error is not None and "illegal" in controller.error


def test_human_submit_is_rejected_when_turn_belongs_to_agent() -> None:
    controller = GameController(
        SetupConfig(
            mode=GameMode.HUMAN_VS_AGENT,
            first_player=Empire.B,
            human_empire=Empire.A,
        )
    )
    action = legal_actions(controller.state)[0]
    assert not controller.submit_human_action(action)
    assert controller.error == "It is not a human turn."
    assert controller.controller_name(Empire.A) == "Human"
    assert controller.settings_for(Empire.B) is controller.config.agent_b
