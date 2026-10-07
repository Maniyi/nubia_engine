from __future__ import annotations

import numpy as np
import pytest

from nubia_engine import (
    Empire,
    GameResult,
    GameState,
    Outcome,
    Piece,
    PieceType,
    ResultReason,
    Square,
    create_initial_state,
    legal_actions,
)
from nubia_training import TrainingExample, legal_action_indices
from nubia_training.neural import (
    MCTSConfig,
    PositionEvaluation,
    puct_score,
    search_state,
)
from nubia_training.neural.errors import MCTSConfigurationError, MCTSSearchError


class UniformEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls = 0

    def evaluate(self, state: GameState) -> PositionEvaluation:
        self.calls += 1
        indices = legal_action_indices(state)
        return PositionEvaluation(
            indices,
            tuple(1.0 / len(indices) for _ in indices),
            self.value,
            state.side_to_move,
        )


def _one_action_state(*, winning: bool, quiet: int = 0) -> GameState:
    board: list[Piece | None] = [None] * 100
    source = Square.from_notation("B:b2" if winning else "A:b1")
    board[source.row * 10 + source.column] = Piece(
        "A-P", PieceType.PEASANT, Empire.A, Empire.A
    )
    adjacent_columns = (source.column - 1, source.column + 1)
    for ordinal, column in enumerate(adjacent_columns):
        if 0 <= column < 10:
            square = Square(source.row, column)
            board[square.row * 10 + square.column] = Piece(
                f"B-I-{ordinal}", PieceType.IMPERION, Empire.B, Empire.B
            )
    state = GameState(tuple(board), Empire.A, quiet_ply_count=quiet)
    assert len(legal_actions(state)) == 1
    return state


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"simulations": 0}, "simulations"),
        ({"c_puct": 0.0}, "c_puct"),
        ({"temperature": -1.0}, "temperature"),
        ({"max_depth": True}, "max_depth"),
        ({"root_noise_fraction": 2.0}, "fraction"),
        ({"root_noise_fraction": True}, "fraction"),
        ({"root_noise_enabled": True}, "seed"),
        (
            {
                "root_noise_enabled": True,
                "random_seed": 1,
                "root_dirichlet_alpha": 0.0,
            },
            "alpha",
        ),
        ({"root_noise_enabled": 1}, "Boolean"),
        ({"random_seed": True}, "random_seed"),
        ({"temperature": float("inf")}, "temperature"),
        ({"version": 2}, "version"),
    ],
)
def test_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(MCTSConfigurationError, match=message):
        MCTSConfig(**kwargs)  # type: ignore[arg-type]


def test_puct_formula() -> None:
    assert puct_score(
        mean_value=0.25,
        prior=0.5,
        parent_visits=16,
        child_visits=3,
        c_puct=2.0,
    ) == pytest.approx(1.25)


def test_search_accounting_policy_training_compatibility_and_repeatability() -> None:
    state = create_initial_state(Empire.A)
    config = MCTSConfig(simulations=12, max_depth=3)
    first = search_state(state, UniformEvaluator(), config)
    second = search_state(state, UniformEvaluator(), config)
    assert first == second
    assert first.root_visits == first.simulations == 12
    assert int(first.policy.visit_counts.sum()) == 12
    assert np.isclose(float(first.policy.probabilities.sum()), 1.0)
    assert first.action in legal_actions(state)
    assert first.root_children[0].visits > first.root_children[-1].visits
    example = first.policy.to_training_example(state, value_target=first.root_value)
    assert isinstance(example, TrainingExample)
    assert len(example.policy_action_indices) > 1


def test_root_noise_is_seeded_and_root_only() -> None:
    state = create_initial_state(Empire.A)
    one = search_state(
        state,
        UniformEvaluator(),
        MCTSConfig(simulations=4, root_noise_enabled=True, random_seed=9),
    )
    same = search_state(
        state,
        UniformEvaluator(),
        MCTSConfig(simulations=4, root_noise_enabled=True, random_seed=9),
    )
    other = search_state(
        state,
        UniformEvaluator(),
        MCTSConfig(simulations=4, root_noise_enabled=True, random_seed=10),
    )
    assert one == same
    assert tuple(child.prior for child in one.root_children) != tuple(
        child.prior for child in other.root_children
    )


def test_forced_win_backs_up_positive_and_never_evaluates_terminal_leaf() -> None:
    state = _one_action_state(winning=True)
    evaluator = UniformEvaluator(value=-0.75)
    result = search_state(state, evaluator, MCTSConfig(simulations=3, max_depth=4))
    assert result.root_children[0].mean_value == 1.0
    assert result.root_value == 1.0
    assert result.terminal_leaves == 3
    assert result.evaluator_calls == evaluator.calls == 1
    assert result.total_nodes_created == 2
    assert result.principal_variation == (legal_actions(state)[0],)


def test_quiet_draw_backs_up_zero() -> None:
    state = _one_action_state(winning=False, quiet=39)
    result = search_state(state, UniformEvaluator(1.0), MCTSConfig(simulations=2))
    assert result.root_children[0].mean_value == 0.0
    assert result.root_value == 0.0
    assert result.terminal_leaves == 2


def test_depth_guard_evaluates_without_fabricating_draw() -> None:
    state = create_initial_state(Empire.A)
    evaluator = UniformEvaluator(0.5)
    result = search_state(state, evaluator, MCTSConfig(simulations=3, max_depth=1))
    assert result.depth_guard_leaves == 3
    assert result.terminal_leaves == 0
    assert result.expanded_nodes == 1
    assert result.maximum_depth_reached == 1
    assert result.root_value == pytest.approx(-0.5)


def test_temperature_sampling_is_local_and_seeded() -> None:
    state = create_initial_state(Empire.A)
    first = search_state(
        state,
        UniformEvaluator(),
        MCTSConfig(simulations=40, temperature=1.0, random_seed=1),
    )
    repeat = search_state(
        state,
        UniformEvaluator(),
        MCTSConfig(simulations=40, temperature=1.0, random_seed=1),
    )
    other = search_state(
        state,
        UniformEvaluator(),
        MCTSConfig(simulations=40, temperature=1.0, random_seed=2),
    )
    assert first == repeat
    assert first.action != other.action


def test_terminal_root_and_invalid_evaluations_are_rejected() -> None:
    terminal = GameState(
        create_initial_state(Empire.A).board,
        Empire.A,
        result=GameResult(Outcome.DRAW, None, (ResultReason.THREEFOLD_REPETITION,)),
    )
    with pytest.raises(MCTSSearchError, match="terminal root"):
        search_state(terminal, UniformEvaluator(), MCTSConfig(simulations=1))

    class WrongPerspective(UniformEvaluator):
        def evaluate(self, state: GameState) -> PositionEvaluation:
            evaluation = super().evaluate(state)
            return PositionEvaluation(
                evaluation.action_indices,
                evaluation.priors,
                evaluation.value,
                Empire.B if state.side_to_move is Empire.A else Empire.A,
            )

    with pytest.raises(MCTSSearchError, match="perspective"):
        search_state(
            create_initial_state(Empire.A),
            WrongPerspective(),
            MCTSConfig(simulations=1),
        )

    class InvalidEvaluator:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def evaluate(self, state: GameState) -> object:
            indices = legal_action_indices(state)
            if self.mode == "type":
                return object()
            kwargs: dict[str, int] = {}
            if self.mode == "model":
                kwargs["model_architecture_version"] = 2
            elif self.mode == "checkpoint":
                kwargs["checkpoint_version"] = 2
            elif self.mode == "order":
                indices = tuple(reversed(indices))
            return PositionEvaluation(
                indices,
                tuple(1.0 / len(indices) for _ in indices),
                0.0,
                state.side_to_move,
                **kwargs,
            )

    for mode, message in (
        ("type", "return PositionEvaluation"),
        ("model", "model architecture"),
        ("checkpoint", "checkpoint version"),
        ("order", "engine action order"),
    ):
        with pytest.raises(MCTSSearchError, match=message):
            search_state(
                create_initial_state(Empire.A),
                InvalidEvaluator(mode),  # type: ignore[arg-type]
                MCTSConfig(simulations=1),
            )


def test_visit_policy_aliases_and_equality() -> None:
    result = search_state(
        create_initial_state(Empire.A),
        UniformEvaluator(),
        MCTSConfig(simulations=2),
    )
    assert result.selected_action == result.action
    assert result.visit_policy == result.policy
    assert result.selected_action_index == result.action_index
    assert result.simulation_count == result.root_visits == 2
    assert result.total_nodes_created == result.total_nodes
    assert result.model_calls == result.evaluator_calls
    assert result.root_child_statistics == result.root_children
    assert result.root_children[0].index == result.root_children[0].action_index
    assert (
        result.root_children[0].mean_action_value == result.root_children[0].mean_value
    )
    assert np.array_equal(
        result.policy.legal_action_indices, result.policy.action_indices
    )
    assert np.array_equal(
        result.policy.visit_probabilities, result.policy.probabilities
    )
    assert result.policy != object()
