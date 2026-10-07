from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nubia_engine import Empire, apply_action, create_initial_state, legal_actions
from nubia_playground.app import SetupScreen, build_parser, setup_from_args
from nubia_playground.controller import (
    AgentKind,
    AgentSettings,
    GameController,
    GameMode,
    SetupConfig,
    SetupValidationError,
    create_agent,
)


@pytest.fixture(scope="module")
def neural_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from nubia_training.neural import (
        ModelConfig,
        PolicyValueNetwork,
        TrainingConfig,
        save_checkpoint,
    )

    path = tmp_path_factory.mktemp("playground-neural") / "playground.pt"
    model = PolicyValueNetwork(
        ModelConfig(
            trunk_channels=8,
            residual_blocks=1,
            policy_embedding_dim=2,
            value_hidden_dim=8,
            normalization_groups=2,
        )
    )
    save_checkpoint(
        path,
        model,
        TrainingConfig(),
        dataset_id="playground-test",
        dataset_fingerprint="0" * 64,
        completed_epoch=0,
        optimizer_step=0,
        examples_seen=0,
        latest_metrics={},
    )
    return path


def test_agent_choices_include_neural_and_preserve_classical_choices() -> None:
    assert [kind.value for kind in AgentKind] == [
        "Random",
        "Heuristic",
        "Minimax",
        "Iterative Minimax",
        "Neural Policy",
        "Neural MCTS",
    ]


@pytest.mark.parametrize("kind", [AgentKind.NEURAL_POLICY, AgentKind.NEURAL_MCTS])
def test_neural_settings_require_checkpoint(kind: AgentKind) -> None:
    with pytest.raises(SetupValidationError, match="checkpoint"):
        create_agent(Empire.A, AgentSettings(kind=kind))


def test_playground_import_and_classical_creation_are_lazy() -> None:
    code = (
        "import sys; import nubia_playground; "
        "from nubia_engine import Empire; "
        "nubia_playground.create_agent(Empire.A, nubia_playground.AgentSettings()); "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("kind", [AgentKind.NEURAL_POLICY, AgentKind.NEURAL_MCTS])
@pytest.mark.parametrize("empire", list(Empire))
def test_neural_agents_return_legal_actions_for_both_sides(
    neural_checkpoint: Path, kind: AgentKind, empire: Empire
) -> None:
    settings = AgentSettings(
        kind=kind,
        checkpoint=str(neural_checkpoint),
        device="cpu",
        simulations=2,
    )
    agent = create_agent(empire, settings)
    state = create_initial_state(empire)
    before = state
    action = agent.choose_action(state)
    assert action in legal_actions(state)
    assert state is before
    if kind is AgentKind.NEURAL_POLICY:
        assert agent.temperature == 0.0  # type: ignore[attr-defined]
    else:
        assert agent.config.simulations == 2  # type: ignore[attr-defined]
        assert agent.config.temperature == 0.0  # type: ignore[attr-defined]
        assert not agent.config.root_noise_enabled  # type: ignore[attr-defined]
        assert agent.last_search_result is not None  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", [AgentKind.NEURAL_POLICY, AgentKind.NEURAL_MCTS])
def test_human_move_gets_exact_legal_neural_response_and_restart_reuses_model(
    neural_checkpoint: Path, kind: AgentKind
) -> None:
    settings = AgentSettings(
        kind=kind,
        checkpoint=str(neural_checkpoint),
        device="cpu",
        simulations=2,
    )
    controller = GameController(
        SetupConfig(
            mode=GameMode.HUMAN_VS_AGENT,
            human_empire=Empire.A,
            agent_b=settings,
        )
    )
    human_action = legal_actions(controller.state)[0]
    assert controller.select_square(human_action.source)
    assert controller.submit_human_action(human_action)
    snapshot = controller.state
    neural_action = controller.current_agent.choose_action(snapshot)  # type: ignore[union-attr]
    expected = apply_action(snapshot, neural_action)
    assert controller.apply_agent_choice(snapshot, neural_action)
    assert controller.state == expected
    assert controller.last_agent_diagnostics is not None
    loaded_agent = controller.agents[Empire.B]
    controller.restart()
    assert controller.agents[Empire.B] is loaded_agent
    assert controller.state.ply_number == 0


def test_bad_neural_checkpoint_does_not_create_or_mutate_game(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    settings = AgentSettings(
        kind=AgentKind.NEURAL_POLICY,
        checkpoint=str(missing),
    )
    with pytest.raises(SetupValidationError, match="does not exist"):
        GameController(SetupConfig(agent_b=settings))

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(SetupValidationError, match="sidecar"):
        create_agent(
            Empire.B,
            AgentSettings(
                kind=AgentKind.NEURAL_POLICY,
                checkpoint=str(corrupt),
            ),
        )


def test_cli_neural_configuration_and_invalid_combinations() -> None:
    parser = build_parser()
    policy = setup_from_args(
        parser,
        parser.parse_args(
            [
                "--opponent",
                "neural-policy",
                "--checkpoint",
                "model.pt",
                "--device",
                "cpu",
            ]
        ),
    )
    assert policy.kinds[Empire.B] is AgentKind.NEURAL_POLICY
    assert policy.checkpoint == "model.pt"
    assert policy.device == "cpu"

    mcts = setup_from_args(
        parser,
        parser.parse_args(
            [
                "--opponent",
                "neural-mcts",
                "--checkpoint",
                "model.pt",
                "--device",
                "cuda",
                "--simulations",
                "7",
            ]
        ),
    )
    assert mcts.kinds[Empire.B] is AgentKind.NEURAL_MCTS
    assert mcts.values["simulations"] == "7"

    with pytest.raises(SystemExit):
        setup_from_args(parser, parser.parse_args(["--opponent", "neural-mcts"]))
    with pytest.raises(SystemExit):
        setup_from_args(
            parser,
            parser.parse_args(["--opponent", "random", "--simulations", "2"]),
        )


def test_setup_screen_forwards_device_and_simulations() -> None:
    setup = SetupScreen(checkpoint="model.pt", device="cuda")
    setup.kinds[Empire.B] = AgentKind.NEURAL_MCTS
    setup.values["simulations"] = "9"
    settings = setup.parse().agent_b
    assert settings.checkpoint == "model.pt"
    assert settings.device == "cuda"
    assert settings.simulations == 9

    setup = SetupScreen(mode=GameMode.HUMAN_VS_HUMAN)
    setup.kinds[Empire.A] = AgentKind.NEURAL_POLICY
    assert GameController(setup.parse()).agents == {}


def test_requested_unavailable_cuda_is_a_focused_setup_error(
    neural_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SetupValidationError, match=r"CUDA.*unavailable"):
        create_agent(
            Empire.A,
            AgentSettings(
                kind=AgentKind.NEURAL_POLICY,
                checkpoint=str(neural_checkpoint),
                device="cuda",
            ),
        )


@pytest.mark.parametrize("kind", [AgentKind.NEURAL_POLICY, AgentKind.NEURAL_MCTS])
def test_cuda_inference_smoke_has_no_gradients_and_advances_state(
    neural_checkpoint: Path, kind: AgentKind
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agent = create_agent(
        Empire.A,
        AgentSettings(
            kind=kind,
            checkpoint=str(neural_checkpoint),
            device="cuda",
            simulations=1,
        ),
    )
    state = create_initial_state(Empire.A)
    action = agent.choose_action(state)
    assert action in legal_actions(state)
    assert apply_action(state, action).ply_number == 1
    model = agent.evaluator.model  # type: ignore[attr-defined]
    assert not model.training
    assert next(model.parameters()).device.type == "cuda"
    assert all(parameter.grad is None for parameter in model.parameters())
