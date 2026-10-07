"""Optional PyTorch policy-value network and bounded training tools."""

from nubia_training.neural.agents import MCTSAgent, NeuralPolicyAgent
from nubia_training.neural.checkpoints import (
    CHECKPOINT_VERSION,
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from nubia_training.neural.data import (
    ExampleProvenance,
    NeuralBatch,
    NeuralDatasetItem,
    ShardDataset,
    collate_examples,
)
from nubia_training.neural.device import select_device
from nubia_training.neural.evaluator import (
    PRIOR_SUM_TOLERANCE,
    CheckpointPositionEvaluator,
    NeuralPositionEvaluator,
    PositionEvaluation,
    PositionEvaluator,
    PyTorchPositionEvaluator,
)
from nubia_training.neural.losses import (
    LossResult,
    mask_policy_logits,
    policy_probabilities,
    sparse_policy_value_loss,
)
from nubia_training.neural.mcts import (
    MCTS_VERSION,
    MCTSConfig,
    MCTSSearchResult,
    RootChildStatistics,
    RootVisitPolicy,
    puct_score,
    run_mcts,
    search_state,
)
from nubia_training.neural.model import (
    MODEL_ARCHITECTURE_VERSION,
    ModelConfig,
    ModelOutput,
    PolicyValueNetwork,
    ResidualBlock,
    model_memory_report,
    trainable_parameter_count,
)
from nubia_training.neural.training import (
    TRAINING_CONFIG_VERSION,
    EpochSummary,
    TrainingConfig,
    TrainingResult,
    TrainingSummary,
    seed_training,
    train_model,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "MCTS_VERSION",
    "MODEL_ARCHITECTURE_VERSION",
    "PRIOR_SUM_TOLERANCE",
    "TRAINING_CONFIG_VERSION",
    "CheckpointPositionEvaluator",
    "EpochSummary",
    "ExampleProvenance",
    "LoadedCheckpoint",
    "LossResult",
    "MCTSAgent",
    "MCTSConfig",
    "MCTSSearchResult",
    "ModelConfig",
    "ModelOutput",
    "NeuralBatch",
    "NeuralDatasetItem",
    "NeuralPolicyAgent",
    "NeuralPositionEvaluator",
    "PolicyValueNetwork",
    "PositionEvaluation",
    "PositionEvaluator",
    "PyTorchPositionEvaluator",
    "ResidualBlock",
    "RootChildStatistics",
    "RootVisitPolicy",
    "ShardDataset",
    "TrainingConfig",
    "TrainingResult",
    "TrainingSummary",
    "collate_examples",
    "load_checkpoint",
    "mask_policy_logits",
    "model_memory_report",
    "policy_probabilities",
    "puct_score",
    "run_mcts",
    "save_checkpoint",
    "search_state",
    "seed_training",
    "select_device",
    "sparse_policy_value_loss",
    "train_model",
    "trainable_parameter_count",
]
