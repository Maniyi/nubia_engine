"""Optional PyTorch policy-value network and bounded training tools."""

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
from nubia_training.neural.losses import (
    LossResult,
    mask_policy_logits,
    policy_probabilities,
    sparse_policy_value_loss,
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
    "MODEL_ARCHITECTURE_VERSION",
    "TRAINING_CONFIG_VERSION",
    "EpochSummary",
    "ExampleProvenance",
    "LoadedCheckpoint",
    "LossResult",
    "ModelConfig",
    "ModelOutput",
    "NeuralBatch",
    "NeuralDatasetItem",
    "PolicyValueNetwork",
    "ResidualBlock",
    "ShardDataset",
    "TrainingConfig",
    "TrainingResult",
    "TrainingSummary",
    "collate_examples",
    "load_checkpoint",
    "mask_policy_logits",
    "model_memory_report",
    "policy_probabilities",
    "save_checkpoint",
    "seed_training",
    "select_device",
    "sparse_policy_value_loss",
    "train_model",
    "trainable_parameter_count",
]
