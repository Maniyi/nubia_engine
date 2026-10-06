"""Public machine-learning representation foundation for NUBIA."""

from nubia_training.action_space import (
    ACTION_KIND_BLOCK_SIZE,
    ACTION_KIND_OFFSETS,
    ACTION_KIND_ORDER,
    ACTION_SPACE_SIZE,
    ACTION_SPACE_SPEC,
    ACTION_SPACE_VERSION,
    ActionSpaceSpec,
    action_to_index,
    index_to_action,
    legal_action_indices,
    legal_action_mask,
)
from nubia_training.datasets import (
    DATASET_MANIFEST_VERSION,
    DatasetManifest,
    build_dataset,
    inspect_dataset,
    load_dataset,
    read_manifest,
    validate_dataset,
)
from nubia_training.encoding import (
    GLOBAL_FEATURE_NAMES,
    PIECE_TYPE_ORDER,
    SPATIAL_CHANNEL_NAMES,
    STATE_ENCODING_SPEC,
    STATE_ENCODING_VERSION,
    EncodedState,
    StateEncodingSpec,
    encode_state,
    encoded_states_equal,
    immediate_repetition_consequences,
)
from nubia_training.errors import (
    ActionIndexCollisionError,
    AgentSpecError,
    DatasetValidationError,
    IllegalOrStaleActionError,
    InvalidActionIndexError,
    InvalidEncodingError,
    InvalidPolicyTargetError,
    RecordIntegrityError,
    RecordValidationError,
    ReplayValidationError,
    RepresentationError,
    SchemaVersionError,
    TrainingDataError,
    UnresolvedActionIndexError,
    UnsupportedActionKindError,
    UnsupportedPieceTypeError,
    VersionMismatchError,
)
from nubia_training.examples import (
    POLICY_SUM_TOLERANCE,
    TRAINING_EXAMPLE_VERSION,
    TrainingExample,
)
from nubia_training.records import (
    AGENT_SPEC_VERSION,
    RAW_GAME_RECORD_VERSION,
    STANDARD_SETUP_VERSION,
    AgentSpec,
    PlyRecord,
    RawGameRecord,
    read_raw_game,
    write_raw_game,
)
from nubia_training.replay import (
    ProvenancedExample,
    ReplayReport,
    game_to_examples,
    replay_game,
)
from nubia_training.storage import DATASET_SHARD_VERSION

__all__ = [
    "ACTION_KIND_BLOCK_SIZE",
    "ACTION_KIND_OFFSETS",
    "ACTION_KIND_ORDER",
    "ACTION_SPACE_SIZE",
    "ACTION_SPACE_SPEC",
    "ACTION_SPACE_VERSION",
    "AGENT_SPEC_VERSION",
    "DATASET_MANIFEST_VERSION",
    "DATASET_SHARD_VERSION",
    "GLOBAL_FEATURE_NAMES",
    "PIECE_TYPE_ORDER",
    "POLICY_SUM_TOLERANCE",
    "RAW_GAME_RECORD_VERSION",
    "SPATIAL_CHANNEL_NAMES",
    "STANDARD_SETUP_VERSION",
    "STATE_ENCODING_SPEC",
    "STATE_ENCODING_VERSION",
    "TRAINING_EXAMPLE_VERSION",
    "ActionIndexCollisionError",
    "ActionSpaceSpec",
    "AgentSpec",
    "AgentSpecError",
    "DatasetManifest",
    "DatasetValidationError",
    "EncodedState",
    "IllegalOrStaleActionError",
    "InvalidActionIndexError",
    "InvalidEncodingError",
    "InvalidPolicyTargetError",
    "PlyRecord",
    "ProvenancedExample",
    "RawGameRecord",
    "RecordIntegrityError",
    "RecordValidationError",
    "ReplayReport",
    "ReplayValidationError",
    "RepresentationError",
    "SchemaVersionError",
    "StateEncodingSpec",
    "TrainingDataError",
    "TrainingExample",
    "UnresolvedActionIndexError",
    "UnsupportedActionKindError",
    "UnsupportedPieceTypeError",
    "VersionMismatchError",
    "action_to_index",
    "agent_spec",
    "build_dataset",
    "create_agent",
    "encode_state",
    "encoded_states_equal",
    "game_to_examples",
    "generate_game",
    "generate_series",
    "immediate_repetition_consequences",
    "index_to_action",
    "inspect_dataset",
    "legal_action_indices",
    "legal_action_mask",
    "load_dataset",
    "read_manifest",
    "read_raw_game",
    "replay_game",
    "validate_dataset",
    "write_raw_game",
]

_GENERATION_EXPORTS = {
    "agent_spec",
    "create_agent",
    "generate_game",
    "generate_series",
}


def __getattr__(name: str) -> object:
    """Load AI-backed generation helpers only when explicitly requested."""

    if name in _GENERATION_EXPORTS:
        from nubia_training import generation

        return getattr(generation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
