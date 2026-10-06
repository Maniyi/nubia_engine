"""Focused failures raised by the machine-learning representation layer."""


class RepresentationError(ValueError):
    """Base class for invalid or incompatible training representations."""


class InvalidEncodingError(RepresentationError):
    """Raised when encoded numerical data violates its public specification."""


class UnsupportedPieceTypeError(RepresentationError):
    """Raised when the engine exposes an unknown piece type."""


class UnsupportedActionKindError(RepresentationError):
    """Raised when the engine exposes an unknown action kind."""


class IllegalOrStaleActionError(RepresentationError):
    """Raised when an action is not legal in the supplied state."""


class InvalidActionIndexError(RepresentationError):
    """Raised when an action index is not a valid vocabulary index."""


class UnresolvedActionIndexError(RepresentationError):
    """Raised when an index does not identify a current legal action."""


class ActionIndexCollisionError(RepresentationError):
    """Raised when distinct current legal actions share a structural index."""


class InvalidPolicyTargetError(RepresentationError):
    """Raised when a sparse policy target is not a legal probability distribution."""


class VersionMismatchError(RepresentationError):
    """Raised when representation component versions are incompatible."""


class TrainingDataError(ValueError):
    """Base class for expected record and dataset pipeline failures."""


class SchemaVersionError(TrainingDataError):
    """Raised when persisted data uses an unsupported schema version."""


class RecordValidationError(TrainingDataError):
    """Raised when a raw game record is malformed or inconsistent."""


class RecordIntegrityError(TrainingDataError):
    """Raised when canonical game content or storage conflicts."""


class ReplayValidationError(TrainingDataError):
    """Raised when authoritative replay disagrees with a raw record."""


class DatasetValidationError(TrainingDataError):
    """Raised when a shard or manifest fails integrity validation."""


class AgentSpecError(TrainingDataError):
    """Raised for unsupported or non-reproducible generation agents."""
