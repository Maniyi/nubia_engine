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
