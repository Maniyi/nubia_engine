"""Focused errors raised by the optional neural-training package."""


class NeuralError(Exception):
    """Base class for expected neural subsystem failures."""


class NeuralConfigurationError(NeuralError, ValueError):
    """Raised when a model or training configuration is invalid."""


class NeuralInputError(NeuralError, ValueError):
    """Raised when tensors do not satisfy a public neural contract."""


class DeviceUnavailableError(NeuralError, RuntimeError):
    """Raised when an explicitly requested accelerator is unavailable."""


class CheckpointError(NeuralError):
    """Raised when checkpoint integrity or compatibility validation fails."""


class NonFiniteTrainingError(NeuralError, RuntimeError):
    """Raised when a loss or gradient becomes non-finite."""


class NeuralInferenceError(NeuralError, ValueError):
    """Raised when a position evaluation violates the inference contract."""


class MCTSConfigurationError(NeuralError, ValueError):
    """Raised when an MCTS configuration is invalid or incompatible."""


class MCTSSearchError(NeuralError, ValueError):
    """Raised when MCTS cannot search the supplied position safely."""


class SelfPlayError(NeuralError, ValueError):
    """Raised when self-play configuration, records, or corpora are invalid."""


class IterationError(NeuralError, ValueError):
    """Raised when a bounded training iteration cannot complete safely."""
