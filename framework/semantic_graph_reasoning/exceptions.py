"""Project-specific exceptions with credential-safe messages."""


class SemanticGraphReasoningError(Exception):
    """Base exception for expected Semantic Graph Reasoning failures."""


class ConfigurationError(SemanticGraphReasoningError):
    """Raised when a provider is missing necessary non-secret configuration."""


class DatasetFormatError(SemanticGraphReasoningError):
    """Raised for documented but unsupported external dataset formats."""
