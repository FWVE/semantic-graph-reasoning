"""Project-specific exceptions with credential-safe messages."""


class MASGEError(Exception):
    """Base exception for expected MA-SGE failures."""


class ConfigurationError(MASGEError):
    """Raised when a provider is missing necessary non-secret configuration."""


class DatasetFormatError(MASGEError):
    """Raised for documented but unsupported external dataset formats."""
