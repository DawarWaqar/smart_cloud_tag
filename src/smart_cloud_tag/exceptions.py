"""
Custom exceptions for the smart_cloud_tag package.
"""


class SmartCloudTagError(Exception):
    """Base exception for smart_cloud_tag package."""

    pass


class SchemaValidationError(SmartCloudTagError):
    """Raised when schema validation fails."""

    pass


class LLMError(SmartCloudTagError):
    """Raised when LLM operations fail."""

    pass


class StorageError(SmartCloudTagError):
    """Raised when storage operations fail."""

    pass


class ConfigurationError(SmartCloudTagError):
    """Raised when configuration is invalid."""

    pass


class FileProcessingError(SmartCloudTagError):
    """Raised when file processing fails."""

    pass
