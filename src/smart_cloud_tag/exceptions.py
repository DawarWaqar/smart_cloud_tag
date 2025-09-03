class SmartCloudTagError(Exception):
    pass


class SchemaValidationError(SmartCloudTagError):
    pass


class LLMError(SmartCloudTagError):
    pass


class StorageError(SmartCloudTagError):
    pass


class ConfigurationError(SmartCloudTagError):
    pass


class FileProcessingError(SmartCloudTagError):
    pass
