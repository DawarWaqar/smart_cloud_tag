"""
Schema validation and mapping helpers for the smart_cloud_tag package.
"""

from typing import Dict, List, Any, Optional
from .models import TaggingConfig, ObjectTags
from .exceptions import SchemaValidationError


def get_provider_tag_limits(provider: str) -> Dict[str, int]:
    """
    Get tag limits for a specific storage provider.

    Args:
        provider: Storage provider type ("aws", "azure", "gcp")

    Returns:
        Dictionary with tag limits

    Raises:
        SchemaValidationError: If provider is not supported
    """
    limits = {
        "aws": {"max_tags": 10, "max_key": 128, "max_value": 256},
        "azure": {"max_tags": 10, "max_key": 128, "max_value": 256},
        "gcp": {"max_tags": 64, "max_key": 128, "max_value": 1024},
    }

    if provider not in limits:
        raise SchemaValidationError(f"Unsupported storage provider: {provider}")

    return limits[provider]


def validate_tagging_config(config: TaggingConfig, provider: str) -> None:
    """
    Validate tagging configuration.

    Args:
        config: TaggingConfig to validate
        provider: Storage provider type for limit validation

    Raises:
        SchemaValidationError: If validation fails
    """
    limits = get_provider_tag_limits(provider)

    # Check tags length
    if len(config.tags) >= limits["max_tags"]:
        raise SchemaValidationError(f"Maximum {limits['max_tags']} tags per object")

    # Check for duplicate tag keys
    if len(config.tags) != len(set(config.tags.keys())):
        raise SchemaValidationError("Tag keys must be unique")

    # Validate tag key format
    for key in config.tags.keys():
        if not key or not key.strip():
            raise SchemaValidationError("Tag keys cannot be empty")
        if len(key) > limits["max_key"]:
            raise SchemaValidationError(
                f"Tag key '{key}' exceeds {limits['max_key']} character limit"
            )
        if not key.replace("-", "").replace("_", "").isalnum():
            raise SchemaValidationError(f"Tag key '{key}' contains invalid characters")


def validate_tag_values(values: List[str], tag_keys: List[str], provider: str) -> None:
    """
    Validate tag values from LLM response.

    Args:
        values: List of tag values
        tag_keys: List of tag keys
        provider: Storage provider type for limit validation

    Raises:
        SchemaValidationError: If validation fails
    """
    limits = get_provider_tag_limits(provider)

    if len(values) != len(tag_keys):
        raise SchemaValidationError(
            f"Expected {len(tag_keys)} values, got {len(values)}"
        )

    for i, value in enumerate(values):
        if not value or not value.strip():
            raise SchemaValidationError(
                f"Tag value for '{tag_keys[i]}' cannot be empty"
            )
        if len(value) > limits["max_value"]:
            raise SchemaValidationError(
                f"Tag value for '{tag_keys[i]}' exceeds {limits['max_value']} character limit"
            )


def create_tag_mapping(
    tag_keys: List[str], values: List[str], provider: str = "aws"
) -> Dict[str, str]:
    """
    Create a mapping from tag keys to tag values.

    Args:
        tag_keys: List of tag keys
        values: List of tag values
        provider: Storage provider type for limit validation

    Returns:
        Dictionary mapping tag keys to values

    Raises:
        SchemaValidationError: If validation fails
    """
    validate_tag_values(values, tag_keys, provider)

    # Create the mapping
    tag_mapping = {}
    for key, value in zip(tag_keys, values):
        tag_mapping[key] = value.strip()

    return tag_mapping


def validate_existing_tags(tags: Dict[str, str], provider: str) -> None:
    """
    Validate existing tags from storage object.

    Args:
        tags: Dictionary of existing tags
        provider: Storage provider type for limit validation

    Raises:
        SchemaValidationError: If validation fails
    """
    limits = get_provider_tag_limits(provider)

    if len(tags) > limits["max_tags"]:
        raise SchemaValidationError(
            f"Object has {len(tags)} tags, exceeding the limit of {limits['max_tags']}"
        )

    for key, value in tags.items():
        if not key or not key.strip():
            raise SchemaValidationError("Tag key cannot be empty")
        if len(key) > limits["max_key"]:
            raise SchemaValidationError(
                f"Tag key '{key}' exceeds {limits['max_key']} character limit"
            )
        if not value or not value.strip():
            raise SchemaValidationError(f"Tag value for '{key}' cannot be empty")
        if len(value) > limits["max_value"]:
            raise SchemaValidationError(
                f"Tag value for '{key}' exceeds {limits['max_value']} character limit"
            )


def sanitize_tag_key(key: str) -> str:
    """
    Sanitize tag key to ensure it's valid for AWS S3.

    Args:
        key: Raw tag key

    Returns:
        Sanitized tag key
    """
    # Remove leading/trailing whitespace
    sanitized = key.strip()

    # Replace invalid characters with underscores
    invalid_chars = [
        " ",
        "\t",
        "\n",
        "\r",
        "\\",
        "/",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
    ]
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")

    # Ensure it starts with a letter or number
    if sanitized and not sanitized[0].isalnum():
        sanitized = "tag_" + sanitized

    # Truncate if too long
    if len(sanitized) > 128:
        sanitized = sanitized[:128]

    return sanitized


def sanitize_tag_value(value: str) -> str:
    """
    Sanitize tag value to ensure it's valid for AWS S3.

    Args:
        value: Raw tag value

    Returns:
        Sanitized tag value
    """
    # Remove leading/trailing whitespace
    sanitized = value.strip()

    # Replace invalid characters with spaces
    invalid_chars = ["\t", "\n", "\r", "\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars:
        sanitized = sanitized.replace(char, " ")

    # Normalize multiple spaces
    import re

    sanitized = re.sub(r"\s+", " ", sanitized)

    # Truncate if too long
    if len(sanitized) > 256:
        sanitized = sanitized[:256]

    return sanitized


def merge_and_validate_tags(
    existing_tags: Dict[str, str],
    new_tags: Dict[str, str],
    tag_keys: List[str],
    provider: str = "aws",
) -> Dict[str, str]:
    """
    Merge existing and new tags, ensuring storage provider compliance.

    Args:
        existing_tags: Existing tags on the object
        new_tags: New tags to apply
        tag_keys: Tag keys for the new tags
        provider: Storage provider type for limit validation

    Returns:
        Merged and validated tags dictionary

    Raises:
        SchemaValidationError: If validation fails
    """
    limits = get_provider_tag_limits(provider)

    # Validate inputs
    validate_existing_tags(existing_tags, provider)
    validate_tag_values(list(new_tags.values()), tag_keys, provider)

    # Start with existing non-tag keys
    merged = {k: v for k, v in existing_tags.items() if k not in tag_keys}

    # Add new tag keys
    merged.update(new_tags)

    # Check total limit
    if len(merged) > limits["max_tags"]:
        raise SchemaValidationError(
            f"Total tags ({len(merged)}) would exceed the limit of {limits['max_tags']}"
        )

    return merged


def create_object_tags_result(
    existing_tags: Dict[str, str],
    proposed_tags: Optional[Dict[str, str]] = None,
    applied_tags: Optional[Dict[str, str]] = None,
    skipped_reason: Optional[str] = None,
) -> ObjectTags:
    """
    Create an ObjectTags result object.

    Args:
        existing_tags: Existing tags on the object
        proposed_tags: Proposed tags from LLM (optional)
        applied_tags: Actually applied tags (optional)
        skipped_reason: Reason if object was skipped (optional)

    Returns:
        ObjectTags object
    """
    return ObjectTags(
        existing=existing_tags,
        proposed=proposed_tags,
        applied=applied_tags,
        skipped_reason=skipped_reason,
    )
