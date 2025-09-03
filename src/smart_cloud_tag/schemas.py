from typing import Dict, List, Any, Optional
from .models import TaggingConfig, ObjectTags
from .exceptions import SchemaValidationError


def get_provider_tag_limits(provider: str) -> Dict[str, int]:
    limits = {
        "aws": {"max_tags": 10, "max_key": 128, "max_value": 256},
        "azure": {"max_tags": 10, "max_key": 128, "max_value": 256},
        "gcp": {"max_tags": 64, "max_key": 128, "max_value": 1024},
    }

    if provider not in limits:
        raise SchemaValidationError(f"Unsupported storage provider: {provider}")

    return limits[provider]


def validate_tagging_config(config: TaggingConfig, provider: str) -> None:
    limits = get_provider_tag_limits(provider)

    if len(config.tags) >= limits["max_tags"]:
        raise SchemaValidationError(f"Maximum {limits['max_tags']} tags per object")

    if len(config.tags) != len(set(config.tags.keys())):
        raise SchemaValidationError("Tag keys must be unique")

    for key in config.tags.keys():
        if not key or not key.strip():
            raise SchemaValidationError("Tag keys cannot be empty")
        if len(key) > limits["max_key"]:
            raise SchemaValidationError(f"Tag key '{key}' exceeds {limits['max_key']} character limit")
        if not key.replace("-", "").replace("_", "").isalnum():
            raise SchemaValidationError(f"Tag key '{key}' contains invalid characters")


def validate_tag_values(values: List[str], tag_keys: List[str], provider: str) -> None:
    limits = get_provider_tag_limits(provider)

    if len(values) != len(tag_keys):
        raise SchemaValidationError(f"Expected {len(tag_keys)} values, got {len(values)}")

    for i, value in enumerate(values):
        if not value or not value.strip():
            raise SchemaValidationError(f"Tag value for '{tag_keys[i]}' cannot be empty")
        if len(value) > limits["max_value"]:
            raise SchemaValidationError(f"Tag value for '{tag_keys[i]}' exceeds {limits['max_value']} character limit")


def create_tag_mapping(tag_keys: List[str], values: List[str], provider: str = "aws") -> Dict[str, str]:
    validate_tag_values(values, tag_keys, provider)

    tag_mapping = {}
    for key, value in zip(tag_keys, values):
        tag_mapping[key] = value.strip()

    return tag_mapping


def validate_existing_tags(tags: Dict[str, str], provider: str) -> None:
    limits = get_provider_tag_limits(provider)

    if len(tags) > limits["max_tags"]:
        raise SchemaValidationError(f"Object has {len(tags)} tags, exceeding the limit of {limits['max_tags']}")

    for key, value in tags.items():
        if not key or not key.strip():
            raise SchemaValidationError("Tag key cannot be empty")
        if len(key) > limits["max_key"]:
            raise SchemaValidationError(f"Tag key '{key}' exceeds {limits['max_key']} character limit")
        if not value or not value.strip():
            raise SchemaValidationError(f"Tag value for '{key}' cannot be empty")
        if len(value) > limits["max_value"]:
            raise SchemaValidationError(f"Tag value for '{key}' exceeds {limits['max_value']} character limit")


def merge_and_validate_tags(
    existing_tags: Dict[str, str],
    new_tags: Dict[str, str],
    tag_keys: List[str],
    provider: str = "aws",
) -> Dict[str, str]:
    limits = get_provider_tag_limits(provider)

    validate_existing_tags(existing_tags, provider)
    validate_tag_values(list(new_tags.values()), tag_keys, provider)

    merged = {k: v for k, v in existing_tags.items() if k not in tag_keys}
    merged.update(new_tags)

    if len(merged) > limits["max_tags"]:
        raise SchemaValidationError(f"Total tags ({len(merged)}) would exceed the limit of {limits['max_tags']}")

    return merged


def create_object_tags_result(
    existing_tags: Dict[str, str],
    proposed_tags: Optional[Dict[str, str]] = None,
    applied_tags: Optional[Dict[str, str]] = None,
    skipped_reason: Optional[str] = None,
) -> ObjectTags:
    return ObjectTags(
        existing=existing_tags,
        proposed=proposed_tags,
        applied=applied_tags,
        skipped_reason=skipped_reason,
    )