"""
Utility functions for the smart_cloud_tag package.
"""

import json
import csv
import io
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import magic
from .models import FileType
from .exceptions import FileProcessingError


def parse_s3_uri(uri: str) -> str:
    """
    Parse S3 URI to get bucket name.

    Args:
        uri: S3 URI in format 's3://bucket' or 's3://bucket/'

    Returns:
        Bucket name

    Raises:
        ValueError: If URI format is invalid
    """
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with 's3://'")

    parsed = urlparse(uri)
    bucket = parsed.netloc

    if not bucket:
        raise ValueError("Invalid S3 URI: missing bucket name")

    return bucket


def is_supported_file_type(filename: str) -> bool:
    """
    Check if file type is supported for processing.

    Args:
        filename: Name of the file

    Returns:
        True if file type is supported
    """
    supported_extensions = {".txt", ".md", ".json", ".csv"}
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""
    return f".{file_ext}" in supported_extensions


def get_file_type(filename: str) -> Optional[FileType]:
    """
    Get the file type enum for a filename.

    Args:
        filename: Name of the file

    Returns:
        FileType enum or None if not supported
    """
    if not is_supported_file_type(filename):
        return None

    file_ext = filename.lower().split(".")[-1]
    try:
        return FileType(file_ext)
    except ValueError:
        return None


def detect_mime_type(content: bytes) -> str:
    """
    Detect MIME type from content bytes.

    Args:
        content: File content as bytes

    Returns:
        MIME type string
    """
    try:
        return magic.from_buffer(content, mime=True)
    except Exception:
        # Fallback to basic detection
        if content.startswith(b"{") or content.startswith(b"["):
            return "application/json"
        elif b"," in content[:1000] and b"\n" in content[:1000]:
            return "text/csv"
        else:
            return "text/plain"


def parse_file_content(content: bytes, file_type: FileType) -> str:
    """
    Parse file content based on file type.

    Args:
        content: Raw file content as bytes
        file_type: Type of the file

    Returns:
        Parsed text content

    Raises:
        FileProcessingError: If parsing fails
    """
    try:
        if file_type == FileType.JSON:
            # Parse JSON and convert to readable text
            data = json.loads(content.decode("utf-8"))
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif file_type == FileType.CSV:
            # Parse CSV and convert to readable text
            text_content = content.decode("utf-8")
            reader = csv.reader(io.StringIO(text_content))
            rows = list(reader)

            # Convert to readable format
            output = []
            for i, row in enumerate(rows):
                if i == 0:  # Header
                    output.append(f"Headers: {', '.join(row)}")
                else:
                    output.append(f"Row {i}: {', '.join(row)}")

            return "\n".join(output)

        else:  # TXT, MD
            return content.decode("utf-8")

    except Exception as e:
        raise FileProcessingError(f"Failed to parse {file_type.value} file: {str(e)}")


def truncate_content(content: str, max_bytes: int) -> str:
    """
    Truncate content to fit within max_bytes limit.

    Args:
        content: Text content to truncate
        max_bytes: Maximum bytes allowed

    Returns:
        Truncated content
    """
    content_bytes = content.encode("utf-8")
    if len(content_bytes) <= max_bytes:
        return content

    # Truncate and ensure we don't cut in the middle of a character
    truncated_bytes = content_bytes[:max_bytes]
    try:
        return truncated_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Find the last complete UTF-8 character
        while truncated_bytes and truncated_bytes[-1] & 0xC0 == 0x80:
            truncated_bytes = truncated_bytes[:-1]
        return truncated_bytes.decode("utf-8", errors="ignore")


def merge_tags(
    existing_tags: Dict[str, str], new_tags: Dict[str, str], tag_keys: List[str]
) -> Dict[str, str]:
    """
    Merge existing and new tags, respecting AWS S3 limits.

    Args:
        existing_tags: Existing tags on the object
        new_tags: New tags to apply
        tag_keys: Tag keys for the new tags

    Returns:
        Merged tags dictionary

    Raises:
        ValueError: If total tags would exceed 10
    """
    # Start with existing non-tag keys
    merged = {k: v for k, v in existing_tags.items() if k not in tag_keys}

    # Add new tag keys
    merged.update(new_tags)

    # Check AWS S3 limit
    if len(merged) > 10:
        raise ValueError(f"Total tags ({len(merged)}) would exceed the limit of 10")

    return merged


def format_llm_prompt(
    tags: Dict[str, Optional[List[str]]], content_preview: str, filename: str
) -> str:
    """
    Format the default prompt for the LLM.

    Args:
        tags: Dictionary of tag keys and their allowed values (null means deduce)
        content_preview: Preview of the file content
        filename: Name of the file being analyzed (required)

    Returns:
        Formatted prompt string
    """
    # Validate that filename is provided and not empty
    if not filename or not filename.strip():
        raise ValueError("filename is required and cannot be empty")

    tag_keys = list(tags.keys())

    # Build constraints for each tag
    constraints = []
    for key, allowed_values in tags.items():
        if allowed_values is None:
            constraints.append(
                f"- {key}: deduce appropriate value based on content and key name"
            )
        else:
            constraints.append(f"- {key}: must be one of {allowed_values}")

    constraints_text = "\n".join(constraints)

    # Include filename in the prompt (always required)
    filename_context = f"\nFile being analyzed: {filename}\n"

    prompt = f"""Analyze the following content and generate exactly {len(tag_keys)} tag values.

Tag keys and constraints:
{constraints_text}{filename_context}
Content preview:
{content_preview}

Instructions:
1. Generate exactly {len(tag_keys)} values, one for each tag key
2. Return only the values in order, separated by commas
3. Keep values concise (1-3 words when possible)
4. Make values relevant and descriptive for the content
5. For tags with allowed values, use only those values
6. For tags without allowed values, deduce appropriate values based on content and key name
7. If you see any abbreviations, interpret them according to context, following these examples:
    Example: "BOL#: 7782-CA-TOR-2025" → "bill_of_lading" (a shipping document)
    Example: "PO# 5567-AB" → "purchase_order" (a procurement document)

File Context Guidelines:
- Consider the filename as additional context for tagging decisions
- Use filename context to inform your understanding of the document type and content
- However, if the filename is not relevant to the content, ignore it

Example output format:
value1, value2, value3

Generated tags:"""

    return prompt


def format_custom_llm_prompt(
    custom_template: str,
    tags: Dict[str, Optional[List[str]]],
    content_preview: str,
    filename: str,
) -> str:
    """
    Format a custom prompt template with the required placeholders.

    Args:
        custom_template: User-provided custom prompt template
        tags: Dictionary of tag keys and their allowed values
        content_preview: Preview of the file content
        filename: Name of the file being analyzed

    Returns:
        Formatted custom prompt string

    Raises:
        ValueError: If required placeholders are missing from template
    """
    # Validate that custom template contains all required placeholders
    required_placeholders = ["{tags}", "{content}", "{filename}"]
    missing_placeholders = []

    for placeholder in required_placeholders:
        if placeholder not in custom_template:
            missing_placeholders.append(placeholder)

    if missing_placeholders:
        raise ValueError(
            f"Custom prompt template is missing required placeholders: {missing_placeholders}. "
            f"Required placeholders: {required_placeholders}"
        )

    # Format the custom template with actual values
    formatted_prompt = custom_template.format(
        tags=tags, content=content_preview, filename=filename
    )

    return formatted_prompt


def parse_llm_response(response: str, tag_keys: List[str]) -> List[str]:
    """
    Parse LLM response into list of tag values.

    Args:
        response: Raw response from LLM
        tag_keys: Expected tag keys

    Returns:
        List of tag values

    Raises:
        ValueError: If response cannot be parsed
    """
    # Clean the response
    cleaned = response.strip()

    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        "Generated tags:",
        "Tags:",
        "Values:",
        "Here are the tags:",
        "The tags are:",
        "Based on the content:",
    ]

    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()

    # Split by comma and clean each value
    values = [v.strip().strip("\"'") for v in cleaned.split(",")]

    # Filter out empty values
    values = [v for v in values if v]

    # Pad or truncate to match tag keys length
    if len(values) < len(tag_keys):
        # Pad with generic values
        while len(values) < len(tag_keys):
            values.append("general")
    elif len(values) > len(tag_keys):
        # Truncate to match tag keys
        values = values[: len(tag_keys)]

    if len(values) != len(tag_keys):
        raise ValueError(f"Expected {len(tag_keys)} values, got {len(values)}")

    return values
