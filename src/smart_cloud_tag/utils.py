import json
import csv
import io
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import magic
from .models import FileType
from .exceptions import FileProcessingError


def parse_s3_uri(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with 's3://'")

    parsed = urlparse(uri)
    bucket = parsed.netloc

    if not bucket:
        raise ValueError("Invalid S3 URI: missing bucket name")

    return bucket


def is_supported_file_type(filename: str) -> bool:
    supported_extensions = {".txt", ".md", ".json", ".csv"}
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""
    return f".{file_ext}" in supported_extensions


def get_file_type(filename: str) -> Optional[FileType]:
    if not is_supported_file_type(filename):
        return None

    file_ext = filename.lower().split(".")[-1]
    try:
        return FileType(file_ext)
    except ValueError:
        return None


def detect_mime_type(content: bytes) -> str:
    try:
        return magic.from_buffer(content, mime=True)
    except Exception:
        if content.startswith(b"{") or content.startswith(b"["):
            return "application/json"
        elif b"," in content[:1000] and b"\n" in content[:1000]:
            return "text/csv"
        else:
            return "text/plain"


def parse_file_content(content: bytes, file_type: FileType) -> str:
    try:
        if file_type == FileType.JSON:
            data = json.loads(content.decode("utf-8"))
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif file_type == FileType.CSV:
            text_content = content.decode("utf-8")
            reader = csv.reader(io.StringIO(text_content))
            rows = list(reader)

            output = []
            for i, row in enumerate(rows):
                if i == 0:
                    output.append(f"Headers: {', '.join(row)}")
                else:
                    output.append(f"Row {i}: {', '.join(row)}")

            return "\n".join(output)

        else:
            return content.decode("utf-8")

    except Exception as e:
        raise FileProcessingError(f"Failed to parse {file_type.value} file: {str(e)}")


def truncate_content(content: str, max_bytes: int) -> str:
    content_bytes = content.encode("utf-8")
    if len(content_bytes) <= max_bytes:
        return content

    truncated_bytes = content_bytes[:max_bytes]
    try:
        return truncated_bytes.decode("utf-8")
    except UnicodeDecodeError:
        while truncated_bytes and truncated_bytes[-1] & 0xC0 == 0x80:
            truncated_bytes = truncated_bytes[:-1]
        return truncated_bytes.decode("utf-8", errors="ignore")


def merge_tags(
    existing_tags: Dict[str, str], new_tags: Dict[str, str], tag_keys: List[str]
) -> Dict[str, str]:
    merged = {k: v for k, v in existing_tags.items() if k not in tag_keys}
    merged.update(new_tags)

    if len(merged) > 10:
        raise ValueError(f"Total tags ({len(merged)}) would exceed the limit of 10")

    return merged


def format_llm_prompt(
    tags: Dict[str, Optional[List[str]]], content_preview: str, filename: str
) -> str:
    from .config import DEFAULT_PROMPT_TEMPLATE

    if not filename or not filename.strip():
        raise ValueError("filename is required and cannot be empty")

    tag_keys = list(tags.keys())

    constraints = []
    for key, allowed_values in tags.items():
        if allowed_values is None:
            constraints.append(
                f"- {key}: deduce appropriate value based on content and key name"
            )
        else:
            constraints.append(f"- {key}: must be one of {allowed_values}")

    constraints_text = "\n".join(constraints)
    filename_context = f"\nFile being analyzed: {filename}\n"

    prompt = DEFAULT_PROMPT_TEMPLATE.format(
        num_tags=len(tag_keys),
        constraints_text=constraints_text,
        filename_context=filename_context,
        content_preview=content_preview,
    )

    return prompt


def format_custom_llm_prompt(
    custom_template: str,
    tags: Dict[str, Optional[List[str]]],
    content_preview: str,
    filename: str,
) -> str:
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

    formatted_prompt = custom_template.format(
        tags=tags, content=content_preview, filename=filename
    )

    return formatted_prompt


def parse_llm_response(response: str, tag_keys: List[str]) -> List[str]:
    cleaned = response.strip()

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

    values = [v.strip().strip("\"'") for v in cleaned.split(",")]
    values = [v for v in values if v]

    if len(values) < len(tag_keys):
        while len(values) < len(tag_keys):
            values.append("general")
    elif len(values) > len(tag_keys):
        values = values[: len(tag_keys)]

    if len(values) != len(tag_keys):
        raise ValueError(f"Expected {len(tag_keys)} values, got {len(values)}")

    return values
