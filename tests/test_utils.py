import pytest
from smart_cloud_tag.utils import (
    parse_s3_uri,
    is_supported_file_type,
    get_file_type,
    detect_mime_type,
    parse_file_content,
    truncate_content,
    format_llm_prompt,
    parse_llm_response,
)
from smart_cloud_tag.models import FileType
from smart_cloud_tag.exceptions import FileProcessingError


class TestUtils:
    def test_parse_s3_uri(self):
        assert parse_s3_uri("s3://my-bucket") == "my-bucket"
        assert parse_s3_uri("s3://my-bucket/") == "my-bucket"
        assert parse_s3_uri("s3://my-bucket/path/file.txt") == "my-bucket"

        with pytest.raises(ValueError):
            parse_s3_uri("invalid://bucket")

        with pytest.raises(ValueError):
            parse_s3_uri("s3://")

    def test_is_supported_file_type(self):
        assert is_supported_file_type("test.txt") == True
        assert is_supported_file_type("test.md") == True
        assert is_supported_file_type("test.json") == True
        assert is_supported_file_type("test.csv") == True
        assert is_supported_file_type("test.pdf") == False
        assert is_supported_file_type("test") == False

    def test_get_file_type(self):
        assert get_file_type("test.txt") == FileType.TXT
        assert get_file_type("test.md") == FileType.MD
        assert get_file_type("test.json") == FileType.JSON
        assert get_file_type("test.csv") == FileType.CSV
        assert get_file_type("test.pdf") == None
        assert get_file_type("test") == None

    def test_detect_mime_type(self):
        assert detect_mime_type(b'{"key": "value"}') == "application/json"
        # Note: The CSV detection might not work reliably without magic library
        # So we'll test the fallback behavior
        result = detect_mime_type(b"name,age\nJohn,30")
        assert result in ["text/csv", "text/plain"]  # Either is acceptable
        assert detect_mime_type(b"plain text") == "text/plain"

    def test_parse_file_content_json(self):
        content = b'{"name": "John", "age": 30}'
        result = parse_file_content(content, FileType.JSON)
        assert "name" in result
        assert "John" in result

    def test_parse_file_content_csv(self):
        content = b"name,age\nJohn,30\nJane,25"
        result = parse_file_content(content, FileType.CSV)
        assert "Headers: name, age" in result
        assert "Row 1: John, 30" in result

    def test_parse_file_content_txt(self):
        content = b"This is plain text"
        result = parse_file_content(content, FileType.TXT)
        assert result == "This is plain text"

    def test_parse_file_content_invalid_json(self):
        content = b"invalid json"
        with pytest.raises(FileProcessingError):
            parse_file_content(content, FileType.JSON)

    def test_truncate_content(self):
        content = "This is a test"
        result = truncate_content(content, 100)
        assert result == content

        result = truncate_content(content, 5)
        assert len(result.encode("utf-8")) <= 5

    def test_format_llm_prompt(self):
        tags = {"type": ["document", "image"], "category": None}
        content = "Sample content"
        filename = "test.txt"

        prompt = format_llm_prompt(tags, content, filename)

        assert "test.txt" in prompt
        assert "Sample content" in prompt
        assert "type: must be one of ['document', 'image']" in prompt
        assert "category: deduce appropriate value" in prompt

    def test_format_llm_prompt_empty_filename(self):
        tags = {"type": ["document"]}
        content = "Sample content"

        with pytest.raises(ValueError):
            format_llm_prompt(tags, content, "")

    def test_parse_llm_response(self):
        response = "document, finance, true"
        tag_keys = ["type", "department", "confidential"]

        result = parse_llm_response(response, tag_keys)
        assert result == ["document", "finance", "true"]

    def test_parse_llm_response_with_prefix(self):
        response = "Generated tags: document, finance, true"
        tag_keys = ["type", "department", "confidential"]

        result = parse_llm_response(response, tag_keys)
        assert result == ["document", "finance", "true"]

    def test_parse_llm_response_insufficient_values(self):
        response = "document, finance"
        tag_keys = ["type", "department", "confidential"]

        result = parse_llm_response(response, tag_keys)
        assert result == ["document", "finance", "general"]

    def test_parse_llm_response_excess_values(self):
        response = "document, finance, true, extra"
        tag_keys = ["type", "department", "confidential"]

        result = parse_llm_response(response, tag_keys)
        assert result == ["document", "finance", "true"]
