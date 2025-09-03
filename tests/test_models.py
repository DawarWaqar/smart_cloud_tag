import pytest
from pydantic import ValidationError
from smart_cloud_tag.models import (
    ProcessingMode,
    FileType,
    TaggingConfig,
    ObjectTags,
    TaggingResult,
    LLMRequest,
    LLMResponse,
)


class TestModels:
    def test_processing_mode(self):
        assert ProcessingMode.PREVIEW == "preview"
        assert ProcessingMode.APPLY == "apply"

    def test_file_type(self):
        assert FileType.TXT == "txt"
        assert FileType.MD == "md"
        assert FileType.JSON == "json"
        assert FileType.CSV == "csv"

    def test_tagging_config_valid(self):
        config = TaggingConfig(
            llm_model="gpt-4.1",
            storage_uri="s3://test-bucket",
            tags={"type": ["document", "image"]},
            max_bytes=5000,
        )
        assert config.llm_model == "gpt-4.1"
        assert config.storage_uri == "s3://test-bucket"
        assert config.tags == {"type": ["document", "image"]}
        assert config.max_bytes == 5000

    def test_tagging_config_invalid_tags_count(self):
        with pytest.raises(ValidationError):
            TaggingConfig(
                llm_model="gpt-4.1",
                storage_uri="s3://test-bucket",
                tags={f"tag{i}": ["value"] for i in range(10)},
                max_bytes=5000,
            )

    def test_tagging_config_empty_tags(self):
        with pytest.raises(ValidationError):
            TaggingConfig(
                llm_model="gpt-4.1",
                storage_uri="s3://test-bucket",
                tags={},
                max_bytes=5000,
            )

    def test_tagging_config_invalid_max_bytes(self):
        with pytest.raises(ValidationError):
            TaggingConfig(
                llm_model="gpt-4.1",
                storage_uri="s3://test-bucket",
                tags={"type": ["document"]},
                max_bytes=0,
            )

    def test_object_tags(self):
        tags = ObjectTags(
            existing={"old": "value"},
            proposed={"new": "value"},
            applied={"final": "value"},
            skipped_reason="test reason",
        )
        assert tags.existing == {"old": "value"}
        assert tags.proposed == {"new": "value"}
        assert tags.applied == {"final": "value"}
        assert tags.skipped_reason == "test reason"

    def test_tagging_result(self):
        config = TaggingConfig(
            llm_model="gpt-4.1",
            storage_uri="s3://test-bucket",
            tags={"type": ["document"]},
        )

        result = TaggingResult(
            mode=ProcessingMode.PREVIEW, config=config, results={}, summary={}
        )

        assert result.mode == ProcessingMode.PREVIEW
        assert result.config == config
        assert result.results == {}
        assert result.summary == {}

    def test_tagging_result_add_result(self):
        config = TaggingConfig(
            llm_model="gpt-4.1",
            storage_uri="s3://test-bucket",
            tags={"type": ["document"]},
        )

        result = TaggingResult(
            mode=ProcessingMode.PREVIEW, config=config, results={}, summary={}
        )

        tags = ObjectTags(existing={"old": "value"})
        result.add_result("test-file.txt", tags)

        assert "test-file.txt" in result.results
        assert result.results["test-file.txt"] == tags

    def test_tagging_result_get_summary_stats(self):
        config = TaggingConfig(
            llm_model="gpt-4.1",
            storage_uri="s3://test-bucket",
            tags={"type": ["document"]},
        )

        result = TaggingResult(
            mode=ProcessingMode.PREVIEW, config=config, results={}, summary={}
        )

        # Add some test results
        result.add_result(
            "file1.txt", ObjectTags(existing={}, proposed={"type": "document"})
        )
        result.add_result(
            "file2.txt", ObjectTags(existing={}, skipped_reason="unsupported")
        )
        result.add_result(
            "file3.txt", ObjectTags(existing={}, applied={"type": "document"})
        )

        stats = result.get_summary_stats()
        assert stats["total_objects"] == 3
        assert stats["processed"] == 1  # Only file1 has proposed tags
        assert stats["skipped"] == 1  # file2
        assert stats["applied"] == 1  # file3
        assert stats["success_rate"] == 1 / 3

    def test_llm_request(self):
        request = LLMRequest(
            content="Sample content",
            tags={"type": ["document", "image"]},
            filename="test.txt",
            custom_prompt_template="Custom template",
        )

        assert request.content == "Sample content"
        assert request.tags == {"type": ["document", "image"]}
        assert request.filename == "test.txt"
        assert request.custom_prompt_template == "Custom template"

    def test_llm_response(self):
        response = LLMResponse(
            tags=["document", "finance", "true"],
            confidence=0.95,
            reasoning="Based on content analysis",
        )

        assert response.tags == ["document", "finance", "true"]
        assert response.confidence == 0.95
        assert response.reasoning == "Based on content analysis"
