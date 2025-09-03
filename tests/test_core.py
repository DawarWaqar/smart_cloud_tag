import pytest
from unittest.mock import Mock, patch
from smart_cloud_tag import SmartCloudTagger
from smart_cloud_tag.exceptions import ConfigurationError


class TestSmartCloudTagger:
    def test_init_with_s3_uri(self):
        with patch("smart_cloud_tag.core.AWSS3Provider") as mock_provider:
            with patch("smart_cloud_tag.core.OpenAIProvider") as mock_llm:
                with patch.dict("os.environ", {"API_KEY": "test-key"}):
                    tagger = SmartCloudTagger(
                        storage_uri="s3://test-bucket",
                        tags={"type": ["document", "image"]},
                    )
                    assert tagger.storage_provider_type == "aws"
                    mock_provider.assert_called_once()

    def test_init_with_azure_uri(self):
        with patch("smart_cloud_tag.core.AzureBlobProvider") as mock_provider:
            with patch("smart_cloud_tag.core.OpenAIProvider") as mock_llm:
                with patch("smart_cloud_tag.core.AZURE_AVAILABLE", True):
                    with patch.dict(
                        "os.environ",
                        {
                            "API_KEY": "test-key",
                            "AZURE_STORAGE_CONNECTION_STRING": "test-connection",
                        },
                    ):
                        tagger = SmartCloudTagger(
                            storage_uri="az://test-container",
                            tags={"type": ["document", "image"]},
                        )
                        assert tagger.storage_provider_type == "azure"
                        mock_provider.assert_called_once()

    def test_init_with_gcp_uri(self):
        with patch("smart_cloud_tag.core.GCSProvider") as mock_provider:
            with patch("smart_cloud_tag.core.OpenAIProvider") as mock_llm:
                with patch("smart_cloud_tag.core.GCS_AVAILABLE", True):
                    with patch.dict(
                        "os.environ",
                        {
                            "API_KEY": "test-key",
                            "GOOGLE_APPLICATION_CREDENTIALS": "test-credentials.json",
                        },
                    ):
                        tagger = SmartCloudTagger(
                            storage_uri="gs://test-bucket",
                            tags={"type": ["document", "image"]},
                        )
                        assert tagger.storage_provider_type == "gcp"
                        mock_provider.assert_called_once()

    def test_init_with_invalid_uri(self):
        with pytest.raises(ConfigurationError):
            SmartCloudTagger(
                storage_uri="invalid://bucket", tags={"type": ["document", "image"]}
            )

    def test_init_without_api_key(self):
        with patch("smart_cloud_tag.core.AWSS3Provider") as mock_provider:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ConfigurationError):
                    SmartCloudTagger(
                        storage_uri="s3://test-bucket",
                        tags={"type": ["document", "image"]},
                    )

    def test_detect_storage_provider(self):
        with patch("smart_cloud_tag.core.AWSS3Provider"):
            with patch("smart_cloud_tag.core.OpenAIProvider"):
                with patch.dict("os.environ", {"API_KEY": "test-key"}):
                    tagger = SmartCloudTagger(
                        storage_uri="s3://test-bucket", tags={"type": ["document"]}
                    )

                    assert tagger._detect_storage_provider("s3://bucket") == "aws"
                    assert tagger._detect_storage_provider("az://container") == "azure"
                    assert tagger._detect_storage_provider("gs://bucket") == "gcp"

                    with pytest.raises(ConfigurationError):
                        tagger._detect_storage_provider("invalid://bucket")

    def test_get_storage_info(self):
        with patch("smart_cloud_tag.core.AWSS3Provider") as mock_provider:
            with patch("smart_cloud_tag.core.OpenAIProvider"):
                with patch.dict("os.environ", {"API_KEY": "test-key"}):
                    mock_instance = Mock()
                    mock_instance.get_bucket_name.return_value = "test-bucket"
                    mock_instance.__class__.__name__ = "AWSS3Provider"
                    mock_provider.return_value = mock_instance

                    tagger = SmartCloudTagger(
                        storage_uri="s3://test-bucket", tags={"type": ["document"]}
                    )

                    info = tagger.get_storage_info()
                    assert info["provider"] == "AWSS3Provider"
                    assert info["bucket"] == "test-bucket"

    def test_get_llm_info(self):
        with patch("smart_cloud_tag.core.AWSS3Provider"):
            with patch("smart_cloud_tag.core.OpenAIProvider") as mock_llm:
                with patch.dict("os.environ", {"API_KEY": "test-key"}):
                    mock_instance = Mock()
                    mock_instance.get_model_name.return_value = "gpt-4.1"
                    mock_instance.__class__.__name__ = "OpenAIProvider"
                    mock_llm.return_value = mock_instance

                    tagger = SmartCloudTagger(
                        storage_uri="s3://test-bucket", tags={"type": ["document"]}
                    )

                    info = tagger.get_llm_info()
                    assert info["provider"] == "OpenAIProvider"
                    assert info["model"] == "gpt-4.1"

    def test_get_tags_info(self):
        with patch("smart_cloud_tag.core.AWSS3Provider"):
            with patch("smart_cloud_tag.core.OpenAIProvider"):
                with patch.dict("os.environ", {"API_KEY": "test-key"}):
                    tags = {"type": ["document", "image"], "category": None}

                    tagger = SmartCloudTagger(storage_uri="s3://test-bucket", tags=tags)

                    info = tagger.get_tags_info()
                    assert info["type"] == "Allowed values: ['document', 'image']"
                    assert info["category"] == "LLM will deduce value"
