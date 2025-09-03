import os
from typing import Dict, Optional, Any, List
from .models import (
    TaggingConfig,
    TaggingResult,
    ProcessingMode,
    LLMRequest,
)
from .providers import (
    AWSS3Provider,
    AzureBlobProvider,
    GCSProvider,
    AWS_AVAILABLE,
    AZURE_AVAILABLE,
    GCS_AVAILABLE,
)
from .llm import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OPENAI_AVAILABLE,
    ANTHROPIC_AVAILABLE,
    GEMINI_AVAILABLE,
)
from .utils import (
    parse_file_content,
    truncate_content,
)
from .schemas import (
    validate_tagging_config,
    create_tag_mapping,
    merge_and_validate_tags,
    create_object_tags_result,
)
from .exceptions import (
    SmartCloudTagError,
    ConfigurationError,
)


class SmartCloudTagger:
    def __init__(
        self,
        storage_uri: str,
        tags: Dict[str, Optional[List[str]]],
        llm_model: str = "",
        llm_provider: str = "openai",
        max_bytes: Optional[int] = 5000,
        custom_prompt_template: Optional[str] = None,
    ):
        self.storage_uri = storage_uri
        self.tags = tags
        self.llm_model = llm_model
        self.llm_provider_type = llm_provider.lower()
        self.max_bytes = max_bytes
        self.custom_prompt_template = custom_prompt_template

        self.storage_provider_type = self._detect_storage_provider(storage_uri)

        if self.llm_provider_type not in ["openai", "anthropic", "gemini"]:
            raise ConfigurationError(
                "llm_provider must be 'openai', 'anthropic', or 'gemini'"
            )

        if not llm_model:
            default_llm_model = {
                "openai": "gpt-4.1",
                "anthropic": "claude-3-5-sonnet-20241022",
                "gemini": "gemini-2.5-pro",
            }
            self.llm_model = default_llm_model[self.llm_provider_type]

        self.config = TaggingConfig(
            llm_model=self.llm_model,
            storage_uri=self.storage_uri,
            tags=tags,
            max_bytes=self.max_bytes,
        )

        try:
            validate_tagging_config(self.config, self.storage_provider_type)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {str(e)}")

        self._init_storage_provider()
        self._init_llm_provider()

        if not self.storage_provider.is_supported_file_type("test.txt"):
            raise ConfigurationError("Storage provider not properly configured")

        if not self.llm_provider.is_available():
            raise ConfigurationError("LLM provider not available")

    def _detect_storage_provider(self, storage_uri: str) -> str:
        storage_uri_lower = storage_uri.lower()

        if storage_uri_lower.startswith("s3://"):
            return "aws"
        elif storage_uri_lower.startswith("az://"):
            return "azure"
        elif storage_uri_lower.startswith("gs://"):
            return "gcp"
        else:
            raise ConfigurationError(
                f"Unsupported storage URI format: {storage_uri}. "
                f"Supported formats: s3:// (AWS), az:// (Azure), gs:// (GCP)"
            )

    def _init_storage_provider(self):
        if self.storage_provider_type == "aws":
            if not AWS_AVAILABLE:
                raise ConfigurationError(
                    "AWS provider not available. Install with: pip install smart_cloud_tag[aws]"
                )
            self.storage_provider = AWSS3Provider(storage_uri=self.storage_uri)

        elif self.storage_provider_type == "azure":
            azure_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not azure_connection_string:
                raise ConfigurationError(
                    "AZURE_STORAGE_CONNECTION_STRING environment variable is required for Azure provider"
                )
            if not AZURE_AVAILABLE:
                raise ConfigurationError(
                    "Azure provider not available. Install with: pip install smart_cloud_tag[azure]"
                )
            self.storage_provider = AzureBlobProvider(
                storage_uri=self.storage_uri,
                connection_string=azure_connection_string,
            )

        elif self.storage_provider_type == "gcp":
            gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not gcp_credentials_path:
                raise ConfigurationError(
                    "GOOGLE_APPLICATION_CREDENTIALS environment variable is required for GCP provider"
                )
            if not GCS_AVAILABLE:
                raise ConfigurationError(
                    "GCP provider not available. Install with: pip install smart_cloud_tag[gcp]"
                )
            self.storage_provider = GCSProvider(
                storage_uri=self.storage_uri,
                credentials_path=gcp_credentials_path,
            )

    def _init_llm_provider(self):
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "API_KEY environment variable is required. Set it in your .env file."
            )

        if self.llm_provider_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ConfigurationError(
                    "OpenAI provider not available. Install with: pip install smart_cloud_tag[openai]"
                )
            self.llm_provider = OpenAIProvider(
                model=self.llm_model,
                api_key=self.api_key,
            )

        elif self.llm_provider_type == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ConfigurationError(
                    "Anthropic provider not available. Install with: pip install smart_cloud_tag[anthropic]"
                )
            self.llm_provider = AnthropicProvider(
                model=self.llm_model,
                api_key=self.api_key,
            )

        elif self.llm_provider_type == "gemini":
            if not GEMINI_AVAILABLE:
                raise ConfigurationError(
                    "Gemini provider not available. Install with: pip install smart_cloud_tag[gemini]"
                )
            self.llm_provider = GeminiProvider(
                model=self.llm_model,
                api_key=self.api_key,
            )

    def preview_tags(self, max_bytes: Optional[int] = None) -> TaggingResult:
        return self._process_objects(ProcessingMode.PREVIEW, max_bytes)

    def apply_tags(self, max_bytes: Optional[int] = None) -> TaggingResult:
        return self._process_objects(ProcessingMode.APPLY, max_bytes)

    def _process_objects(
        self,
        mode: ProcessingMode,
        max_bytes: Optional[int] = None,
    ) -> TaggingResult:
        process_max_bytes = max_bytes or self.config.max_bytes
        result = TaggingResult(mode=mode, config=self.config, results={}, summary={})

        try:
            objects = list(self.storage_provider.list_objects())

            if not objects:
                result.summary = {"message": "No objects found in bucket"}
                return result

            for obj_key in objects:
                try:
                    if not self.storage_provider.is_supported_file_type(obj_key):
                        result.add_result(
                            obj_key,
                            create_object_tags_result(
                                existing_tags={}, skipped_reason="Unsupported file type"
                            ),
                        )
                        continue

                    existing_tags = self.storage_provider.get_object_tags(obj_key)
                    content_bytes, file_type = self.storage_provider.get_object_content(
                        obj_key, process_max_bytes
                    )

                    text_content = parse_file_content(content_bytes, file_type)
                    truncated_content = truncate_content(
                        text_content, process_max_bytes
                    )

                    llm_request = LLMRequest(
                        content=truncated_content,
                        tags=self.config.tags,
                        filename=obj_key,
                        custom_prompt_template=self.custom_prompt_template,
                    )

                    llm_response = self.llm_provider.generate_tags(llm_request)
                    tag_keys = list(self.config.tags.keys())
                    proposed_tags = create_tag_mapping(
                        tag_keys, llm_response.tags, self.storage_provider_type
                    )

                    if mode == ProcessingMode.PREVIEW:
                        result.add_result(
                            obj_key,
                            create_object_tags_result(
                                existing_tags=existing_tags, proposed_tags=proposed_tags
                            ),
                        )
                    else:
                        try:
                            final_tags = merge_and_validate_tags(
                                existing_tags,
                                proposed_tags,
                                tag_keys,
                                self.storage_provider_type,
                            )

                            self.storage_provider.set_object_tags(obj_key, final_tags)

                            result.add_result(
                                obj_key,
                                create_object_tags_result(
                                    existing_tags=existing_tags,
                                    proposed_tags=proposed_tags,
                                    applied_tags=final_tags,
                                ),
                            )
                        except Exception as e:
                            result.add_result(
                                obj_key,
                                create_object_tags_result(
                                    existing_tags=existing_tags,
                                    proposed_tags=proposed_tags,
                                    skipped_reason=f"Failed to apply tags: {str(e)}",
                                ),
                            )

                except Exception as e:
                    result.add_result(
                        obj_key,
                        create_object_tags_result(
                            existing_tags={},
                            skipped_reason=f"Processing error: {str(e)}",
                        ),
                    )

            result.summary = result.get_summary_stats()

        except Exception as e:
            raise SmartCloudTagError(f"Failed to process objects: {str(e)}")

        return result

    def get_storage_info(self) -> Dict[str, str]:
        return {
            "provider": self.storage_provider.__class__.__name__,
            "bucket": self.storage_provider.get_bucket_name(),
        }

    def get_llm_info(self) -> Dict[str, str]:
        return {
            "provider": self.llm_provider.__class__.__name__,
            "model": self.llm_provider.get_model_name(),
        }

    def get_tags_info(self) -> Dict[str, Any]:
        tag_info = {}
        for key, allowed_values in self.config.tags.items():
            if allowed_values is None:
                tag_info[key] = "LLM will deduce value"
            else:
                tag_info[key] = f"Allowed values: {allowed_values}"

        return tag_info
