"""
Data models for the smart_cloud_tag package.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ProcessingMode(str, Enum):
    """Processing modes for tag operations."""

    PREVIEW = "preview"
    APPLY = "apply"


class FileType(str, Enum):
    """Supported file types."""

    TXT = "txt"
    MD = "md"
    JSON = "json"
    CSV = "csv"


class TaggingConfig(BaseModel):
    """Configuration for the tagging process."""

    llm_model: str = Field(..., description="LLM model name (e.g., 'gpt-4o-mini')")
    storage_uri: str = Field(
        ...,
        description="Storage URI (e.g., s3://bucket or gs://bucket)",
    )
    tags: Dict[str, Optional[List[str]]] = Field(
        ...,
        description="Dictionary of tag keys and their allowed values (null means LLM deduces)",
    )
    max_bytes: int = Field(
        default=5000, description="Maximum bytes to read from each file"
    )

    @validator("tags")
    def validate_tags(cls, v):
        """Validate tags for AWS S3 compliance."""
        if len(v) >= 10:
            raise ValueError("Tags must be < 10.")
        if len(v) == 0:
            raise ValueError("tags cannot be empty.")
        return v

    @validator("max_bytes")
    def validate_max_bytes(cls, v):
        """Validate max_bytes is positive."""
        if v <= 0:
            raise ValueError("max_bytes must be positive.")
        return v


class ObjectTags(BaseModel):
    """Tags for a single object."""

    existing: Dict[str, str] = Field(default_factory=dict, description="Existing tags")
    proposed: Optional[Dict[str, str]] = Field(
        None, description="Proposed tags from LLM"
    )
    applied: Optional[Dict[str, str]] = Field(None, description="Actually applied tags")
    skipped_reason: Optional[str] = Field(
        None, description="Reason if object was skipped"
    )


class TaggingResult(BaseModel):
    """Result of a tagging operation."""

    mode: ProcessingMode
    config: TaggingConfig
    results: Dict[str, ObjectTags] = Field(
        default_factory=dict, description="Results by object URI"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics"
    )

    def add_result(self, uri: str, tags: ObjectTags) -> None:
        """Add a result for a specific object."""
        self.results[uri] = tags

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        total_objects = len(self.results)
        processed = sum(
            1 for tags in self.results.values() if tags.proposed is not None
        )
        skipped = sum(
            1 for tags in self.results.values() if tags.skipped_reason is not None
        )
        applied = sum(1 for tags in self.results.values() if tags.applied is not None)

        return {
            "total_objects": total_objects,
            "processed": processed,
            "skipped": skipped,
            "applied": applied,
            "success_rate": processed / total_objects if total_objects > 0 else 0,
        }


class LLMRequest(BaseModel):
    """Request to the LLM for tag generation."""

    content: str = Field(..., description="File content to analyze")
    tags: Dict[str, Optional[List[str]]] = Field(
        ..., description="Tag keys and their allowed values"
    )

    filename: str = Field(..., description="Name of the file being analyzed")
    custom_prompt_template: Optional[str] = Field(
        None, description="Custom prompt template (optional)"
    )


class LLMResponse(BaseModel):
    """Response from the LLM with generated tags."""

    tags: List[str] = Field(..., description="Generated tag values in order")
    confidence: Optional[float] = Field(
        None, description="Confidence score if available"
    )
    reasoning: Optional[str] = Field(None, description="Reasoning for tag choices")
