from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ProcessingMode(str, Enum):
    PREVIEW = "preview"
    APPLY = "apply"


class FileType(str, Enum):
    TXT = "txt"
    MD = "md"
    JSON = "json"
    CSV = "csv"


class TaggingConfig(BaseModel):
    llm_model: str = Field(..., description="LLM model name")
    storage_uri: str = Field(..., description="Storage URI")
    tags: Dict[str, Optional[List[str]]] = Field(
        ..., description="Tag keys and allowed values"
    )
    max_bytes: int = Field(
        default=5000, description="Maximum bytes to read from each file"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        if len(v) >= 10:
            raise ValueError("Tags must be < 10")
        if len(v) == 0:
            raise ValueError("tags cannot be empty")
        return v

    @field_validator("max_bytes")
    @classmethod
    def validate_max_bytes(cls, v):
        if v <= 0:
            raise ValueError("max_bytes must be positive")
        return v


class ObjectTags(BaseModel):
    existing: Dict[str, str] = Field(default_factory=dict)
    proposed: Optional[Dict[str, str]] = None
    applied: Optional[Dict[str, str]] = None
    skipped_reason: Optional[str] = None


class TaggingResult(BaseModel):
    mode: ProcessingMode
    config: TaggingConfig
    results: Dict[str, ObjectTags] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)

    def add_result(self, uri: str, tags: ObjectTags) -> None:
        self.results[uri] = tags

    def get_summary_stats(self) -> Dict[str, Any]:
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
    content: str = Field(..., description="File content to analyze")
    tags: Dict[str, Optional[List[str]]] = Field(
        ..., description="Tag keys and allowed values"
    )
    filename: str = Field(..., description="Name of the file being analyzed")
    custom_prompt_template: Optional[str] = None


class LLMResponse(BaseModel):
    tags: List[str] = Field(..., description="Generated tag values in order")
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
