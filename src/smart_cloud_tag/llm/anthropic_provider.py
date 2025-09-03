from typing import Dict, List, Optional

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from ..exceptions import LLMError
from ..models import LLMRequest, LLMResponse
from ..utils import format_llm_prompt, format_custom_llm_prompt
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        if not ANTHROPIC_AVAILABLE:
            raise LLMError(
                "Anthropic not available. Install with: pip install smart_cloud_tag[anthropic]"
            )

        self.model = model
        self.api_key = api_key

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise LLMError(f"Failed to initialize Anthropic client: {str(e)}")

    def generate_tags(self, request: LLMRequest) -> LLMResponse:
        try:
            if request.custom_prompt_template:
                prompt = format_custom_llm_prompt(
                    request.custom_prompt_template,
                    request.tags,
                    request.content,
                    request.filename,
                )
            else:
                prompt = format_llm_prompt(
                    request.tags, request.content, request.filename
                )

            response = self.client.messages.create(
                model=self.model,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text if response.content else ""

            from ..utils import parse_llm_response

            tag_keys = list(request.tags.keys())
            tags = parse_llm_response(content, tag_keys)

            return LLMResponse(tags=tags)

        except Exception as e:
            raise LLMError(f"Failed to generate tags with Anthropic: {str(e)}")

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE

    def get_model_name(self) -> str:
        return self.model
