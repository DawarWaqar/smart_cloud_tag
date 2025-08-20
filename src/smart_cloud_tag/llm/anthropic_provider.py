"""
Anthropic Claude LLM provider implementation.
"""

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
    """
    Anthropic Claude LLM provider implementation.
    """

    def __init__(self, model: str, api_key: str):
        """
        Initialize Anthropic provider.

        Args:
            model: Claude model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key

        Raises:
            LLMError: If initialization fails
        """
        if not ANTHROPIC_AVAILABLE:
            raise LLMError(
                "Anthropic not available. Install with: pip install smart-cloud[anthropic]"
            )

        self.model = model
        self.api_key = api_key

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise LLMError(f"Failed to initialize Anthropic client: {str(e)}")

    def generate_tags(self, request: LLMRequest) -> LLMResponse:
        """
        Generate tags using Anthropic Claude.

        Args:
            request: LLM request with tags, content, and filename

        Returns:
            LLM response with generated tags

        Raises:
            LLMError: If generation fails
        """
        try:
            # Format the prompt
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

            # Send to Anthropic API
            response = self.client.messages.create(
                model=self.model,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract content from response
            content = response.content[0].text if response.content else ""

            # Parse the response into tag values
            from ..utils import parse_llm_response

            tag_keys = list(request.tags.keys())
            tags = parse_llm_response(content, tag_keys)

            return LLMResponse(tags=tags)

        except Exception as e:
            raise LLMError(f"Failed to generate tags with Anthropic: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if Anthropic provider is available.

        Returns:
            True if available
        """
        return ANTHROPIC_AVAILABLE

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            Model name
        """
        return self.model
