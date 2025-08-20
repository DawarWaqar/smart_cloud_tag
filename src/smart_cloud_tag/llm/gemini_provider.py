"""
Google Gemini LLM provider implementation.
"""

from typing import Dict, List, Optional

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from ..exceptions import LLMError
from ..models import LLMRequest, LLMResponse
from ..utils import format_llm_prompt, format_custom_llm_prompt
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider implementation.
    """

    def __init__(self, model: str, api_key: str):
        """
        Initialize Gemini provider.

        Args:
            model: Gemini model name (e.g., "gemini-1.5-pro")
            api_key: Google API key

        Raises:
            LLMError: If initialization fails
        """
        if not GEMINI_AVAILABLE:
            raise LLMError(
                "Gemini not available. Install with: pip install smart-cloud[gemini]"
            )

        self.model = model
        self.api_key = api_key

        try:
            genai.configure(api_key=api_key)
            self.model_instance = genai.GenerativeModel(model)
        except Exception as e:
            raise LLMError(f"Failed to initialize Gemini client: {str(e)}")

    def generate_tags(self, request: LLMRequest) -> LLMResponse:
        """
        Generate tags using Google Gemini.

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

            # Send to Gemini API
            response = self.model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                ),
            )

            # Extract content from response
            content = response.text if response.text else ""

            # Parse the response into tag values
            from ..utils import parse_llm_response

            tag_keys = list(request.tags.keys())
            tags = parse_llm_response(content, tag_keys)

            return LLMResponse(tags=tags)

        except Exception as e:
            raise LLMError(f"Failed to generate tags with Gemini: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if Gemini provider is available.

        Returns:
            True if available
        """
        return GEMINI_AVAILABLE

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            Model name
        """
        return self.model
