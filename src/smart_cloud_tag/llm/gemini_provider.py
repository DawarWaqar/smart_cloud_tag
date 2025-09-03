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
    def __init__(self, model: str, api_key: str):
        if not GEMINI_AVAILABLE:
            raise LLMError(
                "Gemini not available. Install with: pip install smart_cloud_tag[gemini]"
            )

        self.model = model
        self.api_key = api_key

        try:
            genai.configure(api_key=api_key)
            self.model_instance = genai.GenerativeModel(model)
        except Exception as e:
            raise LLMError(f"Failed to initialize Gemini client: {str(e)}")

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

            response = self.model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                ),
            )

            content = response.text if response.text else ""

            from ..utils import parse_llm_response

            tag_keys = list(request.tags.keys())
            tags = parse_llm_response(content, tag_keys)

            return LLMResponse(tags=tags)

        except Exception as e:
            raise LLMError(f"Failed to generate tags with Gemini: {str(e)}")

    def is_available(self) -> bool:
        return GEMINI_AVAILABLE

    def get_model_name(self) -> str:
        return self.model
