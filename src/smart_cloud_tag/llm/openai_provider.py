import os
import openai
from typing import Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base import LLMProvider
from ..models import LLMRequest, LLMResponse
from ..utils import format_llm_prompt, format_custom_llm_prompt, parse_llm_response
from ..exceptions import LLMError


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

        if not self.api_key:
            raise LLMError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
            )

        try:
            self.client = OpenAI(api_key=self.api_key)
            self.client.models.list()

        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {str(e)}")

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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates semantic tags for documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                top_p=0.9,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from OpenAI")

            tag_keys = list(request.tags.keys())
            tag_values = parse_llm_response(content, tag_keys)

            return LLMResponse(
                tags=tag_values,
                confidence=None,
                reasoning=None,
            )

        except openai.RateLimitError:
            raise LLMError("OpenAI rate limit exceeded. Please try again later.")
        except openai.AuthenticationError:
            raise LLMError("OpenAI authentication failed. Please check your API key.")
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise LLMError(f"Unexpected error calling OpenAI: {str(e)}")

    def get_model_name(self) -> str:
        return self.model

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
