"""
LLM providers for the smart_cloud_tag package.
"""

from .base import LLMProvider

# Try to import OpenAI provider (default)
try:
    from .openai_provider import OpenAIProvider

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIProvider = None

# Try to import Anthropic provider
try:
    from .anthropic_provider import AnthropicProvider

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AnthropicProvider = None

# Try to import Gemini provider
try:
    from .gemini_provider import GeminiProvider

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiProvider = None

__all__ = ["LLMProvider", "OpenAIProvider", "AnthropicProvider", "GeminiProvider"]
