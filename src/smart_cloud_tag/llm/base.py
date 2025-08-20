"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement.
    """

    @abstractmethod
    def generate_tags(self, request: LLMRequest) -> LLMResponse:
        """
        Generate tags using the LLM.

        Args:
            request: LLMRequest containing content and tags

        Returns:
            LLMResponse with generated tags

        Raises:
            LLMError: If LLM operation fails
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the LLM model being used.

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and configured.

        Returns:
            True if available
        """
        pass
