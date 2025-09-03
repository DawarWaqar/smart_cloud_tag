from abc import ABC, abstractmethod
from ..models import LLMRequest, LLMResponse


class LLMProvider(ABC):
    @abstractmethod
    def generate_tags(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass
