from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple
from ..models import FileType


class StorageProvider(ABC):
    @abstractmethod
    def list_objects(self) -> Iterator[str]:
        pass

    @abstractmethod
    def get_object_content(self, key: str, max_bytes: int) -> Tuple[bytes, FileType]:
        pass

    @abstractmethod
    def get_object_tags(self, key: str) -> Dict[str, str]:
        pass

    @abstractmethod
    def set_object_tags(self, key: str, tags: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def is_supported_file_type(self, key: str) -> bool:
        pass

    @abstractmethod
    def get_bucket_name(self) -> str:
        pass
