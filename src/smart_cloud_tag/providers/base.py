"""
Abstract base class for storage providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Iterator, Optional, Tuple
from ..models import FileType


class StorageProvider(ABC):
    """
    Abstract base class for storage providers.

    This class defines the interface that all storage providers must implement.
    """

    @abstractmethod
    def list_objects(self) -> Iterator[str]:
        """
        List all objects in the storage bucket recursively.

        Yields:
            Object keys/URIs
        """
        pass

    @abstractmethod
    def get_object_content(self, key: str, max_bytes: int) -> Tuple[bytes, FileType]:
        """
        Get object content and determine file type.

        Args:
            key: Object key/URI
            max_bytes: Maximum bytes to read

        Returns:
            Tuple of (content_bytes, file_type)

        Raises:
            StorageError: If operation fails
        """
        pass

    @abstractmethod
    def get_object_tags(self, key: str) -> Dict[str, str]:
        """
        Get existing tags for an object.

        Args:
            key: Object key/URI

        Returns:
            Dictionary of existing tags

        Raises:
            StorageError: If operation fails
        """
        pass

    @abstractmethod
    def set_object_tags(self, key: str, tags: Dict[str, str]) -> None:
        """
        Set tags for an object.

        Args:
            key: Object key/URI
            tags: Dictionary of tags to set

        Raises:
            StorageError: If operation fails
        """
        pass

    @abstractmethod
    def is_supported_file_type(self, key: str) -> bool:
        """
        Check if file type is supported for processing.

        Args:
            key: Object key/URI

        Returns:
            True if file type is supported
        """
        pass

    @abstractmethod
    def get_bucket_name(self) -> str:
        """
        Get the bucket name.

        Returns:
            Bucket name
        """
        pass
