"""
Azure Blob Storage provider implementation.
"""

import os
from typing import List, Tuple, Dict, Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

from .base import StorageProvider
from ..models import FileType
from ..utils import get_file_type
from ..exceptions import StorageError


class AzureBlobProvider(StorageProvider):
    """
    Azure Blob Storage provider implementation.
    """

    def __init__(
        self,
        storage_uri: str,
        connection_string: str,
    ):
        """
        Initialize Azure Blob provider.

        Args:
            storage_uri: Storage URI with container (e.g., "az://container")
            connection_string: Azure connection string
        """
        # Parse Azure URI
        self.container_name = self._parse_azure_uri(storage_uri)
        self.connection_string = connection_string

        # Initialize Azure client
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )

            # Test connection
            self.container_client.get_container_properties()

        except AzureError as e:
            raise StorageError(f"Failed to initialize Azure Blob client: {str(e)}")

    def _parse_azure_uri(self, uri: str) -> str:
        """Parse Azure URI to get container name."""
        if not uri.startswith("az://"):
            raise ValueError("Azure URI must start with 'az://'")

        parts = uri[5:].split("/", 1)  # Remove "az://" and split
        container = parts[0]

        if not container:
            raise ValueError("Invalid Azure URI: missing container name")

        return container

    def list_objects(self):
        """List all blobs in the container recursively."""
        try:
            for blob in self.container_client.list_blobs():
                # Skip directory markers (blobs ending with /)
                if not blob.name.endswith("/"):
                    yield blob.name
        except AzureError as e:
            raise StorageError(f"Failed to list blobs: {str(e)}")

    def get_object_content(
        self, blob_name: str, max_bytes: int
    ) -> Tuple[bytes, FileType]:
        """Get blob content."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)

            # Download content - first get properties to check size
            properties = blob_client.get_blob_properties()

            # If blob is smaller than max_bytes, download all; otherwise download just max_bytes
            if properties.size <= max_bytes:
                download_stream = blob_client.download_blob(max_concurrency=1)
            else:
                download_stream = blob_client.download_blob(
                    max_concurrency=1, offset=0, length=max_bytes
                )

            content = download_stream.readall()

            # Get file type
            file_type = get_file_type(blob_name)
            if not file_type:
                raise StorageError(f"Unsupported file type: {blob_name}")

            return content, file_type

        except AzureError as e:
            raise StorageError(f"Failed to get blob content: {str(e)}")

    def get_object_tags(self, blob_name: str) -> Dict[str, str]:
        """Get blob tags."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            
            # Azure Blob Storage doesn't have native tags like S3
            # Return empty dict for now
            return {}
            
        except AzureError as e:
            raise StorageError(f"Failed to get blob tags: {str(e)}")

    def set_object_tags(self, blob_name: str, tags: Dict[str, str]) -> None:
        """Set blob tags."""
        try:
            # Azure Blob Storage doesn't have native tags like S3
            # This is a no-op for now
            pass
            
        except AzureError as e:
            raise StorageError(f"Failed to set blob tags: {str(e)}")

    def is_supported_file_type(self, filename: str) -> bool:
        """Check if file type is supported."""
        return get_file_type(filename) is not None

    def get_bucket_name(self) -> str:
        """Get the container name."""
        return self.container_name
