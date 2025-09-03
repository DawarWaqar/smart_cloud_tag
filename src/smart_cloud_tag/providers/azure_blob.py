import os
from typing import Tuple, Dict
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

from .base import StorageProvider
from ..models import FileType
from ..utils import get_file_type
from ..exceptions import StorageError


class AzureBlobProvider(StorageProvider):
    def __init__(self, storage_uri: str, connection_string: str):
        self.container_name = self._parse_azure_uri(storage_uri)
        self.connection_string = connection_string

        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )

            self.container_client.get_container_properties()

        except AzureError as e:
            raise StorageError(f"Failed to initialize Azure Blob client: {str(e)}")

    def _parse_azure_uri(self, uri: str) -> str:
        if not uri.startswith("az://"):
            raise ValueError("Azure URI must start with 'az://'")

        parts = uri[5:].split("/", 1)
        container = parts[0]

        if not container:
            raise ValueError("Invalid Azure URI: missing container name")

        return container

    def list_objects(self):
        try:
            for blob in self.container_client.list_blobs():
                if not blob.name.endswith("/"):
                    yield blob.name
        except AzureError as e:
            raise StorageError(f"Failed to list blobs: {str(e)}")

    def get_object_content(
        self, blob_name: str, max_bytes: int
    ) -> Tuple[bytes, FileType]:
        try:
            blob_client = self.container_client.get_blob_client(blob_name)

            properties = blob_client.get_blob_properties()

            if properties.size <= max_bytes:
                download_stream = blob_client.download_blob(max_concurrency=1)
            else:
                download_stream = blob_client.download_blob(
                    max_concurrency=1, offset=0, length=max_bytes
                )

            content = download_stream.readall()

            file_type = get_file_type(blob_name)
            if not file_type:
                raise StorageError(f"Unsupported file type: {blob_name}")

            return content, file_type

        except AzureError as e:
            raise StorageError(f"Failed to get blob content: {str(e)}")

    def get_object_tags(self, blob_name: str) -> Dict[str, str]:
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            tags = blob_client.get_blob_tags()
            return tags or {}

        except AzureError as e:
            raise StorageError(f"Failed to get blob tags: {str(e)}")

    def set_object_tags(self, blob_name: str, tags: Dict[str, str]) -> None:
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.set_blob_tags(tags)

        except AzureError as e:
            raise StorageError(f"Failed to set blob tags: {str(e)}")

    def is_supported_file_type(self, filename: str) -> bool:
        return get_file_type(filename) is not None

    def get_bucket_name(self) -> str:
        return self.container_name
