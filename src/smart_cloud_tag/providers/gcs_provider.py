"""
Google Cloud Storage provider implementation.

This provider uses GCS metadata for tagging since GCS doesn't support native tags like S3.
"""

import re
from typing import Dict, Iterator, Optional, Tuple
from urllib.parse import urlparse

try:
    from google.cloud import storage
    from google.cloud.exceptions import GoogleCloudError
    from google.cloud.storage.blob import Blob
    from google.cloud.storage.bucket import Bucket

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    GoogleCloudError = Exception
    Blob = None
    Bucket = None

from ..exceptions import StorageError
from ..models import FileType
from ..utils import get_file_type
from .base import StorageProvider


class GCSProvider(StorageProvider):
    """
    Google Cloud Storage provider implementation.

    Uses GCS metadata for tagging since GCS doesn't support native tags.
    Metadata has different limits: 64 pairs, 128 chars for keys, 1024 chars for values.
    """

    def __init__(self, storage_uri: str, credentials_path: Optional[str] = None):
        """
        Initialize GCS provider.

        Args:
            storage_uri: GCS URI in format 'gs://bucket-name'
            credentials_path: Optional path to service account JSON file

        Raises:
            StorageError: If initialization fails
        """
        if not GCS_AVAILABLE:
            raise StorageError(
                "Google Cloud Storage not available. "
                "Install with: pip install 'smart-cloud[gcp]'"
            )

        self.bucket_name = self._parse_gcs_uri(storage_uri)

        try:
            if credentials_path:
                self.client = storage.Client.from_service_account_json(credentials_path)
            else:
                # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
                self.client = storage.Client()

            self.bucket = self.client.bucket(self.bucket_name)
            # Verify bucket exists and is accessible
            self.bucket.reload()

        except GoogleCloudError as e:
            raise StorageError(f"Failed to initialize GCS client: {str(e)}")
        except Exception as e:
            raise StorageError(f"Unexpected error initializing GCS client: {str(e)}")

    def _parse_gcs_uri(self, uri: str) -> str:
        """
        Parse GCS URI to extract bucket name.

        Args:
            uri: GCS URI (e.g., 'gs://my-bucket')

        Returns:
            Bucket name

        Raises:
            StorageError: If URI format is invalid
        """
        if not uri.startswith("gs://"):
            raise StorageError(
                f"Invalid GCS URI format: {uri}. Must start with 'gs://'"
            )

        # Remove 'gs://' prefix
        path = uri[5:]

        if "/" in path:
            bucket_name = path.split("/", 1)[0]
        else:
            bucket_name = path

        if not bucket_name:
            raise StorageError(f"Invalid GCS URI: missing bucket name in {uri}")

        return bucket_name

    def list_objects(self):
        """
        List all objects in the GCS bucket recursively.

        Yields:
            Object names

        Raises:
            StorageError: If listing fails
        """
        try:
            for blob in self.bucket.list_blobs():
                # Skip directory markers (blobs ending with /)
                if not blob.name.endswith("/"):
                    yield blob.name
        except GoogleCloudError as e:
            raise StorageError(f"Failed to list objects: {str(e)}")

    def get_object_content(
        self, obj_name: str, max_bytes: int
    ) -> Tuple[bytes, FileType]:
        """
        Get object content from GCS.

        Args:
            obj_name: Object name
            max_bytes: Maximum bytes to read

        Returns:
            Tuple of (content_bytes, file_type)

        Raises:
            StorageError: If retrieval fails
        """
        try:
            blob = self.bucket.blob(obj_name)

            # Get file type
            file_type = get_file_type(obj_name)
            if not file_type:
                raise StorageError(f"Unsupported file type for {obj_name}")

            # Download content
            if max_bytes > 0:
                # Download only the first max_bytes
                content = blob.download_as_bytes(start=0, end=max_bytes-1)
            else:
                # Download entire object
                content = blob.download_as_bytes()

            return content, file_type

        except GoogleCloudError as e:
            raise StorageError(f"Failed to get object content: {str(e)}")

    def get_object_tags(self, obj_name: str) -> Dict[str, str]:
        """
        Get object metadata as tags from GCS.

        Args:
            obj_name: Object name

        Returns:
            Dictionary of metadata tags

        Raises:
            StorageError: If retrieval fails
        """
        try:
            blob = self.bucket.blob(obj_name)
            blob.reload()  # Ensure metadata is loaded
            
            # Convert metadata to tags
            return blob.metadata or {}
            
        except GoogleCloudError as e:
            raise StorageError(f"Failed to get object tags: {str(e)}")

    def set_object_tags(self, obj_name: str, tags: Dict[str, str]) -> None:
        """
        Set object metadata as tags in GCS.

        Args:
            obj_name: Object name
            tags: Dictionary of tags to set

        Raises:
            StorageError: If setting fails
        """
        try:
            blob = self.bucket.blob(obj_name)
            
            # Set metadata
            blob.metadata = tags
            blob.patch()
            
        except GoogleCloudError as e:
            raise StorageError(f"Failed to set object tags: {str(e)}")

    def is_supported_file_type(self, filename: str) -> bool:
        """
        Check if file type is supported.

        Args:
            filename: Name of the file

        Returns:
            True if file type is supported
        """
        return get_file_type(filename) is not None

    def get_bucket_name(self) -> str:
        """
        Get the bucket name.

        Returns:
            Bucket name
        """
        return self.bucket_name
