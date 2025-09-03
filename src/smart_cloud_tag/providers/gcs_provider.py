import re
from typing import Dict, Iterator, Optional, Tuple

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
    def __init__(self, storage_uri: str, credentials_path: Optional[str] = None):
        if not GCS_AVAILABLE:
            raise StorageError(
                "Google Cloud Storage not available. "
                "Install with: pip install 'smart_cloud_tag[gcp]'"
            )

        self.bucket_name = self._parse_gcs_uri(storage_uri)

        try:
            if credentials_path:
                self.client = storage.Client.from_service_account_json(credentials_path)
            else:
                self.client = storage.Client()

            self.bucket = self.client.bucket(self.bucket_name)
            self.bucket.reload()

        except GoogleCloudError as e:
            raise StorageError(f"Failed to initialize GCS client: {str(e)}")
        except Exception as e:
            raise StorageError(f"Unexpected error initializing GCS client: {str(e)}")

    def _parse_gcs_uri(self, uri: str) -> str:
        if not uri.startswith("gs://"):
            raise StorageError(
                f"Invalid GCS URI format: {uri}. Must start with 'gs://'"
            )

        path = uri[5:]

        if "/" in path:
            bucket_name = path.split("/", 1)[0]
        else:
            bucket_name = path

        if not bucket_name:
            raise StorageError(f"Invalid GCS URI: missing bucket name in {uri}")

        return bucket_name

    def list_objects(self):
        try:
            for blob in self.bucket.list_blobs():
                if not blob.name.endswith("/"):
                    yield blob.name
        except GoogleCloudError as e:
            raise StorageError(f"Failed to list objects: {str(e)}")

    def get_object_content(
        self, obj_name: str, max_bytes: int
    ) -> Tuple[bytes, FileType]:
        try:
            blob = self.bucket.blob(obj_name)

            file_type = get_file_type(obj_name)
            if not file_type:
                raise StorageError(f"Unsupported file type for {obj_name}")

            if max_bytes > 0:
                content = blob.download_as_bytes(start=0, end=max_bytes - 1)
            else:
                content = blob.download_as_bytes()

            return content, file_type

        except GoogleCloudError as e:
            raise StorageError(f"Failed to get object content: {str(e)}")

    def get_object_tags(self, obj_name: str) -> Dict[str, str]:
        try:
            blob = self.bucket.blob(obj_name)
            blob.reload()

            return blob.metadata or {}

        except GoogleCloudError as e:
            raise StorageError(f"Failed to get object tags: {str(e)}")

    def set_object_tags(self, obj_name: str, tags: Dict[str, str]) -> None:
        try:
            blob = self.bucket.blob(obj_name)

            blob.metadata = tags
            blob.patch()

        except GoogleCloudError as e:
            raise StorageError(f"Failed to set object tags: {str(e)}")

    def is_supported_file_type(self, filename: str) -> bool:
        return get_file_type(filename) is not None

    def get_bucket_name(self) -> str:
        return self.bucket_name
