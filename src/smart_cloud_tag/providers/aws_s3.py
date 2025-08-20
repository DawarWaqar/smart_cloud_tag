"""
AWS S3 storage provider implementation.
"""

import os
import boto3
from typing import List, Tuple, Dict, Optional
from botocore.exceptions import ClientError, NoCredentialsError

from .base import StorageProvider
from ..models import FileType
from ..utils import parse_s3_uri, get_file_type
from ..exceptions import StorageError


class AWSS3Provider(StorageProvider):
    """
    AWS S3 storage provider implementation.
    """

    def __init__(
        self,
        storage_uri: str,
    ):
        """
        Initialize AWS S3 provider.

        Args:
            storage_uri: Storage URI (e.g., s3://bucket/)
        """
        # Parse S3 URI to get bucket name
        self.bucket_name = parse_s3_uri(storage_uri)

        # Get credentials from parameters or environment variables
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = os.getenv("AWS_REGION")

        # Validate required credentials
        if (
            not self.aws_access_key_id
            or not self.aws_secret_access_key
            or not self.region_name
        ):
            raise StorageError(
                "AWS credentials not provided. Set AWS environment variables."
            )

        # Initialize S3 client
        try:
            # Use explicit credentials
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )

            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)

        except NoCredentialsError:
            raise StorageError(
                "AWS credentials not found. Please check your configuration."
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket_name}' does not exist.")
            elif error_code == "403":
                raise StorageError(
                    f"Access denied to bucket '{self.bucket_name}'. Check your IAM permissions."
                )
            else:
                raise StorageError(f"AWS S3 error: {str(e)}")
        except Exception as e:
            raise StorageError(f"Failed to initialize S3 client: {str(e)}")

    def list_objects(self):
        """
        List all objects in the S3 bucket recursively.

        Yields:
            Object keys

        Raises:
            StorageError: If listing fails
        """
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Skip directory markers (objects ending with /)
                        if not obj["Key"].endswith("/"):
                            yield obj["Key"]

        except ClientError as e:
            raise StorageError(f"Failed to list objects: {str(e)}")

    def get_object_content(
        self, obj_key: str, max_bytes: int
    ) -> Tuple[bytes, FileType]:
        """
        Get object content from S3.

        Args:
            obj_key: Object key
            max_bytes: Maximum bytes to read

        Returns:
            Tuple of (content_bytes, file_type)

        Raises:
            StorageError: If retrieval fails
        """
        try:
            # Get file type
            file_type = get_file_type(obj_key)
            if not file_type:
                raise StorageError(f"Unsupported file type for {obj_key}")

            # Get object content with range request if needed
            if max_bytes > 0:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=obj_key, Range=f"bytes=0-{max_bytes-1}"
                )
            else:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=obj_key
                )

            content = response["Body"].read()
            return content, file_type

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise StorageError(
                    f"Object '{obj_key}' not found in bucket '{self.bucket_name}'."
                )
            else:
                raise StorageError(f"Failed to get object content: {str(e)}")

    def get_object_tags(self, obj_key: str) -> Dict[str, str]:
        """
        Get tags for an S3 object.

        Args:
            obj_key: Object key

        Returns:
            Dictionary of tags

        Raises:
            StorageError: If retrieval fails
        """
        try:
            response = self.s3_client.get_object_tagging(
                Bucket=self.bucket_name, Key=obj_key
            )

            # Convert tag list to dictionary
            tags = {}
            for tag in response.get("TagSet", []):
                tags[tag["Key"]] = tag["Value"]

            return tags

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                # Object doesn't exist, return empty tags
                return {}
            else:
                raise StorageError(f"Failed to get object tags: {str(e)}")

    def set_object_tags(self, obj_key: str, tags: Dict[str, str]) -> None:
        """
        Set tags for an S3 object.

        Args:
            obj_key: Object key
            tags: Dictionary of tags to set

        Raises:
            StorageError: If setting fails
        """
        try:
            # Convert dictionary to tag list format
            tag_set = [{"Key": key, "Value": value} for key, value in tags.items()]

            self.s3_client.put_object_tagging(
                Bucket=self.bucket_name, Key=obj_key, Tagging={"TagSet": tag_set}
            )

        except ClientError as e:
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
