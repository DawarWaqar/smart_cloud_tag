import os
import boto3
from typing import Tuple, Dict
from botocore.exceptions import ClientError, NoCredentialsError

from .base import StorageProvider
from ..models import FileType
from ..utils import parse_s3_uri, get_file_type
from ..exceptions import StorageError


class AWSS3Provider(StorageProvider):
    def __init__(self, storage_uri: str):
        self.bucket_name = parse_s3_uri(storage_uri)

        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = os.getenv("AWS_REGION")

        if (
            not self.aws_access_key_id
            or not self.aws_secret_access_key
            or not self.region_name
        ):
            raise StorageError(
                "AWS credentials not provided. Set AWS environment variables."
            )

        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )

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
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if not obj["Key"].endswith("/"):
                            yield obj["Key"]

        except ClientError as e:
            raise StorageError(f"Failed to list objects: {str(e)}")

    def get_object_content(
        self, obj_key: str, max_bytes: int
    ) -> Tuple[bytes, FileType]:
        try:
            file_type = get_file_type(obj_key)
            if not file_type:
                raise StorageError(f"Unsupported file type for {obj_key}")

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
        try:
            response = self.s3_client.get_object_tagging(
                Bucket=self.bucket_name, Key=obj_key
            )

            tags = {}
            for tag in response.get("TagSet", []):
                tags[tag["Key"]] = tag["Value"]

            return tags

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                return {}
            else:
                raise StorageError(f"Failed to get object tags: {str(e)}")

    def set_object_tags(self, obj_key: str, tags: Dict[str, str]) -> None:
        try:
            tag_set = [{"Key": key, "Value": value} for key, value in tags.items()]

            self.s3_client.put_object_tagging(
                Bucket=self.bucket_name, Key=obj_key, Tagging={"TagSet": tag_set}
            )

        except ClientError as e:
            raise StorageError(f"Failed to set object tags: {str(e)}")

    def is_supported_file_type(self, filename: str) -> bool:
        return get_file_type(filename) is not None

    def get_bucket_name(self) -> str:
        return self.bucket_name
