from .base import StorageProvider

try:
    from .aws_s3 import AWSS3Provider

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    AWSS3Provider = None

try:
    from .azure_blob import AzureBlobProvider

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    AzureBlobProvider = None

try:
    from .gcs_provider import GCSProvider

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    GCSProvider = None

__all__ = ["StorageProvider", "AWSS3Provider", "AzureBlobProvider", "GCSProvider"]
