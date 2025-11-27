"""S3-backed object storage helpers."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # pip install python-dotenv


def _load_dotenv() -> None:
    """Best-effort load of a .env file for local development."""
    candidates = []
    env_override = os.getenv("PIPELINE_ENV_FILE")
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")
    for candidate in candidates:
        if candidate and candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break


_load_dotenv()


class ObjectStorageError(RuntimeError):
    """Base exception for storage issues."""


class ObjectStorageConfigurationError(ObjectStorageError):
    """Raised when the storage client cannot be configured."""


class ObjectStorageOperationError(ObjectStorageError):
    """Raised when a storage operation fails."""


class ObjectStorageClient(ABC):
    """Simple contract for interacting with object storage."""

    @abstractmethod
    def download(
        self, object_key: str, destination: Path | str, bucket: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload(
        self, source: Path | str, object_key: str, bucket: Optional[str] = None
    ) -> None:
        raise NotImplementedError

class S3ObjectStorageClient(ObjectStorageClient):
    """Backed by boto3 for interacting with S3-compatible storage."""

    def __init__(
        self,
        *,
        bucket: Optional[str],
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise ObjectStorageConfigurationError(
                "boto3 is required for S3 object storage support. Install boto3."
            ) from exc

        session_kwargs: dict[str, str] = {}
        if endpoint_url:
            session_kwargs["endpoint_url"] = endpoint_url
        if region_name:
            session_kwargs["region_name"] = region_name
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key

        self._bucket = bucket
        self._s3_client = boto3.client("s3", **session_kwargs)

    def download(
        self, object_key: str, destination: Path | str, bucket: Optional[str] = None
    ) -> None:
        bucket_name = bucket or self._bucket
        if not bucket_name:
            raise ObjectStorageConfigurationError(
                "No S3 bucket specified. Provide OBJECT_STORAGE_BUCKET or pass 'bucket' to download()."
            )
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3_client.download_file(bucket_name, object_key, str(dest_path))
        except Exception as exc:
            raise ObjectStorageOperationError(
                f"Failed to download '{object_key}' from bucket '{bucket_name}'."
            ) from exc

    def upload(
        self, source: Path | str, object_key: str, bucket: Optional[str] = None
    ) -> None:
        bucket_name = bucket or self._bucket
        if not bucket_name:
            raise ObjectStorageConfigurationError(
                "No S3 bucket specified. Provide OBJECT_STORAGE_BUCKET or pass 'bucket' to upload()."
            )
        src_path = Path(source)
        if not src_path.is_file():
            raise ObjectStorageOperationError(
                f"Cannot upload '{src_path}': file does not exist."
            )
        try:
            self._s3_client.upload_file(str(src_path), bucket_name, object_key)
        except Exception as exc:
            raise ObjectStorageOperationError(
                f"Failed to upload '{src_path}' to bucket '{bucket_name}' as '{object_key}'."
            ) from exc

def build_storage_client(*, bucket: Optional[str] = None) -> ObjectStorageClient:
    """Return an S3 client configured via environment variables or .env."""
    bucket_val = bucket or os.getenv("OBJECT_STORAGE_BUCKET")
    endpoint_url = os.getenv("OBJECT_STORAGE_ENDPOINT_URL")
    region_name = os.getenv("OBJECT_STORAGE_REGION")
    access_key = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
    secret_key = os.getenv("OBJECT_STORAGE_SECRET_KEY")

    if not bucket_val:
        raise ObjectStorageConfigurationError(
            "S3 bucket not specified. Set OBJECT_STORAGE_BUCKET in environment or pass bucket param."
        )

    return S3ObjectStorageClient(
        bucket=bucket_val,
        endpoint_url=endpoint_url,
        region_name=region_name,
        access_key=access_key,
        secret_key=secret_key,
    )
