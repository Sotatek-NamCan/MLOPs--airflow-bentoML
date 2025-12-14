from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)


def _artifact_cache_root() -> Path:
    override = os.getenv("PIPELINE_ARTIFACT_CACHE_DIR")
    if override:
        return Path(override)
    project_root = os.getenv("PIPELINE_PROJECT_ROOT")
    if project_root:
        return Path(project_root) / "data" / "artifact_cache"
    return Path("/tmp/pipeline_artifacts")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3":
        raise ValueError(f"Unsupported URI scheme for '{uri}'. Expected s3://.")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Incomplete S3 URI: {uri}")
    return bucket, key


def ensure_local_artifact(location: str) -> Path:
    """Return a local path for the given artifact, downloading from S3 if needed."""
    if location.startswith("s3://"):
        bucket, object_key = _parse_s3_uri(location)
        cache_root = _artifact_cache_root()
        destination = cache_root / bucket / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            client = build_storage_client(bucket=bucket)
            client.download(object_key=object_key, destination=destination, bucket=bucket)
        except ObjectStorageConfigurationError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Unable to configure storage client for download.") from exc
        except ObjectStorageOperationError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to download artifact from {location}.") from exc
        return destination

    path = Path(location)
    if not path.exists():
        raise FileNotFoundError(f"Artifact path not found: {location}")
    return path


def upload_local_artifact(local_path: Path | str, destination: str) -> str:
    """Upload a local file to the desired destination (S3 or local path)."""
    src = Path(local_path)
    if not src.is_file():
        raise FileNotFoundError(f"Cannot upload missing artifact: {src}")

    if destination.startswith("s3://"):
        bucket, object_key = _parse_s3_uri(destination)
        try:
            client = build_storage_client(bucket=bucket)
            client.upload(source=src, object_key=object_key, bucket=bucket)
        except ObjectStorageConfigurationError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Unable to configure storage client for upload.") from exc
        except ObjectStorageOperationError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to upload artifact to {destination}.") from exc
        return destination

    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    return str(dest_path)


def copy_artifact(source: str, destination: str) -> str:
    """Copy an artifact between arbitrary locations."""
    local_src = ensure_local_artifact(source)
    return upload_local_artifact(local_src, destination)


def artifact_name(location: str) -> str:
    if location.startswith("s3://"):
        parsed = urlparse(location)
        return Path(parsed.path).name
    return Path(location).name
