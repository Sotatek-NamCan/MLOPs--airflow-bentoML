"""Finalize model artifact location by copying/uploading to the target destination."""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)


def _is_s3_uri(value: str) -> bool:
    return value.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str | None, str | None]:
    parsed = urlparse(uri)
    bucket = parsed.netloc or None
    key = parsed.path.lstrip("/") or None
    return bucket, key


def _download_from_s3(uri: str, work_dir: Path) -> Path:
    bucket, key = _parse_s3_uri(uri)
    if not bucket or not key:
        raise SystemExit(f"Invalid S3 URI: {uri}")
    destination = work_dir / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = build_storage_client(bucket=bucket)
        client.download(object_key=key, destination=destination, bucket=bucket)
    except ObjectStorageConfigurationError as exc:
        raise SystemExit("Failed to configure storage client for model download.") from exc
    except ObjectStorageOperationError as exc:
        raise SystemExit(f"Failed to download model artifact from {uri}.") from exc
    return destination


def _upload_to_s3(local_path: Path, uri: str) -> None:
    bucket, key = _parse_s3_uri(uri)
    if not bucket or not key:
        raise SystemExit(f"Invalid S3 URI: {uri}")
    try:
        client = build_storage_client(bucket=bucket)
        client.upload(source=local_path, object_key=key, bucket=bucket)
    except ObjectStorageConfigurationError as exc:
        raise SystemExit("Failed to configure storage client for model upload.") from exc
    except ObjectStorageOperationError as exc:
        raise SystemExit(f"Failed to upload model artifact to {uri}.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy model artifact to the final destination.")
    parser.add_argument("--model-artifact-path", required=True, help="Path or URI produced by the training step.")
    parser.add_argument("--target-output-path", required=True, help="Base directory or URI for the final artifact.")

    args = parser.parse_args()
    artifact_name = Path(urlparse(args.model_artifact_path).path or args.model_artifact_path).name
    base = args.target_output_path.rstrip("/")
    saved_location = f"{base}/{artifact_name}" if base else artifact_name

    temp_dir = None
    source_path: Path
    if _is_s3_uri(args.model_artifact_path):
        temp_dir = Path(tempfile.mkdtemp(prefix="save_results_"))
        source_path = _download_from_s3(args.model_artifact_path, temp_dir)
    else:
        source_path = Path(args.model_artifact_path).resolve()

    print(
        f"Saving model artifact from {args.model_artifact_path} to {saved_location}",
        file=sys.stderr,
    )

    if _is_s3_uri(saved_location):
        _upload_to_s3(source_path, saved_location)
    else:
        destination = Path(saved_location).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(saved_location)


if __name__ == "__main__":
    main()
