"""Container-friendly wrapper around :func:`pipeline_worker.ingestion.ingest_data`."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)

from pipeline_worker.ingestion import ingest_data


def _default_project_root() -> Path:
    candidate = (
        os.environ.get("PIPELINE_PROJECT_ROOT")
        or os.environ.get("AIRFLOW_PROJ_DIR")
        or os.environ.get("AIRFLOW_HOME")
    )
    if candidate:
        return Path(candidate).resolve()
    return Path.cwd()


def _parse_json(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid JSON payload: {exc}") from exc


def _build_ingestion_config(args: argparse.Namespace, ingestion_cfg: Dict[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "object_storage": ingestion_cfg.get("object_storage", {}) or {},
        "file_path": ingestion_cfg.get("file_path"),
        "file_extension": ingestion_cfg.get("file_extension"),
        "zip_extract_dir": ingestion_cfg.get("zip_extract_dir"),
    }

    if not config["file_extension"] and args.data_format:
        normalized_format = args.data_format.lstrip(".")
        config["file_extension"] = f".{normalized_format}"

    if not config["file_path"]:
        config["file_path"] = args.data_source

    data_source = (args.data_source or "").strip()
    if data_source.lower().startswith("s3://"):
        # Split s3://bucket/key to populate object storage information automatically
        path_no_scheme = data_source[len("s3://") :]
        bucket, _, object_key = path_no_scheme.partition("/")
        if bucket:
            config.setdefault("object_storage", {})
            config["object_storage"].update(
                {
                    "enabled": True,
                    "bucket": bucket,
                    "object_key": object_key or "",
                }
            )

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest dataset and return local path.")
    parser.add_argument("--data-source", required=True, help="Input URI or local path.")
    parser.add_argument("--data-format", default="csv", help="Input data format (csv, json, ...).")
    parser.add_argument("--input-schema-version", help="Schema version (unused hook).")
    parser.add_argument(
        "--ingestion-config",
        default="{}",
        help="Additional ingestion config as JSON.",
    )
    parser.add_argument(
        "--project-root",
        default=str(_default_project_root()),
        help="Path to repository root for resolving cache directories.",
    )
    parser.add_argument(
        "--upload-bucket",
        help="S3 bucket where the ingested dataset should be uploaded.",
    )
    parser.add_argument(
        "--upload-object-key",
        help="S3 object key for the ingested dataset. Requires --upload-bucket or OBJECT_STORAGE_BUCKET.",
    )

    args = parser.parse_args()
    ingestion_cfg = _parse_json(args.ingestion_config)
    config = _build_ingestion_config(args, ingestion_cfg)
    project_root = Path(args.project_root).resolve()

    _, local_path = ingest_data(config=config, project_root=project_root)
    upload_key = args.upload_object_key

    if upload_key:
        upload_bucket = args.upload_bucket or os.getenv("OBJECT_STORAGE_BUCKET")
        if not upload_bucket:
            raise SystemExit(
                "upload-object-key was provided but no upload bucket is configured. "
                "Set --upload-bucket or OBJECT_STORAGE_BUCKET."
            )
        try:
            client = build_storage_client(bucket=upload_bucket)
            client.upload(source=local_path, object_key=upload_key, bucket=upload_bucket)
        except ObjectStorageConfigurationError as exc:
            raise SystemExit(
                "Failed to configure object storage client for dataset upload."
            ) from exc
        except ObjectStorageOperationError as exc:
            raise SystemExit(
                f"Failed to upload dataset to s3://{upload_bucket}/{upload_key}."
            ) from exc
        print(f"s3://{upload_bucket}/{upload_key}")
    else:
        print(str(local_path))


if __name__ == "__main__":
    main()
