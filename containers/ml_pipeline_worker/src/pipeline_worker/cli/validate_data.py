"""Run Great Expectations checks against a dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import pandas as pd

from pipeline_worker.ingestion import DataIngestorFactory
from pipeline_worker.validation import (
    DataValidationError,
    ValidationConfig,
    validate_dataframe,
)
from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)


def _resolve_extension(dataset_path: Path, data_format: str | None) -> str:
    if data_format:
        normalized = data_format.strip()
        normalized = normalized.lstrip(".")
        return f".{normalized}" if normalized else ""
    suffix = dataset_path.suffix
    if suffix:
        return suffix
    raise SystemExit(
        "Unable to determine dataset format automatically. Please set --data-format."
    )


def _parse_validation_config(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid validation config JSON: {exc}") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _is_s3_uri(value: str) -> bool:
    return value.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str | None, str | None]:
    parsed = urlparse(uri)
    bucket = parsed.netloc or None
    key = parsed.path.lstrip("/") or None
    return bucket, key


def _materialize_dataset(dataset_uri: str, work_dir: Path) -> Path:
    if not _is_s3_uri(dataset_uri):
        return Path(dataset_uri).resolve()
    bucket, key = _parse_s3_uri(dataset_uri)
    if not bucket or not key:
        raise SystemExit(f"Invalid S3 dataset URI: {dataset_uri}")
    destination = work_dir / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = build_storage_client(bucket=bucket)
        client.download(object_key=key, destination=destination, bucket=bucket)
    except ObjectStorageConfigurationError as exc:
        raise SystemExit(
            "Failed to configure object storage client for dataset download."
        ) from exc
    except ObjectStorageOperationError as exc:
        raise SystemExit(
            f"Failed to download dataset from s3://{bucket}/{key}."
        ) from exc
    return destination


def _write_report_payload(payload: str, explicit_path: str | None) -> Path:
    if explicit_path:
        dest = Path(explicit_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload, encoding="utf-8")
        return dest
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8")
    try:
        tmp.write(payload)
    finally:
        tmp.close()
    return Path(tmp.name)


def _upload_report(path: Path, bucket: str, object_key: str) -> str:
    try:
        client = build_storage_client(bucket=bucket)
        client.upload(source=path, object_key=object_key, bucket=bucket)
    except ObjectStorageConfigurationError as exc:
        raise SystemExit(
            "Failed to configure object storage client for validation report upload."
        ) from exc
    except ObjectStorageOperationError as exc:
        raise SystemExit(
            f"Failed to upload validation report to s3://{bucket}/{object_key}."
        ) from exc
    return f"s3://{bucket}/{object_key}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset with Great Expectations.")
    parser.add_argument("--dataset-path", required=True, help="Local dataset path to validate.")
    parser.add_argument("--data-format", help="Optional dataset format (csv, json, parquet, ...).")
    parser.add_argument("--target-column", required=True, help="Target column used for training.")
    parser.add_argument(
        "--validation-config",
        default="{}",
        help="Validation overrides as JSON payload.",
    )
    parser.add_argument(
        "--report-output-path",
        help="Optional local file path where the validation summary JSON will be written.",
    )
    parser.add_argument(
        "--report-upload-bucket",
        help="S3 bucket for uploading the validation summary.",
    )
    parser.add_argument(
        "--report-upload-object-key",
        help="S3 object key for uploading the validation summary.",
    )

    args = parser.parse_args()
    report_upload_key = args.report_upload_object_key
    report_upload_bucket = args.report_upload_bucket or os.getenv("OBJECT_STORAGE_BUCKET")

    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="validate_data_") as tmp_dir:
        dataset_path = _materialize_dataset(args.dataset_path, Path(tmp_dir))
        extension = _resolve_extension(dataset_path, args.data_format)
        ingestor = DataIngestorFactory.get_data_ingestor(extension)
        dataframe = ingestor.ingest(dataset_path)

        validation_overrides = _parse_validation_config(args.validation_config)
        validation_config = ValidationConfig.from_raw(
            validation_overrides,
            target_column=args.target_column,
        )

        try:
            summary = validate_dataframe(dataframe, validation_config)
            summary["success"] = True
        except DataValidationError as exc:
            summary = dict(exc.summary or {})
            summary["success"] = False
            summary.setdefault("results", exc.results)
            summary["failures"] = exc.failures
            exit_code = 1

    payload = json.dumps(summary, default=_json_default)
    report_path = _write_report_payload(payload, args.report_output_path)
    uploaded_uri = None
    if report_upload_key:
        if not report_upload_bucket:
            raise SystemExit(
                "report-upload-object-key provided but no bucket configured. "
                "Set --report-upload-bucket or OBJECT_STORAGE_BUCKET."
            )
        uploaded_uri = _upload_report(report_path, report_upload_bucket, report_upload_key)

    print(payload)
    if uploaded_uri:
        print(f"[Validation] Summary uploaded to {uploaded_uri}", file=sys.stderr)

    if not args.report_output_path:
        try:
            report_path.unlink()
        except OSError:
            pass

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
