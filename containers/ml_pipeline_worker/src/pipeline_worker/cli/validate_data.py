"""Run Great Expectations checks against a dataset."""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline_worker.artifacts import ensure_local_artifact, upload_local_artifact
from pipeline_worker.datasets import resolve_dataset_extension
from pipeline_worker.ingestion import DataIngestorFactory
from pipeline_worker.validation import (
    DataValidationError,
    ValidationConfig,
    validate_dataframe,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset with Great Expectations.")
    parser.add_argument("--dataset-path", required=True, help="Dataset path or S3 URI to validate.")
    parser.add_argument("--data-format", help="Optional dataset format (csv, json, parquet, ...).")
    parser.add_argument("--target-column", required=True, help="Target column used for training.")
    parser.add_argument(
        "--validation-config",
        default="{}",
        help="Validation overrides as JSON payload.",
    )
    parser.add_argument(
        "--report-uri",
        help="Optional destination (S3/local) for the validation summary JSON.",
    )

    args = parser.parse_args()
    dataset_path = ensure_local_artifact(args.dataset_path)
    try:
        extension = resolve_dataset_extension(dataset_path, args.data_format)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ingestor = DataIngestorFactory.get_data_ingestor(extension)
    dataframe = ingestor.ingest(dataset_path)

    validation_overrides = _parse_validation_config(args.validation_config)
    try:
        validation_config = ValidationConfig.from_raw(
            validation_overrides,
            target_column=args.target_column,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid validation config: {exc}") from exc

    payload_text: str | None = None
    try:
        summary = validate_dataframe(dataframe, validation_config)
        summary["success"] = True
        payload_text = json.dumps(summary, default=_json_default)
        print(payload_text)
    except DataValidationError as exc:
        payload = dict(exc.summary or {})
        payload["success"] = False
        payload.setdefault("results", exc.results)
        payload["failures"] = exc.failures
        payload_text = json.dumps(payload, default=_json_default)
        print(payload_text)
        raise SystemExit(1) from exc
    finally:
        if args.report_uri and payload_text is not None:
            tmp_dir = Path(tempfile.mkdtemp())
            report_path = tmp_dir / "validation_summary.json"
            report_path.write_text(payload_text, encoding="utf-8")
            upload_local_artifact(report_path, args.report_uri)


if __name__ == "__main__":
    main()
