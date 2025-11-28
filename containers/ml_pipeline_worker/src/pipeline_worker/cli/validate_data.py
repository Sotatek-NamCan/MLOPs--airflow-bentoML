"""Run Great Expectations checks against a dataset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline_worker.ingestion import DataIngestorFactory
from pipeline_worker.validation import (
    DataValidationError,
    ValidationConfig,
    validate_dataframe,
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

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).resolve()
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
        print(json.dumps(summary, default=_json_default))
    except DataValidationError as exc:
        payload = dict(exc.summary or {})
        payload["success"] = False
        payload.setdefault("results", exc.results)
        payload["failures"] = exc.failures
        print(json.dumps(payload, default=_json_default))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
