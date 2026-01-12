"""Generate profiling statistics for a dataset."""
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
from pipeline_worker.profiling import build_profile_summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback
            return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _parse_profile_config(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid profile config JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Create profiling statistics for datasets.")
    parser.add_argument("--dataset-path", required=True, help="Local path or S3 URI to the dataset.")
    parser.add_argument("--data-format", help="Optional dataset format.")
    parser.add_argument("--profile-config", default="{}", help="Profiling configuration as JSON payload.")
    parser.add_argument(
        "--profile-report-uri",
        required=True,
        help="Destination for the profiling report (S3 or local path).",
    )

    args = parser.parse_args()
    dataset_path = ensure_local_artifact(args.dataset_path)
    try:
        extension = resolve_dataset_extension(dataset_path, args.data_format)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ingestor = DataIngestorFactory.get_data_ingestor(extension)
    dataframe = ingestor.ingest(dataset_path)
    try:
        profile_summary = build_profile_summary(dataframe, _parse_profile_config(args.profile_config))
    except ValueError as exc:
        raise SystemExit(f"Invalid profile config: {exc}") from exc

    payload = json.dumps(profile_summary, indent=2, default=_json_default)
    print(payload)

    tmp_dir = Path(tempfile.mkdtemp())
    report_path = tmp_dir / "profile_summary.json"
    report_path.write_text(payload, encoding="utf-8")
    upload_local_artifact(report_path, args.profile_report_uri)


if __name__ == "__main__":
    main()
