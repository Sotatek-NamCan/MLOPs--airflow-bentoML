"""Clean a dataset by removing outliers, filling nulls, and applying transformations."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from pipeline_worker.artifacts import ensure_local_artifact, upload_local_artifact
from pipeline_worker.datasets import resolve_dataset_extension
from pipeline_worker.ingestion import DataIngestorFactory
from pipeline_worker.preprocessing import CleaningConfig, clean_dataframe


def _parse_cleaning_config(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid cleaning config JSON: {exc}") from exc


def _write_dataframe(df, destination: Path, extension: str) -> None:
    normalized = (extension or "").lower()
    if normalized in ("", ".csv"):
        df.to_csv(destination, index=False)
        return
    if normalized == ".parquet":
        df.to_parquet(destination, index=False)
        return
    if normalized in {".json", ".ndjson"}:
        df.to_json(destination, orient="records", indent=2)
        return
    raise SystemExit(f"Unsupported output format: {extension}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean datasets before validation.")
    parser.add_argument("--dataset-path", required=True, help="Local path or S3 URI to the ingested dataset.")
    parser.add_argument("--data-format", help="Optional data format override.")
    parser.add_argument("--cleaning-config", default="{}", help="Cleaning configuration as JSON payload.")
    parser.add_argument("--output-uri", required=True, help="Destination for the cleaned dataset (local or S3).")
    parser.add_argument(
        "--report-uri",
        help="Optional destination for a JSON summary describing the cleaning operations.",
    )

    args = parser.parse_args()
    dataset_path = ensure_local_artifact(args.dataset_path)
    try:
        extension = resolve_dataset_extension(dataset_path, args.data_format)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ingestor = DataIngestorFactory.get_data_ingestor(extension)
    dataframe = ingestor.ingest(dataset_path)

    cleaning_config = CleaningConfig.from_raw(_parse_cleaning_config(args.cleaning_config))
    cleaned_df, summary = clean_dataframe(dataframe, cleaning_config)

    tmp_dir = Path(tempfile.mkdtemp())
    output_extension = extension or ".csv"
    output_path = tmp_dir / f"cleaned_dataset{output_extension}"
    _write_dataframe(cleaned_df, output_path, output_extension)
    stored_location = upload_local_artifact(output_path, args.output_uri)
    print(f"Cleaned dataset stored at: {stored_location}")

    if args.report_uri:
        summary_path = tmp_dir / "cleaning_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        upload_local_artifact(summary_path, args.report_uri)


if __name__ == "__main__":
    main()
