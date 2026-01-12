"""Build visualization dashboards for dataset exploration."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline_worker.artifacts import ensure_local_artifact, upload_local_artifact
from pipeline_worker.datasets import resolve_dataset_extension
from pipeline_worker.ingestion import DataIngestorFactory
from pipeline_worker.profiling import render_visualizations


def _parse_viz_config(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid visualization config JSON: {exc}") from exc


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization assets for datasets.")
    parser.add_argument("--dataset-path", required=True, help="Local path or S3 URI to the dataset.")
    parser.add_argument("--data-format", help="Optional dataset format override.")
    parser.add_argument("--visualization-config", default="{}", help="Visualization config JSON.")
    parser.add_argument(
        "--visualization-uri",
        required=True,
        help="Destination (S3/local) where the visualization archive will be uploaded.",
    )

    args = parser.parse_args()
    dataset_path = ensure_local_artifact(args.dataset_path)
    try:
        extension = resolve_dataset_extension(dataset_path, args.data_format)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ingestor = DataIngestorFactory.get_data_ingestor(extension)
    dataframe = ingestor.ingest(dataset_path)

    tmp_dir = Path(tempfile.mkdtemp())
    viz_dir = tmp_dir / "visualizations"
    try:
        summary = render_visualizations(dataframe, viz_dir, _parse_viz_config(args.visualization_config))
    except ValueError as exc:
        raise SystemExit(f"Invalid visualization config: {exc}") from exc
    summary_path = viz_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    archive_base = tmp_dir / "visualization_artifacts"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=viz_dir)
    stored_uri = upload_local_artifact(Path(archive_path), args.visualization_uri)
    print(f"Visualization assets uploaded to: {stored_uri}")


if __name__ == "__main__":
    main()
