"""Container-friendly wrapper around :func:`pipeline_worker.ingestion.ingest_data`."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from pipeline_worker.ingestion import ingest_data
from pipeline_worker.artifacts import upload_local_artifact


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
        "file_path": ingestion_cfg.get("file_path"),
        "file_extension": ingestion_cfg.get("file_extension"),
        "zip_extract_dir": ingestion_cfg.get("zip_extract_dir"),
        "cache_dir": ingestion_cfg.get("cache_dir"),
    }

    if not config["file_extension"] and args.data_format:
        normalized_format = args.data_format.lstrip(".")
        config["file_extension"] = f".{normalized_format}"

    if not config["file_path"]:
        config["file_path"] = args.data_source

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
        "--output-uri",
        help="Optional S3 URI where the ingested dataset will be stored.",
    )

    args = parser.parse_args()
    ingestion_cfg = _parse_json(args.ingestion_config)
    config = _build_ingestion_config(args, ingestion_cfg)
    project_root = Path(args.project_root).resolve()

    _, local_path = ingest_data(config=config, project_root=project_root)
    if args.output_uri:
        stored_location = upload_local_artifact(local_path, args.output_uri)
        print(stored_location)
    else:
        print(str(local_path))


if __name__ == "__main__":
    main()
