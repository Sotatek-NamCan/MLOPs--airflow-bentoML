"""Train model entrypoint for containerized execution."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)
from pipeline_worker.train_utils import train_and_save_model


def _parse_hyperparameters(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid hyperparameters JSON: {exc}") from exc


def _is_s3_uri(value: str) -> bool:
    return value.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str | None, str | None]:
    parsed = urlparse(uri)
    bucket = parsed.netloc or None
    key = parsed.path.lstrip("/") or None
    return bucket, key


def _materialize_training_data(dataset_uri: str, work_dir: Path) -> Path:
    if not _is_s3_uri(dataset_uri):
        return Path(dataset_uri).resolve()
    bucket, key = _parse_s3_uri(dataset_uri)
    if not bucket or not key:
        raise SystemExit(f"Invalid S3 training data URI: {dataset_uri}")
    destination = work_dir / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = build_storage_client(bucket=bucket)
        client.download(object_key=key, destination=destination, bucket=bucket)
    except ObjectStorageConfigurationError as exc:
        raise SystemExit(
            "Failed to configure object storage client for training data download."
        ) from exc
    except ObjectStorageOperationError as exc:
        raise SystemExit(
            f"Failed to download training data from s3://{bucket}/{key}."
        ) from exc
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model and emit artifact path.")
    parser.add_argument("--train-data-path", required=True, help="Local dataset path.")
    parser.add_argument("--target-column", required=True, help="Target column name.")
    parser.add_argument("--model-name", required=True, help="Model identifier.")
    parser.add_argument("--model-version", required=True, help="Model version string.")
    parser.add_argument(
        "--hyperparameters",
        default="{}",
        help="Model hyperparameters JSON string.",
    )
    parser.add_argument(
        "--training-scenario",
        default="full_train",
        help="Training scenario label.",
    )
    parser.add_argument(
        "--target-output-path",
        required=True,
        help="Base directory or S3 URI to store artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    hyperparameters = _parse_hyperparameters(args.hyperparameters)

    base_output = args.target_output_path.rstrip("/")
    output_dir = f"{base_output}/models/{args.model_name}_v{args.model_version}/"

    with tempfile.TemporaryDirectory(prefix="train_data_") as tmp_dir:
        local_train_path = _materialize_training_data(args.train_data_path, Path(tmp_dir))
        model_path = train_and_save_model(
            train_data_path=str(local_train_path),
            target_column=args.target_column,
            model_name=args.model_name,
            model_version=args.model_version,
            hyperparameters=hyperparameters,
            training_scenario=args.training_scenario,
            output_dir=output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    print(str(model_path))


if __name__ == "__main__":
    main()
