"""Train model entrypoint for DockerOperator."""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from pipeline_worker.train_utils import train_and_save_model


def _parse_hyperparameters(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid hyperparameters JSON: {exc}") from exc


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

    model_path = train_and_save_model(
        train_data_path=args.train_data_path,
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
