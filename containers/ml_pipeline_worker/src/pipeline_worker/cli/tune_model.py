"""Hyperparameter tuning entrypoint using Optuna."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from pipeline_worker.artifacts import ensure_local_artifact, upload_local_artifact
from pipeline_worker.hpo import (
    parse_json_payload,
    perform_hyperparameter_search,
)
from pipeline_worker.train_utils import load_train_data


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = path.read_text(encoding="utf-8")
    return json.loads(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning.")
    parser.add_argument("--train-data-path", required=True, help="Dataset path or S3 URI.")
    parser.add_argument("--target-column", required=True, help="Target column name.")
    parser.add_argument("--model-name", required=True, help="Model identifier.")
    parser.add_argument(
        "--base-hyperparameters",
        default="{}",
        help="Baseline hyperparameters (JSON).",
    )
    parser.add_argument(
        "--tuning-config",
        default="{}",
        help="Tuning configuration JSON (enabled, n_trials, timeout, search_space).",
    )
    parser.add_argument(
        "--output-uri",
        required=True,
        help="Destination for the best hyperparameters JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    base_params = parse_json_payload(args.base_hyperparameters)
    tuning_cfg = parse_json_payload(args.tuning_config)

    enabled = bool(tuning_cfg.get("enabled", True))
    n_trials = int(tuning_cfg.get("n_trials", 20))
    timeout = tuning_cfg.get("timeout")
    timeout_val = int(timeout) if timeout else None
    search_space = tuning_cfg.get("search_space", {})

    if not enabled or not search_space:
        best_params = base_params
    else:
        dataset_local = ensure_local_artifact(args.train_data_path)
        features, target = load_train_data(str(dataset_local), args.target_column)
        best_params = perform_hyperparameter_search(
            model_name=args.model_name,
            base_hyperparameters=base_params,
            search_space_overrides=search_space,
            features=features,
            target=target,
            n_trials=n_trials,
            timeout=timeout_val,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    tmp_path = Path("/tmp/best_params.json")
    tmp_path.write_text(json.dumps(best_params), encoding="utf-8")
    final_location = upload_local_artifact(tmp_path, args.output_uri)
    print(final_location)


if __name__ == "__main__":
    main()
