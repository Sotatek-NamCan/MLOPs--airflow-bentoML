from __future__ import annotations

import difflib
import math
import os
import pickle
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import train_test_split

from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)

_MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
_MLFLOW_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
_DEFAULT_TRACKING_URI = "http://18.138.227.55:5000/"
_DEFAULT_EXPERIMENT = "airflow_ml_pipeline"

_OBJECT_STORAGE_ACCESS_KEY_ENV = "OBJECT_STORAGE_ACCESS_KEY"
_OBJECT_STORAGE_SECRET_KEY_ENV = "OBJECT_STORAGE_SECRET_KEY"
_OBJECT_STORAGE_REGION_ENV = "OBJECT_STORAGE_REGION"
_OBJECT_STORAGE_ENDPOINT_ENV = "OBJECT_STORAGE_ENDPOINT_URL"

_AWS_ACCESS_KEY_ENV = "AWS_ACCESS_KEY_ID"
_AWS_SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
_AWS_REGION_ENV = "AWS_REGION"
_AWS_DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
_MLFLOW_S3_ENDPOINT_ENV = "MLFLOW_S3_ENDPOINT_URL"

_MODEL_REGISTRY = {
    "random_forest_classifier": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
}


def _parse_s3_uri(uri: str) -> Tuple[Optional[str], Optional[str]]:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3":
        return None, None
    bucket = parsed.netloc or None
    key = parsed.path.lstrip("/") or None
    return bucket, key


def _join_s3_key(prefix: Optional[str], file_name: str) -> str:
    if prefix:
        base = PurePosixPath(prefix.strip("/"))
        key = base / file_name
    else:
        key = PurePosixPath(file_name)
    return key.as_posix()


def _propagate_storage_env_for_mlflow() -> None:
    """Ensure MLflow/boto3 see the S3 credentials we keep in OBJECT_STORAGE_* env vars."""

    def _set_env(target_var: str, value: Optional[str]) -> None:
        if value:
            os.environ[target_var] = value

    access_key = os.getenv(_OBJECT_STORAGE_ACCESS_KEY_ENV)
    secret_key = os.getenv(_OBJECT_STORAGE_SECRET_KEY_ENV)
    region = os.getenv(_OBJECT_STORAGE_REGION_ENV)
    endpoint = os.getenv(_OBJECT_STORAGE_ENDPOINT_ENV)

    _set_env(_AWS_ACCESS_KEY_ENV, access_key)
    _set_env(_AWS_SECRET_KEY_ENV, secret_key)
    _set_env(_AWS_REGION_ENV, region)
    _set_env(_AWS_DEFAULT_REGION_ENV, region)
    _set_env(_MLFLOW_S3_ENDPOINT_ENV, endpoint)


def _configure_mlflow() -> None:
    _propagate_storage_env_for_mlflow()
    tracking_uri = os.getenv(_MLFLOW_TRACKING_URI_ENV, _DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv(_MLFLOW_EXPERIMENT_ENV, _DEFAULT_EXPERIMENT)
    mlflow.set_experiment(experiment_name)


def _log_mlflow_params(
    *,
    model_name: str,
    model_version: str,
    training_scenario: str,
    hyperparameters: Dict[str, Any],
    data_source: str,
) -> None:
    base_params = {
        "model_name": model_name,
        "model_version": model_version,
        "training_scenario": training_scenario,
        "data_source": data_source,
    }
    for key, value in base_params.items():
        mlflow.log_param(key, value)
    for key, value in hyperparameters.items():
        param_key = f"hp_{key}"
        mlflow.log_param(param_key, value)


def load_train_data(path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load training dataset from a local path and split into features/target."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training data path not found: {path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".parquet":
        df = pd.read_parquet(p)
    elif ext == ".json":
        df = pd.read_json(p, orient="records", encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file extension for training data: {ext}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    feature_df = df.drop(columns=[target_column])
    numeric_features = feature_df.select_dtypes(include=["number", "bool"])
    dropped_columns = [col for col in feature_df.columns if col not in numeric_features.columns]
    if dropped_columns:
        print(f"[Training] Dropping non-numeric feature columns: {dropped_columns}")
    if numeric_features.empty:
        raise ValueError(
            "No numeric features remain after dropping non-numeric columns. "
            "Encode categorical features before training."
        )

    X = numeric_features
    y = df[target_column]
    return X, y


def _get_model_class(model_name: str):
    name = model_name.lower()
    model_class = _MODEL_REGISTRY.get(name)
    if not model_class:
        supported = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model_name: {model_name}. Supported models: {supported}")
    return model_class


def _validate_hyperparameters(
    *,
    model_name: str,
    model_class,
    hyperparameters: Dict[str, Any],
) -> None:
    if not hyperparameters:
        return
    valid_params = set(model_class().get_params().keys())
    unknown = sorted(key for key in hyperparameters.keys() if key not in valid_params)
    if not unknown:
        return

    suggestions = []
    for key in unknown:
        matches = difflib.get_close_matches(key, valid_params, n=3, cutoff=0.6)
        if matches:
            suggestions.append(f"{key} -> {', '.join(matches)}")
    suggestion_text = f" Did you mean: {'; '.join(suggestions)}?" if suggestions else ""

    valid_preview = ", ".join(sorted(valid_params)[:15])
    remainder = len(valid_params) - 15
    more_text = f", ... (+{remainder} more)" if remainder > 0 else ""
    raise ValueError(
        "Unsupported hyperparameter name(s) for "
        f"{model_name}: {', '.join(unknown)}."
        f"{suggestion_text} Valid keys include: {valid_preview}{more_text}"
    )


def _log_model_spec(
    *,
    model_name: str,
    model: Any,
    hyperparameters: Dict[str, Any],
) -> None:
    print("[Training] Model specification")
    print(f"[Training] - name: {model_name}")
    print(f"[Training] - estimator: {model.__class__.__name__}")
    if not hyperparameters:
        print("[Training] - hyperparameters: (defaults)")
        return
    print("[Training] - hyperparameters:")
    for key in sorted(hyperparameters):
        print(f"[Training]   - {key}: {hyperparameters[key]!r}")


def select_model(
    model_name: str,
    hyperparameters: Dict[str, Any]
) -> Any:
    """Instantiate the machine learning model corresponding to model_name, with the given hyperparameters."""
    model_class = _get_model_class(model_name)
    _validate_hyperparameters(
        model_name=model_name,
        model_class=model_class,
        hyperparameters=hyperparameters,
    )
    return model_class(**hyperparameters)

def train_and_save_model(
    train_data_path: str,
    target_column: str,
    model_name: str,
    model_version: str,
    hyperparameters: Dict[str, Any],
    training_scenario: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """
    Full workflow for training:
    1. Load train dataset from train_data_path.
    2. Split into train/test (based on test_size and random_state).
    3. Instantiate model with model_name + hyperparameters.
    4. Fit model.
    5. Evaluate basic metric.
    6. Save model artifact into output_dir.
    Returns the path to the saved model file.
    """
    _configure_mlflow()

    with mlflow.start_run(run_name=f"{model_name}_v{model_version}"):
        temp_artifact_dir = None
        # Load data
        try:
            X, y = load_train_data(train_data_path, target_column)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Instantiate model
            model = select_model(model_name, hyperparameters)
            _log_model_spec(
                model_name=model_name,
                model=model,
                hyperparameters=hyperparameters,
            )

            _log_mlflow_params(
                model_name=model_name,
                model_version=model_version,
                training_scenario=training_scenario,
                hyperparameters=hyperparameters,
                data_source=train_data_path,
            )

            # Train
            model.fit(X_train, y_train)

            # Evaluate

            #Try catch & log the metrics error.
            y_pred = model.predict(X_test)
            metrics: Dict[str, float] = {}
            is_classifier = hasattr(model, "predict_proba") or "classifier" in model_name.lower()
            if is_classifier:
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
                metrics["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                print(
                    "[Training] Classification metrics "
                    + ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
                )
            else:
                mse = mean_squared_error(y_test, y_pred)
                metrics["mse"] = mse
                metrics["rmse"] = math.sqrt(mse)
                metrics["mae"] = mean_absolute_error(y_test, y_pred)
                metrics["r2"] = r2_score(y_test, y_pred)
                metrics["explained_variance"] = explained_variance_score(y_test, y_pred)
                print(
                    "[Training] Regression metrics "
                    + ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
                )
            for name, value in metrics.items():
                mlflow.log_metric(name, float(value))

            # Determine output destination
            output_bucket: Optional[str] = None
            output_prefix: Optional[str] = None
            local_output_dir: Path
            if output_dir.lower().startswith("s3://"):
                output_bucket, output_prefix = _parse_s3_uri(output_dir)
                if not output_bucket:
                    raise ValueError("S3 output_dir must include a bucket name.")
                temp_artifact_dir = tempfile.TemporaryDirectory()
                local_output_dir = Path(temp_artifact_dir.name)
            else:
                local_output_dir = Path(output_dir)

            # Save model artifact locally first
            local_output_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{model_name}_v{model_version}.pkl"
            local_model_path = local_output_dir / file_name
            with open(local_model_path, "wb") as f:
                pickle.dump(model, f)

            artifact_path = str(local_model_path)
            print(f"[Training] Saved model artifact to {artifact_path}")

            # Upload to S3 if requested
            if output_bucket:
                object_key = _join_s3_key(output_prefix, file_name)
                try:
                    client = build_storage_client(bucket=output_bucket)
                    client.upload(source=local_model_path, object_key=object_key, bucket=output_bucket)
                except ObjectStorageConfigurationError as exc:
                    raise RuntimeError(
                        "Failed to configure object storage client for model upload."
                    ) from exc
                except ObjectStorageOperationError as exc:
                    raise RuntimeError(
                        f"Failed to upload model artifact to s3://{output_bucket}/{object_key}."
                    ) from exc
                artifact_path = f"s3://{output_bucket}/{object_key}"
                print(f"[Training] Uploaded model artifact to {artifact_path}")
                mlflow.log_param("artifact_path", artifact_path)

            if not output_bucket:
                mlflow.log_param("artifact_path", artifact_path)

            return artifact_path
        finally:
            if temp_artifact_dir is not None:
                temp_artifact_dir.cleanup()
