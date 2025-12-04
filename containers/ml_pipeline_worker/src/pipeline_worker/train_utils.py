from __future__ import annotations

import os
import pickle
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)

_MODEL_ARTIFACT_CACHE_ENV = "MODEL_ARTIFACT_CACHE_DIR"
_MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
_MLFLOW_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
_DEFAULT_TRACKING_URI = "http://localhost:5000"
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


def _artifact_cache_root() -> Path:
    configured = os.getenv(_MODEL_ARTIFACT_CACHE_ENV)
    if configured:
        return Path(configured)
    airflow_home = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))
    return airflow_home / "model_artifacts"


def _parse_s3_uri(uri: str) -> Tuple[Optional[str], Optional[str]]:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3":
        return None, None
    bucket = parsed.netloc or None
    key = parsed.path.lstrip("/") or None
    return bucket, key


def _resolve_local_artifact_dir(bucket: Optional[str], prefix: Optional[str]) -> Path:
    cache_root = _artifact_cache_root()
    parts = []
    if bucket:
        parts.append(bucket)
    if prefix:
        parts.extend([segment for segment in prefix.split("/") if segment])
    return cache_root.joinpath(*parts) if parts else cache_root


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

def select_model(
    model_name: str,
    hyperparameters: Dict[str, Any]
) -> Any:
    """Instantiate the machine learning model corresponding to model_name, with the given hyperparameters."""
    name = model_name.lower()
    if name == "random_forest_classifier":
        return RandomForestClassifier(**hyperparameters)
    elif name == "logistic_regression":
        return LogisticRegression(**hyperparameters)
    elif name == "linear_regression":
        return LinearRegression(**hyperparameters)
    elif name == "random_forest_regressor":
        return RandomForestRegressor(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

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
        # Load data
        X, y = load_train_data(train_data_path, target_column)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Instantiate model
        model = select_model(model_name, hyperparameters)

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
        metric_name: str
        metric_value: float
        if hasattr(model, "predict_proba") or "classifier" in model_name.lower():
            y_pred = model.predict(X_test)
            metric_value = accuracy_score(y_test, y_pred)
            metric_name = "accuracy"
            print(f"[Training] Model {model_name} v{model_version} Accuracy: {metric_value:.4f}")
        else:
            y_pred = model.predict(X_test)
            metric_value = mean_squared_error(y_test, y_pred)
            metric_name = "mse"
            print(f"[Training] Model {model_name} v{model_version} MSE: {metric_value:.4f}")
        mlflow.log_metric(metric_name, metric_value)

        # Determine output destination
        output_bucket: Optional[str] = None
        output_prefix: Optional[str] = None
        local_output_dir: Path
        if output_dir.lower().startswith("s3://"):
            output_bucket, output_prefix = _parse_s3_uri(output_dir)
            if not output_bucket:
                raise ValueError("S3 output_dir must include a bucket name.")
            local_output_dir = _resolve_local_artifact_dir(output_bucket, output_prefix)
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
