from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable, Sequence

from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator


def stringify_command(value: Sequence | str | None):
    if isinstance(value, (list, tuple)):
        normalized = []
        for part in value:
            if isinstance(part, (dict, list, tuple)):
                normalized.append(json.dumps(part))
            elif part is None:
                normalized.append("")
            else:
                normalized.append(str(part))
        return normalized
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if value is None:
        return ""
    return str(value)


class PipelineDockerOperator(DockerOperator):
    """DockerOperator variant that normalizes command/entrypoint payloads."""

    template_fields = tuple(
        field for field in DockerOperator.template_fields if field != "environment"
    )

    def pre_execute(self, context):
        self.command = stringify_command(self.command)
        if self.entrypoint is not None:
            self.entrypoint = stringify_command(self.entrypoint)
        return super().pre_execute(context)


def env_subset(keys: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            env[key] = value
    return env


CONTAINER_PROJECT_DIR = os.environ.get("ML_PIPELINE_CONTAINER_PROJECT_DIR", "/opt/pipeline")
PIPELINE_IMAGE = os.environ.get("ML_TASK_IMAGE", "mlops/pipeline-worker:local")
DOCKER_URL = os.environ.get("ML_PIPELINE_DOCKER_URL", "unix://var/run/docker.sock")
DOCKER_API_VERSION = os.environ.get("ML_PIPELINE_DOCKER_API_VERSION", "auto")

BASE_ENV = env_subset(
    [
        "OBJECT_STORAGE_BUCKET",
        "OBJECT_STORAGE_ENDPOINT_URL",
        "OBJECT_STORAGE_REGION",
        "OBJECT_STORAGE_ACCESS_KEY",
        "OBJECT_STORAGE_SECRET_KEY",
        "OBJECT_STORAGE_DATASET_BUCKET",
        "OBJECT_STORAGE_DATASET_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
    ]
)
BASE_ENV["PIPELINE_PROJECT_ROOT"] = CONTAINER_PROJECT_DIR
BASE_ENV["PIPELINE_ENV_FILE"] = f"{CONTAINER_PROJECT_DIR}/.env"

BASE_OPERATOR_KWARGS = {
    "image": PIPELINE_IMAGE,
    "docker_url": DOCKER_URL,
    "api_version": DOCKER_API_VERSION,
    "mount_tmp_dir": False,
    "auto_remove": "success",
}


def pipeline_task(
    task_id: str,
    command: Sequence | str,
    *,
    do_xcom_push: bool,
    extra_env: dict[str, str] | None = None,
) -> PipelineDockerOperator:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return PipelineDockerOperator(
        task_id=task_id,
        command=command,
        environment=env,
        do_xcom_push=do_xcom_push,
        **BASE_OPERATOR_KWARGS,
    )


PIPELINE_PARAMS = {
    "data_source": Param("", type="string", description="Location of the dataset to ingest."),
    "data_format": Param("csv", type="string", description="Dataset format (csv, json, parquet...)."),
    "input_schema_version": Param("v1", type="string", description="Schema version (optional hook)."),
    "model_name": Param("random_forest_classifier", type="string", description="Model identifier."),
    "model_version": Param("1", type="string", description="Model version string."),
    "hyperparameters": Param({"n_estimators": 100, "max_depth": 5}, type="object", description="Model hyperparameters."),
    "training_scenario": Param("full_train", type="string", description="Training scenario label."),
    "target_output_path": Param("s3://bucket/output/", type="string", description="Folder/URI for model artifacts."),
    "test_size": Param(0.2, type="number", description="Validation split ratio."),
    "random_state": Param(42, type="integer", description="Random seed used for training."),
    "ingestion_config": Param({}, type="object", description="Optional overrides for ingestion."),
    "target_column": Param("target", type="string", description="Target column to predict."),
    "data_cleaning": Param(
        {
            "drop_columns": [],
            "deduplicate": True,
            "column_order": [],
            "missing_values": {
                "numeric_strategy": "median",
                "categorical_strategy": "mode",
            },
            "outliers": {
                "method": "iqr",
                "iqr_factor": 1.5,
                "columns": [],
            },
            "transformations": [],
        },
        type="object",
        description="Rules for handling outliers, nulls, duplicates, and transformations.",
    ),
    "data_validation": Param(
        {"min_row_count": 10},
        type="object",
        description="Expectation overrides for dataset validation.",
    ),
    "data_profiling": Param(
        {
            "top_value_count": 5,
            "include_percentiles": [0.25, 0.5, 0.75],
        },
        type="object",
        description="Controls for generating profiling statistics per feature.",
    ),
    "data_visualization": Param(
        {
            "max_numeric_charts": 10,
            "max_categorical_charts": 10,
            "plot_format": "png",
        },
        type="object",
        description="Dashboard/visualization generation options.",
    ),
    "hyperparameter_tuning": Param(
        {
            "enabled": False,
            "n_trials": 20,
            "timeout": 600,
            "search_space": {},
        },
        type="object",
        description="Optuna tuning settings (set enabled true to search).",
    ),
}

DEFAULT_DAG_KWARGS = {
    "start_date": datetime(2024, 1, 1),
    "schedule": None,
    "catchup": False,
    "render_template_as_native_obj": True,
    "tags": ["ml", "pipeline", "docker"],
}


def select_params(*keys: str) -> dict[str, Param]:
    return {key: PIPELINE_PARAMS[key] for key in keys}


__all__ = [
    "CONTAINER_PROJECT_DIR",
    "PIPELINE_PARAMS",
    "select_params",
    "DEFAULT_DAG_KWARGS",
    "pipeline_task",
    "PipelineDockerOperator",
]
