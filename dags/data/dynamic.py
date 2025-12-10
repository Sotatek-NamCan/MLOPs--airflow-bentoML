from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator


def _literal_template(value: str) -> str:
    """Wrap a literal string so Airflow templating doesn't treat it as a file path."""
    return f"{{{{ {json.dumps(value)} }}}}"


def _stringify_command(value: Sequence | str | None):
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


def _env_subset(keys: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            env[key] = value
    return env


def _split_csv_env(var_name: str) -> list[str]:
    raw = os.getenv(var_name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


CONTAINER_PROJECT_DIR = os.environ.get("ML_PIPELINE_CONTAINER_PROJECT_DIR", "/srv/pipeline")
AWS_CONN_ID = os.environ.get("ML_PIPELINE_AWS_CONN_ID", "aws_default")
REGION_NAME = os.environ.get("ML_PIPELINE_AWS_REGION", "eu-central-1")
CLUSTER_NAME = os.environ.get("ML_PIPELINE_ECS_CLUSTER", "apache-airflow-worker-cluster")
TASK_DEFINITION = os.environ.get("ML_PIPELINE_ECS_TASK_DEFINITION", "airflow-worker")
LAUNCH_TYPE = os.environ.get("ML_PIPELINE_ECS_LAUNCH_TYPE", "FARGATE")
CONTAINER_NAME = os.environ.get("ML_PIPELINE_ECS_CONTAINER_NAME", "airflow-worker")
SUBNETS = _split_csv_env("ML_PIPELINE_ECS_SUBNETS") or ["subnet-"]
SECURITY_GROUPS = _split_csv_env("ML_PIPELINE_ECS_SECURITY_GROUPS") or ["sg-"]


NETWORK_CONFIGURATION = {
    "awsvpcConfiguration": {
        "subnets": SUBNETS,
        "securityGroups": SECURITY_GROUPS,
        "assignPublicIp": "ENABLED",
    }
}

BASE_ECS_OPERATOR_KWARGS: dict[str, object] = {
    "aws_conn_id": AWS_CONN_ID,
    "cluster": CLUSTER_NAME,
    "task_definition": TASK_DEFINITION,
    "launch_type": LAUNCH_TYPE,
    "wait_for_completion": True,
}
BASE_ECS_OPERATOR_KWARGS["network_configuration"] = NETWORK_CONFIGURATION
if REGION_NAME:
    BASE_ECS_OPERATOR_KWARGS["region_name"] = REGION_NAME

DEFAULT_DAG_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


BASE_ENV = _env_subset(
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
        "MODEL_ARTIFACT_CACHE_DIR",
    ]
)
BASE_ENV["PIPELINE_PROJECT_ROOT"] = _literal_template(CONTAINER_PROJECT_DIR)
BASE_ENV["PIPELINE_ENV_FILE"] = _literal_template(f"{CONTAINER_PROJECT_DIR}/.env")

ARTIFACT_BUCKET = os.environ.get("ML_PIPELINE_ARTIFACT_BUCKET") or os.environ.get("OBJECT_STORAGE_BUCKET")
if not ARTIFACT_BUCKET:
    raise RuntimeError(
        "Set OBJECT_STORAGE_BUCKET or ML_PIPELINE_ARTIFACT_BUCKET to enable S3 hand-off between tasks."
    )
artifact_prefix = os.environ.get("ML_PIPELINE_ARTIFACT_PREFIX", "ml-pipeline-runs").strip("/")
if not artifact_prefix:
    artifact_prefix = "ml-pipeline-runs"
RUN_ID_SAFE = "{{ run_id | replace(':', '_') }}"
RUN_S3_PREFIX = f"{artifact_prefix}/{{{{ ds_nodash }}}}/{RUN_ID_SAFE}"
DATASET_OBJECT_KEY = f"{RUN_S3_PREFIX}/datasets/ingested.{{{{ params.data_format }}}}"
DATASET_S3_URI = f"s3://{ARTIFACT_BUCKET}/{DATASET_OBJECT_KEY}"
VALIDATION_REPORT_KEY = f"{RUN_S3_PREFIX}/validation/report.json"
TRAINING_OUTPUT_BASE = f"s3://{ARTIFACT_BUCKET}/{RUN_S3_PREFIX}"
MODEL_ARTIFACT_URI = (
    f"{TRAINING_OUTPUT_BASE}/models/{{{{ params.model_name }}}}_v{{{{ params.model_version }}}}/"
    f"{{{{ params.model_name }}}}_v{{{{ params.model_version }}}}.pkl"
)


def _ecs_environment(extra_env: dict[str, str] | None = None) -> list[dict[str, str]]:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return [{"name": key, "value": value} for key, value in env.items() if value is not None]


def _build_overrides(
    command: Sequence | str,
    *,
    extra_env: dict[str, str] | None = None,
) -> dict:
    container_override: dict[str, object] = {
        "name": CONTAINER_NAME,
        "command": _stringify_command(command),
    }
    environment = _ecs_environment(extra_env)
    if environment:
        container_override["environment"] = environment
    return {"containerOverrides": [container_override]}


def _pipeline_task(
    task_id: str,
    command: Sequence | str,
    *,
    xcom_push: bool,
    extra_env: dict[str, str] | None = None,
) -> EcsRunTaskOperator:
    overrides = _build_overrides(command, extra_env=extra_env)
    return EcsRunTaskOperator(
        task_id=task_id,
        overrides=overrides,
        do_xcom_push=xcom_push,
        **BASE_ECS_OPERATOR_KWARGS,
    )


with DAG(
    dag_id="ml_dynamic_pipeline_with_ingestion_and_training",
    default_args=DEFAULT_DAG_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    render_template_as_native_obj=True,
    params={
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
        "data_validation": Param(
            {"min_row_count": 10},
            type="object",
            description="Expectation overrides for dataset validation.",
        ),
    },
    tags=["ml", "pipeline", "ecs"],
) as dag:
    ingest_task = _pipeline_task(
        task_id="ingest_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.ingest_data",
            "--data-source",
            "{{ params.data_source }}",
            "--data-format",
            "{{ params.data_format }}",
            "--input-schema-version",
            "{{ params.input_schema_version }}",
            "--ingestion-config",
            "{{ params.ingestion_config | tojson }}",
            "--project-root",
            CONTAINER_PROJECT_DIR,
            "--upload-bucket",
            ARTIFACT_BUCKET,
            "--upload-object-key",
            DATASET_OBJECT_KEY,
        ],
        xcom_push=False,
    )

    validate_task = _pipeline_task(
        task_id="validate_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.validate_data",
            "--dataset-path",
            DATASET_S3_URI,
            "--data-format",
            "{{ params.data_format }}",
            "--target-column",
            "{{ params.target_column }}",
            "--validation-config",
            "{{ params.data_validation | tojson }}",
            "--report-upload-bucket",
            ARTIFACT_BUCKET,
            "--report-upload-object-key",
            VALIDATION_REPORT_KEY,
        ],
        xcom_push=False,
    )

    train_task = _pipeline_task(
        task_id="train_model",
        command=[
            "-m",
            "pipeline_worker.cli.train_model",
            "--train-data-path",
            DATASET_S3_URI,
            "--target-column",
            "{{ params.target_column }}",
            "--model-name",
            "{{ params.model_name }}",
            "--model-version",
            "{{ params.model_version }}",
            "--hyperparameters",
            "{{ params.hyperparameters | tojson }}",
            "--training-scenario",
            "{{ params.training_scenario }}",
            "--target-output-path",
            TRAINING_OUTPUT_BASE,
            "--test-size",
            "{{ params.test_size }}",
            "--random-state",
            "{{ params.random_state }}",
        ],
        xcom_push=False,
    )

    save_results_task = _pipeline_task(
        task_id="save_results",
        command=[
            "-m",
            "pipeline_worker.cli.save_results",
            "--model-artifact-path",
            MODEL_ARTIFACT_URI,
            "--target-output-path",
            "{{ params.target_output_path }}",
        ],
        xcom_push=False,
    )

    ingest_task >> validate_task >> train_task >> save_results_task
