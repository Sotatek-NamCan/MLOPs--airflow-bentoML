from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator


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


AWS_CONN_ID = "aws_default"
REGION_NAME = "eu-central-1"
CLUSTER_NAME = "apache-airflow-worker-cluster"
TASK_DEFINITION = "nam-task-definition"
LAUNCH_TYPE = "FARGATE"
CONTAINER_NAME = "nam-container"
SUBNETS = [
    "subnet-039e8ef0b81be962a",
    "subnet-0a6794746e9d41db2",
    "subnet-07c63479705688d25",
]
SECURITY_GROUPS = [
    "sg-03173252d375f316e",
]

NETWORK_CONFIGURATION = {
    "awsvpcConfiguration": {
        "subnets": SUBNETS,
        "securityGroups": SECURITY_GROUPS,
        "assignPublicIp": "ENABLED",
    }
}

CONTAINER_PROJECT_DIR = os.environ.get("ML_PIPELINE_CONTAINER_PROJECT_DIR", "/opt/pipeline")

BASE_ENV = _env_subset(
    [
        "OBJECT_STORAGE_BUCKET",
        "OBJECT_STORAGE_ENDPOINT_URL",
        "OBJECT_STORAGE_REGION",
        "OBJECT_STORAGE_ACCESS_KEY",
        "OBJECT_STORAGE_SECRET_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MODEL_ARTIFACT_CACHE_DIR",
    ]
)
BASE_ENV["PIPELINE_PROJECT_ROOT"] = CONTAINER_PROJECT_DIR
BASE_ENV["PIPELINE_ENV_FILE"] = f"{CONTAINER_PROJECT_DIR}/.env"


def _ecs_environment(extra_env: dict[str, str] | None = None) -> list[dict[str, str]]:
    env = dict(BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return [
        {"name": key, "value": value} for key, value in env.items() if value is not None
    ]


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


def pipeline_task(
    task_id: str,
    command: Sequence | str,
    *,
    do_xcom_push: bool,
    extra_env: dict[str, str] | None = None,
) -> EcsRunTaskOperator:
    overrides = _build_overrides(command, extra_env=extra_env)
    return EcsRunTaskOperator(
        task_id=task_id,
        cluster=CLUSTER_NAME,
        task_definition=TASK_DEFINITION,
        launch_type=LAUNCH_TYPE,
        region_name=REGION_NAME,
        aws_conn_id=AWS_CONN_ID,
        network_configuration=NETWORK_CONFIGURATION,
        wait_for_completion=True,
        awslogs_region=REGION_NAME,
        propagate_tags="TASK_DEFINITION",
        overrides=overrides,
        do_xcom_push=do_xcom_push,
    )


DEFAULT_DAG_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DEFAULT_DAG_KWARGS = {
    "start_date": datetime(2024, 1, 1),
    "schedule": None,
    "catchup": False,
    "render_template_as_native_obj": True,
    "tags": ["ml", "pipeline", "ecs"],
}

PARAMS = {
    "data_format": Param("csv", type="string", description="Dataset format (csv, json, parquet...)."),
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
    "ingested_dataset_uri": Param("", type="string", description="Source dataset URI for validation steps."),
    "cleaned_dataset_uri": Param("", type="string", description="Destination URI for cleaned dataset."),
    "cleaning_summary_uri": Param("", type="string", description="Destination URI for cleaning summary report."),
    "data_profile_uri": Param("", type="string", description="Destination URI for profiling report."),
    "data_visualization_uri": Param("", type="string", description="Destination URI for visualization archive."),
    "validation_report_uri": Param("", type="string", description="Destination URI for validation report."),
}

INGESTED_DATASET_URI = "{{ params.ingested_dataset_uri }}"
CLEANED_DATASET_URI = "{{ params.cleaned_dataset_uri }}"
CLEANING_SUMMARY_URI = "{{ params.cleaning_summary_uri }}"
DATA_PROFILE_URI = "{{ params.data_profile_uri }}"
DATA_VISUALIZATION_URI = "{{ params.data_visualization_uri }}"
VALIDATION_REPORT_URI = "{{ params.validation_report_uri }}"


with DAG(
    dag_id="ml_validate_transform_pipeline_2_ecs",
    default_args=DEFAULT_DAG_ARGS,
    params=PARAMS,
    **DEFAULT_DAG_KWARGS,
) as validate_transform_dag:
    clean_dataset = pipeline_task(
        task_id="clean_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.clean_data",
            "--dataset-path",
            INGESTED_DATASET_URI,
            "--data-format",
            "{{ params.data_format }}",
            "--cleaning-config",
            "{{ params.data_cleaning | tojson }}",
            "--output-uri",
            CLEANED_DATASET_URI,
            "--report-uri",
            CLEANING_SUMMARY_URI,
        ],
        do_xcom_push=False,
    )

    profile_dataset = pipeline_task(
        task_id="profile_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.profile_data",
            "--dataset-path",
            CLEANED_DATASET_URI,
            "--data-format",
            "{{ params.data_format }}",
            "--profile-config",
            "{{ params.data_profiling | tojson }}",
            "--profile-report-uri",
            DATA_PROFILE_URI,
        ],
        do_xcom_push=False,
    )

    visualize_dataset = pipeline_task(
        task_id="visualize_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.visualize_data",
            "--dataset-path",
            CLEANED_DATASET_URI,
            "--data-format",
            "{{ params.data_format }}",
            "--visualization-config",
            "{{ params.data_visualization | tojson }}",
            "--visualization-uri",
            DATA_VISUALIZATION_URI,
        ],
        do_xcom_push=False,
    )

    validate_dataset = pipeline_task(
        task_id="validate_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.validate_data",
            "--dataset-path",
            CLEANED_DATASET_URI,
            "--data-format",
            "{{ params.data_format }}",
            "--target-column",
            "{{ params.target_column }}",
            "--validation-config",
            "{{ params.data_validation | tojson }}",
            "--report-uri",
            VALIDATION_REPORT_URI,
        ],
        do_xcom_push=False,
    )

    clean_dataset >> profile_dataset >> visualize_dataset >> validate_dataset
