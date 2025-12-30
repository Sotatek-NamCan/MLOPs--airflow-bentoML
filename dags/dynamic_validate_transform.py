from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable, Sequence

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator


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


class PipelineDockerOperator(DockerOperator):
    """DockerOperator variant that normalizes command/entrypoint payloads."""

    template_fields = tuple(
        field for field in DockerOperator.template_fields if field != "environment"
    )

    def pre_execute(self, context):
        self.command = _stringify_command(self.command)
        if self.entrypoint is not None:
            self.entrypoint = _stringify_command(self.entrypoint)
        return super().pre_execute(context)


def _env_subset(keys: Iterable[str]) -> dict[str, str]:
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


DEFAULT_DAG_KWARGS = {
    "start_date": datetime(2024, 1, 1),
    "schedule": None,
    "catchup": False,
    "render_template_as_native_obj": True,
    "tags": ["ml", "pipeline", "docker"],
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
    dag_id="ml_validate_transform_pipeline_2",
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
