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
    "data_source": Param("", type="string", description="Location of the dataset to ingest."),
    "data_format": Param("csv", type="string", description="Dataset format (csv, json, parquet...)."),
    "input_schema_version": Param("v1", type="string", description="Schema version (optional hook)."),
    "ingestion_config": Param({}, type="object", description="Optional overrides for ingestion."),
    "ingested_dataset_uri": Param("", type="string", description="Destination URI for the ingested dataset."),
}

INGESTED_DATASET_URI = "{{ params.ingested_dataset_uri }}"


with DAG(
    dag_id="ml_ingest_data_pipeline_1_ecs",
    default_args=DEFAULT_DAG_ARGS,
    params=PARAMS,
    **DEFAULT_DAG_KWARGS,
) as ingest_dag:
    ingest_dataset = pipeline_task(
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
            "--output-uri",
            INGESTED_DATASET_URI,
        ],
        do_xcom_push=False,
    )
