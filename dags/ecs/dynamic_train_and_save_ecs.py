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
    "target_column": Param("target", type="string", description="Target column to predict."),
    "model_name": Param("random_forest_classifier", type="string", description="Model identifier."),
    "model_version": Param("1", type="string", description="Model version string."),
    "hyperparameters": Param({"n_estimators": 100, "max_depth": 5}, type="object", description="Model hyperparameters."),
    "training_scenario": Param("full_train", type="string", description="Training scenario label."),
    "target_output_path": Param("s3://bucket/output/", type="string", description="Folder/URI for model artifacts."),
    "test_size": Param(0.2, type="number", description="Validation split ratio."),
    "random_state": Param(42, type="integer", description="Random seed used for training."),
    "ingested_dataset_uri": Param("", type="string", description="Source dataset URI for training."),
    "training_base_uri": Param("", type="string", description="Base URI for training artifacts."),
    "trained_model_uri": Param("", type="string", description="URI of the trained model artifact."),
    "tuning_results_uri": Param("", type="string", description="URI of tuned hyperparameters (optional)."),
}

INGESTED_DATASET_URI = "{{ params.ingested_dataset_uri }}"
TRAINING_BASE_URI = "{{ params.training_base_uri }}"
TRAINED_MODEL_URI = "{{ params.trained_model_uri }}"
TUNING_RESULTS_URI = "{{ params.tuning_results_uri }}"


with DAG(
    dag_id="ml_train_and_save_pipeline_4_ecs",
    default_args=DEFAULT_DAG_ARGS,
    params=PARAMS,
    **DEFAULT_DAG_KWARGS,
) as train_and_save_dag:
    train_model = pipeline_task(
        task_id="train_model",
        command=[
            "-m",
            "pipeline_worker.cli.train_model",
            "--train-data-path",
            INGESTED_DATASET_URI,
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
            TRAINING_BASE_URI,
            "--test-size",
            "{{ params.test_size }}",
            "--random-state",
            "{{ params.random_state }}",
            "--best-params-uri",
            TUNING_RESULTS_URI,
        ],
        do_xcom_push=False,
    )

    save_results = pipeline_task(
        task_id="save_results",
        command=[
            "-m",
            "pipeline_worker.cli.save_results",
            "--model-artifact-path",
            TRAINED_MODEL_URI,
            "--target-output-path",
            "{{ params.target_output_path }}",
        ],
        do_xcom_push=False,
    )

    train_model >> save_results
