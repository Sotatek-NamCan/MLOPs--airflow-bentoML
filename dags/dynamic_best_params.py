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
    "target_column": Param("target", type="string", description="Target column to predict."),
    "model_name": Param("random_forest_classifier", type="string", description="Model identifier."),
    "hyperparameters": Param({"n_estimators": 100, "max_depth": 5}, type="object", description="Model hyperparameters."),
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
    "test_size": Param(0.2, type="number", description="Validation split ratio."),
    "random_state": Param(42, type="integer", description="Random seed used for training."),
    "ingested_dataset_uri": Param("", type="string", description="Source dataset URI for hyperparameter tuning."),
    "tuning_results_uri": Param("", type="string", description="Destination URI for tuned hyperparameters."),
}

INGESTED_DATASET_URI = "{{ params.ingested_dataset_uri }}"
TUNING_RESULTS_URI = "{{ params.tuning_results_uri }}"


with DAG(
    dag_id="ml_best_params_pipeline_3",
    params=PARAMS,
    **DEFAULT_DAG_KWARGS,
) as best_params_dag:
    tune_hyperparameters = pipeline_task(
        task_id="tune_hyperparameters",
        command=[
            "-m",
            "pipeline_worker.cli.tune_model",
            "--train-data-path",
            INGESTED_DATASET_URI,
            "--target-column",
            "{{ params.target_column }}",
            "--model-name",
            "{{ params.model_name }}",
            "--base-hyperparameters",
            "{{ params.hyperparameters | tojson }}",
            "--tuning-config",
            "{{ params.hyperparameter_tuning | tojson }}",
            "--output-uri",
            TUNING_RESULTS_URI,
            "--test-size",
            "{{ params.test_size }}",
            "--random-state",
            "{{ params.random_state }}",
        ],
        do_xcom_push=False,
    )
