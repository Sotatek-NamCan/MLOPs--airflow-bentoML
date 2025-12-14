from __future__ import annotations

import json
import os
import socket
import string
from datetime import datetime
from typing import Iterable, Sequence

from docker.types import Mount

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator


def _looks_like_container_id(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and len(stripped) in (12, 64) and all(ch in string.hexdigits for ch in stripped)


def _log_stream_host() -> str:
    override = os.environ.get("ML_PIPELINE_WORKER_LOG_HOST")
    if override:
        return override
    env_candidate = os.environ.get("HOSTNAME") or socket.getfqdn()
    if env_candidate and not _looks_like_container_id(env_candidate):
        return env_candidate
    return os.environ.get("ML_PIPELINE_WORKER_SERVICE_HOST", "airflow-worker")


LOG_STREAM_HOST = _log_stream_host()


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


class PipelineDockerOperator(DockerOperator):
    """DockerOperator variant that keeps Airflow hostnames stable for log streaming."""

    def pre_execute(self, context):
        self.command = _stringify_command(self.command)
        if self.entrypoint is not None:
            self.entrypoint = _stringify_command(self.entrypoint)
        ti = context.get("ti")
        if ti and not getattr(ti, "hostname", None):
            ti.hostname = LOG_STREAM_HOST
        return super().pre_execute(context)


def _env_subset(keys: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            env[key] = value
    return env


def _normalize_host_path(raw: str | None) -> str | None:
    if not raw:
        return None
    candidate = raw.strip().strip('"').replace("\\", "/")
    if not candidate:
        return None
    if candidate.startswith("/"):
        return candidate
    if len(candidate) >= 2 and candidate[1] == ":":
        drive = candidate[0].lower()
        remainder = candidate[2:].lstrip("/").replace("\\", "/")
        return f"/host_mnt/{drive}/{remainder}"
    return None


def _host_bind_root() -> str | None:
    candidates = [
        os.environ.get("ML_PIPELINE_HOST_PROJECT_DIR"),
        os.environ.get("AIRFLOW_PROJ_DIR"),
    ]
    for candidate in candidates:
        normalized = _normalize_host_path(candidate)
        if normalized:
            return normalized
    return None


HOST_PROJECT_DIR = _host_bind_root()
CONTAINER_PROJECT_DIR = os.environ.get("ML_PIPELINE_CONTAINER_PROJECT_DIR", "/srv/pipeline")
PIPELINE_IMAGE = os.environ.get("ML_TASK_IMAGE", "mlops/pipeline-worker:latest")
DOCKER_URL = os.environ.get("ML_PIPELINE_DOCKER_URL", "unix://var/run/docker.sock")
DOCKER_API_VERSION = os.environ.get("ML_PIPELINE_DOCKER_API_VERSION", "auto")
SHARED_VOLUME_NAME = os.environ.get("ML_PIPELINE_SHARED_VOLUME", "ml_pipeline_workspace")

SHARED_MOUNTS = (
    [
        Mount(
            source=HOST_PROJECT_DIR,
            target=CONTAINER_PROJECT_DIR,
            type="bind",
        )
    ]
    if HOST_PROJECT_DIR
    else [
        Mount(
            source=SHARED_VOLUME_NAME,
            target=CONTAINER_PROJECT_DIR,
            type="volume",
        )
    ]
)

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

BASE_OPERATOR_KWARGS = {
    "image": PIPELINE_IMAGE,
    "docker_url": DOCKER_URL,
    "api_version": DOCKER_API_VERSION,
    "mounts": SHARED_MOUNTS,
    "mount_tmp_dir": False,
    "auto_remove": "success",
}


def _pipeline_task(
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


with DAG(
    dag_id="ml_dynamic_pipeline_with_ingestion_and_training",
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
    tags=["ml", "pipeline", "docker"],
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
        ],
        do_xcom_push=True,
    )

    validate_task = _pipeline_task(
        task_id="validate_dataset",
        command=[
            "-m",
            "pipeline_worker.cli.validate_data",
            "--dataset-path",
            "{{ ti.xcom_pull(task_ids='ingest_dataset') }}",
            "--data-format",
            "{{ params.data_format }}",
            "--target-column",
            "{{ params.target_column }}",
            "--validation-config",
            "{{ params.data_validation | tojson }}",
        ],
        do_xcom_push=False,
    )

    train_task = _pipeline_task(
        task_id="train_model",
        command=[
            "-m",
            "pipeline_worker.cli.train_model",
            "--train-data-path",
            "{{ ti.xcom_pull(task_ids='ingest_dataset') }}",
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
            "{{ params.target_output_path }}",
            "--test-size",
            "{{ params.test_size }}",
            "--random-state",
            "{{ params.random_state }}",
        ],
        do_xcom_push=True,
    )

    save_results_task = _pipeline_task(
        task_id="save_results",
        command=[
            "-m",
            "pipeline_worker.cli.save_results",
            "--model-artifact-path",
            "{{ ti.xcom_pull(task_ids='train_model') }}",
            "--target-output-path",
            "{{ params.target_output_path }}",
        ],
        do_xcom_push=True,
    )

    ingest_task >> validate_task >> train_task >> save_results_task
