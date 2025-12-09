from __future__ import annotations

import json
import os
import socket
import string
from datetime import datetime
from pathlib import Path
from typing import Iterable

from docker.types import Mount

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator


def _looks_like_container_id(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return len(stripped) in (12, 64) and all(ch in string.hexdigits for ch in stripped)


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


def _stringify_command(value):
    if isinstance(value, list):
        normalized = []
        for part in value:
            if isinstance(part, (dict, list)):
                normalized.append(json.dumps(part))
            elif part is None:
                normalized.append("")
            else:
                normalized.append(str(part))
        return normalized
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


class PipelineDockerOperator(DockerOperator):
    """DockerOperator variant that copies the host name for log streaming."""

    def pre_execute(self, context):
        self.command = _stringify_command(self.command)
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


def _host_bind_root() -> Path | None:
    """Return a Docker daemon-friendly project root if available."""
    candidates = [
        os.environ.get("ML_PIPELINE_HOST_PROJECT_DIR"),
        os.environ.get("AIRFLOW_PROJ_DIR"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.strip().replace("\\", "/")
        if normalized.startswith("/"):
            return Path(normalized).expanduser()
    return None


def _upload_python_snippet() -> str:
    return (
        "from pathlib import Path\n"
        "import os\n"
        "from pipeline_worker.storage import build_storage_client\n"
        "client = build_storage_client()\n"
        "source = Path(os.environ['UPLOAD_SOURCE'])\n"
        "object_key = os.environ['UPLOAD_OBJECT_KEY']\n"
        "client.upload(source=source, object_key=object_key)\n"
    )


def _upload_block(local_path: str, object_key: str) -> str:
    snippet = _upload_python_snippet()
    return (
        f'UPLOAD_SOURCE="{local_path}" UPLOAD_OBJECT_KEY="{object_key}" python - <<\'PY\'\n'
        f"{snippet}"
        "PY\n"
    )


HOST_PROJECT_DIR = _host_bind_root()
CONTAINER_PROJECT_DIR = os.environ.get("ML_PIPELINE_CONTAINER_PROJECT_DIR", "/srv/pipeline")
PIPELINE_IMAGE = os.environ.get("ML_TASK_IMAGE", "mlops/pipeline-worker:latest")
DOCKER_URL = os.environ.get("ML_PIPELINE_DOCKER_URL", "unix://var/run/docker.sock")
SHARED_VOLUME_NAME = os.environ.get("ML_PIPELINE_SHARED_VOLUME", "ml_pipeline_workspace")
S3_ARTIFACT_PREFIX = os.environ.get("ML_PIPELINE_ARTIFACT_PREFIX", "pipeline-runs")
RUN_ID_SAFE = "{{ run_id | replace(':', '_') }}"
RUN_LOCAL_DIR = f"{CONTAINER_PROJECT_DIR}/artifacts/{RUN_ID_SAFE}"
RUN_S3_PREFIX = f"{S3_ARTIFACT_PREFIX}/{{{{ ds_nodash }}}}/{RUN_ID_SAFE}"

DATASET_LOCAL_PATH = f"{RUN_LOCAL_DIR}/datasets/ingested.{{{{ params.data_format | default('csv') }}}}"
DATASET_S3_KEY = f"{RUN_S3_PREFIX}/datasets/ingested.{{{{ params.data_format | default('csv') }}}}"
VALIDATION_REPORT_PATH = f"{RUN_LOCAL_DIR}/validation/report.json"
VALIDATION_S3_KEY = f"{RUN_S3_PREFIX}/validation/report.json"
MODEL_LOCAL_PATH = f"{RUN_LOCAL_DIR}/models/model_artifact.bin"
MODEL_S3_KEY = f"{RUN_S3_PREFIX}/models/model_artifact.bin"
RESULT_LOCAL_PATH = f"{RUN_LOCAL_DIR}/results/result.json"
RESULT_S3_KEY = f"{RUN_S3_PREFIX}/results/result.json"

SHARED_MOUNTS = (
    [
        Mount(
            source=HOST_PROJECT_DIR.as_posix(),
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

INGEST_UPLOAD_BLOCK = _upload_block(DATASET_LOCAL_PATH, DATASET_S3_KEY)
VALIDATION_UPLOAD_BLOCK = _upload_block(VALIDATION_REPORT_PATH, VALIDATION_S3_KEY)
MODEL_UPLOAD_BLOCK = _upload_block(MODEL_LOCAL_PATH, MODEL_S3_KEY)
RESULT_UPLOAD_BLOCK = _upload_block(RESULT_LOCAL_PATH, RESULT_S3_KEY)

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
    ingest_task = PipelineDockerOperator(
        task_id="ingest_dataset",
        image=PIPELINE_IMAGE,
        docker_url=DOCKER_URL,
        api_version="auto",
        command=[
            "bash",
            "-c",
            f"""
set -euo pipefail
OUTPUT_PATH=$(python -m pipeline_worker.cli.ingest_data \
    --data-source "{{{{ params.data_source }}}}" \
    --data-format "{{{{ params.data_format }}}}" \
    --input-schema-version "{{{{ params.input_schema_version }}}}" \
    --ingestion-config "{{{{ params.ingestion_config | tojson }}}}" \
    --project-root {CONTAINER_PROJECT_DIR})
TARGET="{DATASET_LOCAL_PATH}"
mkdir -p "$(dirname "$TARGET")"
cp "$OUTPUT_PATH" "$TARGET"
{INGEST_UPLOAD_BLOCK}
""",
        ],
        environment={**BASE_ENV},
        mount_tmp_dir=False,
        auto_remove="success",
        do_xcom_push=False,
        mounts=SHARED_MOUNTS,
    )

    validate_task = PipelineDockerOperator(
        task_id="validate_dataset",
        image=PIPELINE_IMAGE,
        docker_url=DOCKER_URL,
        api_version="auto",
        command=[
            "bash",
            "-c",
            f"""
set -euo pipefail
TARGET="{VALIDATION_REPORT_PATH}"
mkdir -p "$(dirname "$TARGET")"
python -m pipeline_worker.cli.validate_data \
    --dataset-path "{DATASET_LOCAL_PATH}" \
    --data-format "{{{{ params.data_format }}}}" \
    --target-column "{{{{ params.target_column }}}}" \
    --validation-config "{{{{ params.data_validation | tojson }}}}" | tee "$TARGET"
{VALIDATION_UPLOAD_BLOCK}
""",
        ],
        environment={**BASE_ENV},
        mount_tmp_dir=False,
        auto_remove="success",
        do_xcom_push=False,
        mounts=SHARED_MOUNTS,
    )

    train_task = PipelineDockerOperator(
        task_id="train_model",
        image=PIPELINE_IMAGE,
        docker_url=DOCKER_URL,
        api_version="auto",
        command=[
            "bash",
            "-c",
            f"""
set -euo pipefail
OUTPUT_PATH=$(python -m pipeline_worker.cli.train_model \
    --train-data-path "{DATASET_LOCAL_PATH}" \
    --target-column "{{{{ params.target_column }}}}" \
    --model-name "{{{{ params.model_name }}}}" \
    --model-version "{{{{ params.model_version }}}}" \
    --hyperparameters "{{{{ params.hyperparameters | tojson }}}}" \
    --training-scenario "{{{{ params.training_scenario }}}}" \
    --target-output-path "{{{{ params.target_output_path }}}}" \
    --test-size "{{{{ params.test_size }}}}" \
    --random-state "{{{{ params.random_state }}}}")
TARGET="{MODEL_LOCAL_PATH}"
mkdir -p "$(dirname "$TARGET")"
cp "$OUTPUT_PATH" "$TARGET"
{MODEL_UPLOAD_BLOCK}
""",
        ],
        environment={**BASE_ENV},
        mount_tmp_dir=False,
        auto_remove="success",
        do_xcom_push=False,
        mounts=SHARED_MOUNTS,
    )

    save_results_task = PipelineDockerOperator(
        task_id="save_results",
        image=PIPELINE_IMAGE,
        docker_url=DOCKER_URL,
        api_version="auto",
        command=[
            "bash",
            "-c",
            f"""
set -euo pipefail
OUTPUT_PATH=$(python -m pipeline_worker.cli.save_results \
    --model-artifact-path "{MODEL_LOCAL_PATH}" \
    --target-output-path "{{{{ params.target_output_path }}}}")
TARGET="{RESULT_LOCAL_PATH}"
mkdir -p "$(dirname "$TARGET")"
cp "$OUTPUT_PATH" "$TARGET"
{RESULT_UPLOAD_BLOCK}
""",
        ],
        environment={**BASE_ENV},
        mount_tmp_dir=False,
        auto_remove="success",
        do_xcom_push=False,
        mounts=SHARED_MOUNTS,
    )

    ingest_task >> validate_task >> train_task >> save_results_task
