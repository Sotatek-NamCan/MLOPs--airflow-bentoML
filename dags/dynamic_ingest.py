from __future__ import annotations

from airflow import DAG

from pipeline_common import (
    CONTAINER_PROJECT_DIR,
    DEFAULT_DAG_KWARGS,
    INGESTED_DATASET_URI,
    pipeline_task,
    select_params,
)


with DAG(
    dag_id="ml_ingest_data_pipeline_1",
    params=select_params("data_source", "data_format", "input_schema_version", "ingestion_config"),
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
