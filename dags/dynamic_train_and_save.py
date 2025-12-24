from __future__ import annotations

from airflow import DAG

from pipeline_common import (
    DEFAULT_DAG_KWARGS,
    INGESTED_DATASET_URI,
    TRAINED_MODEL_URI,
    TRAINING_BASE_URI,
    TUNING_RESULTS_URI,
    pipeline_task,
    select_params,
)


with DAG(
    dag_id="ml_train_and_save_pipeline_4",
    params=select_params(
        "target_column",
        "model_name",
        "model_version",
        "hyperparameters",
        "training_scenario",
        "target_output_path",
        "test_size",
        "random_state",
    ),
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
