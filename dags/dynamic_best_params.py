from __future__ import annotations

from airflow import DAG

from pipeline_common import DEFAULT_DAG_KWARGS, pipeline_task, select_params


INGESTED_DATASET_URI = "{{ dag_run.conf.get('ingested_dataset_uri') }}"
TUNING_RESULTS_URI = "{{ dag_run.conf.get('tuning_results_uri') }}"


with DAG(
    dag_id="ml_best_params_pipeline_3",
    params=select_params(
        "target_column",
        "model_name",
        "hyperparameters",
        "hyperparameter_tuning",
        "test_size",
        "random_state",
    ),
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
