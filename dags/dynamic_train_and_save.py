from __future__ import annotations

from airflow import DAG

from pipeline_common import DEFAULT_DAG_KWARGS, pipeline_task, select_params


INGESTED_DATASET_URI = "{{ dag_run.conf.get('ingested_dataset_uri') }}"
TRAINING_BASE_URI = "{{ dag_run.conf.get('training_base_uri') }}"
TRAINED_MODEL_URI = "{{ dag_run.conf.get('trained_model_uri') }}"
TUNING_RESULTS_URI = "{{ dag_run.conf.get('tuning_results_uri') }}"


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
