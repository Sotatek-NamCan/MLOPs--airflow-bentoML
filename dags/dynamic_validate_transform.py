from __future__ import annotations

from airflow import DAG

from pipeline_common import DEFAULT_DAG_KWARGS, pipeline_task, select_params


INGESTED_DATASET_URI = "{{ dag_run.conf.get('ingested_dataset_uri') }}"
CLEANED_DATASET_URI = "{{ dag_run.conf.get('cleaned_dataset_uri') }}"
CLEANING_SUMMARY_URI = "{{ dag_run.conf.get('cleaning_summary_uri') }}"
DATA_PROFILE_URI = "{{ dag_run.conf.get('data_profile_uri') }}"
DATA_VISUALIZATION_URI = "{{ dag_run.conf.get('data_visualization_uri') }}"
VALIDATION_REPORT_URI = "{{ dag_run.conf.get('validation_report_uri') }}"


with DAG(
    dag_id="ml_validate_transform_pipeline_2",
    params=select_params(
        "data_format",
        "target_column",
        "data_cleaning",
        "data_validation",
        "data_profiling",
        "data_visualization",
    ),
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
