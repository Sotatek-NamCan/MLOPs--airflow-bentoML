# Param example for Airflow DAG run configs

This template is based on the files in `demos/*/param.json`. Each top-level key
is a full `dag_run.conf` payload for the DAG with the same id.

## Example JSON (fill with your values)

```json
{
  "_note": "Use the object under each DAG id as the Airflow run config.",
  "ml_ingest_data_pipeline_1": {
    "data_source": "s3://your-bucket/datasets/example.csv",
    "data_format": "csv",
    "input_schema_version": "v1",
    "ingestion_config": {
      "file_extension": ".csv",
      "cache_dir": "data/cache/example"
    },
    "ingested_dataset_uri": "s3://your-bucket/artifacts/example/ingested/example.csv"
  },
  "ml_validate_transform_pipeline_2": {
    "data_format": "csv",
    "target_column": "target",
    "data_cleaning": {
      "drop_columns": [],
      "deduplicate": true,
      "column_order": [
        "feature_1",
        "feature_2",
        "target"
      ],
      "missing_values": {
        "numeric_strategy": "median",
        "categorical_strategy": "mode",
        "categorical_fill_value": "missing"
      },
      "outliers": {
        "method": "iqr",
        "iqr_factor": 1.5,
        "columns": [
          "feature_1",
          "feature_2"
        ]
      },
      "transformations": []
    },
    "data_validation": {
      "min_row_count": 100,
      "required_columns": [
        "feature_1",
        "feature_2",
        "target"
      ],
      "non_null_columns": [
        "feature_1",
        "feature_2",
        "target"
      ],
      "value_ranges": {
        "feature_1": { "min": 0 },
        "feature_2": { "min": 0 }
      },
      "allowed_values": {}
    },
    "data_profiling": {
      "top_value_count": 5,
      "include_percentiles": [0.25, 0.5, 0.75]
    },
    "data_visualization": {
      "max_numeric_charts": 10,
      "max_categorical_charts": 6,
      "plot_format": "png"
    },
    "ingested_dataset_uri": "s3://your-bucket/artifacts/example/ingested/example.csv",
    "cleaned_dataset_uri": "s3://your-bucket/artifacts/example/cleaned/example.csv",
    "cleaning_summary_uri": "s3://your-bucket/artifacts/example/reports/cleaning_summary.json",
    "data_profile_uri": "s3://your-bucket/artifacts/example/reports/profile_summary.json",
    "data_visualization_uri": "s3://your-bucket/artifacts/example/reports/visualizations.zip",
    "validation_report_uri": "s3://your-bucket/artifacts/example/reports/validation_summary.json"
  },
  "ml_best_params_pipeline_3": {
    "target_column": "target",
    "model_name": "random_forest_classifier",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 8,
      "min_samples_split": 2,
      "min_samples_leaf": 1
    },
    "hyperparameter_tuning": {
      "enabled": true,
      "n_trials": 20,
      "timeout": 600,
      "search_space": {
        "n_estimators": { "type": "int", "low": 100, "high": 300, "step": 50 },
        "max_depth": { "type": "int", "low": 4, "high": 12 },
        "min_samples_split": { "type": "int", "low": 2, "high": 6 },
        "min_samples_leaf": { "type": "int", "low": 1, "high": 3 }
      }
    },
    "test_size": 0.2,
    "random_state": 42,
    "ingested_dataset_uri": "s3://your-bucket/artifacts/example/ingested/example.csv",
    "tuning_results_uri": "s3://your-bucket/artifacts/example/hpo/best_params.json"
  },
  "ml_train_and_save_pipeline_4": {
    "target_column": "target",
    "model_name": "random_forest_classifier",
    "model_version": "1",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 8,
      "min_samples_split": 2,
      "min_samples_leaf": 1
    },
    "training_scenario": "baseline",
    "target_output_path": "s3://your-bucket/models/example/",
    "test_size": 0.2,
    "random_state": 42,
    "ingested_dataset_uri": "s3://your-bucket/artifacts/example/ingested/example.csv",
    "training_base_uri": "s3://your-bucket/artifacts/example",
    "trained_model_uri": "s3://your-bucket/artifacts/example/models/random_forest_classifier_v1/random_forest_classifier_v1.pkl",
    "tuning_results_uri": "s3://your-bucket/artifacts/example/hpo/best_params.json"
  }
}
```

## Parameter reference

### ml_ingest_data_pipeline_1
- `data_source` (required): dataset path or URI (local path, S3 URI).
- `data_format` (required): `csv`, `parquet`, or `json`.
- `input_schema_version` (optional): schema version label for downstream logic.
- `ingestion_config` (optional object):
  - `file_extension` (optional): expected file extension (e.g., `.csv`).
  - `cache_dir` (optional): local cache directory inside the container.
- `ingested_dataset_uri` (required): output URI for the ingested dataset.

### ml_validate_transform_pipeline_2
- `data_format` (required): same as ingest step (`csv`, `parquet`, `json`).
- `target_column` (required): target column for validation and training.
- `data_cleaning` (optional object):
  - `drop_columns`: list of columns to remove.
  - `deduplicate`: boolean.
  - `column_order`: explicit output column order.
  - `missing_values`: `numeric_strategy` and `categorical_strategy` (e.g., `median`, `mean`, `mode`), plus optional `categorical_fill_value`.
  - `outliers`: `method` (e.g., `iqr`), `iqr_factor`, and `columns`.
  - `transformations`: list of transformation configs (if supported by the worker).
- `data_validation` (optional object):
  - `min_row_count`: minimum row count.
  - `required_columns`: columns that must exist.
  - `non_null_columns`: columns that must not contain nulls.
  - `value_ranges`: numeric bounds by column.
  - `allowed_values`: categorical allow-list by column.
- `data_profiling` (optional object): `top_value_count`, `include_percentiles`.
- `data_visualization` (optional object): `max_numeric_charts`, `max_categorical_charts`, `plot_format`.
- `ingested_dataset_uri` (required): input URI from the ingest step.
- `cleaned_dataset_uri` (required): output URI for the cleaned dataset.
- `cleaning_summary_uri` (required): output URI for cleaning summary JSON.
- `data_profile_uri` (required): output URI for profiling report JSON.
- `data_visualization_uri` (required): output URI for visualization zip.
- `validation_report_uri` (required): output URI for validation report JSON.

### ml_best_params_pipeline_3
- `target_column` (required): target column name.
- `model_name` (required): must match a key in `train_utils.py:_MODEL_REGISTRY`.
- `hyperparameters` (optional object): base model params passed to the estimator.
- `hyperparameter_tuning` (optional object):
  - `enabled`: toggle Optuna tuning.
  - `n_trials`: number of trials.
  - `timeout`: max seconds.
  - `search_space`: param search definitions:
    - `type`: `int`, `float`, or `categorical`.
    - `low`/`high`/`step` for numeric, `choices` for categorical, `log` for log scale.
- `test_size` (optional): validation split ratio.
- `random_state` (optional): random seed.
- `ingested_dataset_uri` (required): input URI from ingest.
- `tuning_results_uri` (required): output URI for best params JSON.

### ml_train_and_save_pipeline_4
- `target_column` (required): target column name.
- `model_name` (required): must match a key in `train_utils.py:_MODEL_REGISTRY`.
- `model_version` (required): version string for artifact naming.
- `hyperparameters` (optional object): model hyperparameters.
- `training_scenario` (optional): label for MLflow logging.
- `target_output_path` (required): final output base URI for saved artifacts.
- `test_size` (optional): validation split ratio.
- `random_state` (optional): random seed.
- `ingested_dataset_uri` (required): input URI from ingest.
- `training_base_uri` (required): working output base used by the trainer.
- `trained_model_uri` (required): artifact URI consumed by `save_results`.
- `tuning_results_uri` (optional): URI to best params JSON (used if HPO ran).
