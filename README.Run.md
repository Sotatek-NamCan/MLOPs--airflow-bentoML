# README.Run.md

This document describes the main tasks inside the pipeline worker container.

## 1) preprocessing.py
Purpose: clean input data before training/validation.

Libraries:
- pandas, numpy

Key components:
- `CleaningConfig`: cleaning configuration (drop columns, de-duplicate, missing values, outliers, transforms).
- `clean_dataframe(df, config)`: main entry point, returns `(cleaned_dataframe, summary)`.

Details:
- Drop columns: removes columns listed in `drop_columns` if they exist.
- Missing values:
  - Numeric columns: `mean` / `median` (default) / `zero`.
  - Categorical columns: `mode` (default) or a fixed fill value (e.g. "missing").
  - `column_fill_values` overrides strategy per column.
- Outliers:
  - Uses IQR (Q1, Q3) and clips with `iqr_factor` (default 1.5).
- Deduplicate: drops duplicate rows when `deduplicate=True`.
- Reorder columns: follows `column_order` when provided.
- Transformations:
  - Standardize (z-score), MinMax scale, Log1p (with optional `shift`), Power transform.
- Summary includes: row counts before/after, dropped columns, missing-value strategy,
  outlier adjustments, transformations, and column list.

Param example:
```json
{
  "drop_columns": ["id", "raw_text"],
  "deduplicate": true,
  "column_order": ["age", "income", "target"],
  "missing_values": {
    "numeric_strategy": "median",
    "categorical_strategy": "mode",
    "categorical_fill_value": "missing",
    "column_fill_values": {
      "zipcode": "00000"
    }
  },
  "outliers": {
    "method": "iqr",
    "columns": ["age", "income"],
    "iqr_factor": 1.5
  },
  "transformations": [
    {
      "type": "standardize",
      "columns": ["age", "income"]
    },
    {
      "type": "log1p",
      "columns": ["spend"],
      "shift": 0
    }
  ]
}
```

## 2) profiling.py
Purpose: descriptive profiling and distribution plots.

Libraries:
- pandas
- matplotlib (imported only when charts are requested)

Key components:
- `build_profile_summary(df, config)`:
  - Builds per-column statistics.
  - Numeric columns: min/max/mean/median/std/skew + percentiles (default 25/50/75).
  - Categorical columns: top N values (default 5).
  - Global stats: row/column counts, null counts, null ratio.
- `render_visualizations(df, output_dir, config)`:
  - Charts: histogram + boxplot for numeric columns, bar chart for categorical columns.
  - Limits via `max_numeric_charts`, `max_categorical_charts`.
  - Saves images (png/jpg) and returns metadata (file name, chart type, column).

Param example:
```json
{
  "top_value_count": 5,
  "include_percentiles": [0.25, 0.5, 0.75],
  "max_numeric_charts": 8,
  "max_categorical_charts": 6,
  "plot_format": "png"
}
```

## 3) hpo.py
Purpose: hyperparameter search with Optuna.

Libraries:
- optuna
- scikit-learn (train_test_split, metrics)

Key components:
- `SearchSpec`: search space definition (int/float/categorical).
- `DEFAULT_SEARCH_SPACES`: default search space for
  - random_forest_classifier, random_forest_regressor
  - logistic_regression, linear_regression
- `perform_hyperparameter_search(...)`:
  - Infers task type (classification/regression) and optimization direction.
  - Merges default search space with overrides from input.
  - Splits train/valid and runs Optuna for `n_trials` or `timeout`.
  - Returns `best_params` merged with `base_hyperparameters`.
- `parse_json_payload(value)`:
  - Parses a JSON string into override config.

Param example:
```json
{
  "model_name": "random_forest_classifier",
  "base_hyperparameters": {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2
  },
  "search_space_overrides": {
    "n_estimators": {
      "type": "int",
      "low": 100,
      "high": 400,
      "step": 50
    },
    "max_depth": {
      "type": "int",
      "low": 4,
      "high": 16
    }
  },
  "features": "X_features_df",
  "target": "y_target_series",
  "n_trials": 20,
  "timeout": 600,
  "test_size": 0.2,
  "random_state": 42
}
```

## 4) validation.py
Purpose: validate datasets before training/serving using Great Expectations.

Libraries:
- great_expectations (PandasDataset)
- pandas

Key components:
- `ValidationConfig`: validation configuration.
- `validate_dataframe(df, config)`:
  - `min_row_count` / `max_row_count`: validate number of rows.
  - `required_columns`: required columns must exist.
  - `non_null_columns`: columns must not be null.
  - `unique_columns`: values in columns must be unique.
  - `value_ranges`: min/max constraints (supports `strict_min`, `strict_max`).
  - `allowed_values`: allowed value set per column.
  - Raises `DataValidationError` on any expectation failure,
    including failures list and summary.

Param example:
```json
{
  "min_row_count": 100,
  "max_row_count": 500000,
  "required_columns": ["age", "income", "target"],
  "non_null_columns": ["age", "income", "target"],
  "unique_columns": ["customer_id"],
  "value_ranges": {
    "age": {
      "min": 18,
      "max": 90
    },
    "income": {
      "min": 0,
      "strict_min": true
    }
  },
  "allowed_values": {
    "segment": ["A", "B", "C"]
  }
}
```

## 5) train_utils.py
Purpose: load data, train model, log to MLflow, and save artifacts.

Libraries:
- pandas, scikit-learn
- mlflow
- object storage client (from `pipeline_worker.storage`)

Key components:
- `load_train_data(path, target_column)`:
  - Supports CSV/Parquet/JSON.
  - Validates the target column exists.
  - Drops non-numeric/bool features; raises if no numeric features remain.
- `select_model(model_name, hyperparameters)`:
  - Supports: random_forest_classifier, random_forest_regressor,
    logistic_regression, linear_regression.
- `train_and_save_model(...)`:
  - Sets MLflow tracking URI + experiment via env vars.
  - Splits train/test, trains model, evaluates metrics.
  - Classification: accuracy/precision/recall/f1.
  - Regression: mse/rmse/mae/r2/explained_variance.
  - Logs params/metrics to MLflow.
  - Saves `.pkl` locally or to S3 when `output_dir` starts with `s3://`.

Param example:
```json
{
  "train_data_path": "data/train.csv",
  "target_column": "target",
  "model_name": "random_forest_classifier",
  "model_version": "1.0.0",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 2,
    "min_samples_leaf": 1
  },
  "training_scenario": "baseline",
  "output_dir": "s3://ml-models/airflow",
  "test_size": 0.2,
  "random_state": 42
}
```

Environment variables:
- MLflow: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`.
