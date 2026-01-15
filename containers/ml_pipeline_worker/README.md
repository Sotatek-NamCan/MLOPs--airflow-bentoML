# Pipeline Worker Image

This directory contains the build context for the container image that hosts the
training/ingestion logic previously embedded inside the Airflow DAGs.

## Build

```bash
docker build -t mlops/pipeline-worker:latest containers/ml_pipeline_worker
```

Use the `ML_TASK_IMAGE` environment variable in Airflow to point the
`DockerOperator` tasks towards the desired tag (for example an image published
in your registry).

## Contents

The actual application code lives under `src/pipeline_worker`. It exposes a
couple of light-weight CLI entrypoints that the DAG uses via
`python -m pipeline_worker.cli.<command>`.

### train

```
python -m pipeline_worker.cli.train_model `
  --train-data-path "s3://cpnam-s3-tfbackend/data/bank.csv" `
  --target-column "age" `
  --model-name "logistic_regression" `
  --model-version "2" `
  --target-output-path "s3://cpnam-s3-tfbackend/output/bank_marketing/" `
  --hyperparameters '{\"C\": 0.5, \"solver\": \"liblinear\", \"max_iter\": 200}' `
  --training-scenario "incremental_train" `
  --test-size 0.15 `
  --random-state 123

```

### ingest

```
python -m pipeline_worker.cli.tune_model `
    --train-data-path "s3://cpnam-s3-tfbackend/data/bank.csv" `
    --target-column "age" `
    --model-name "logistic_regression" `
    --base-hyperparameters '{\"solver\": \"liblinear\"}' `
    --tuning-config '{\"enabled\": true, \"n_trials\": 40, \"timeout\": 900, \"search_space\": {\"C\": {\"type\": \"float\", \"low\": 0.001, \"high\": 10, \"log\": true}, \"max_iter\": {\"type\": \"int\", \"low\": 100, \"high\": 400, \"step\": 50}}}' `
    --output-uri "s3://cpnam-s3-tfbackend/output/bank_marketing/best_params.json" `
    --test-size 0.2 `
    --random-state 123

```

### tune
```
python -m pipeline_worker.cli.tune_model `
    --train-data-path "s3://cpnam-s3-tfbackend/data/bank.csv" `
    --target-column "age" `
    --model-name "logistic_regression" `
    --base-hyperparameters '{\"solver\": \"liblinear\"}' `
    --tuning-config '{\"enabled\": true, \"n_trials\": 40, \"timeout\": 900, \"search_space\": {\"C\": {\"type\": \"float\", \"low\": 0.001, \"high\": 10, \"log\": true}, \"max_iter\": {\"type\": \"int\", \"low\": 100, \"high\": 400, \"step\": 50}}}' `
    --output-uri "s3://cpnam-s3-tfbackend/output/bank_marketing/best_params.json" `
    --test-size 0.2 `
    --random-state 123

```


### validate

```
python -m pipeline_worker.cli.validate_data `
    --dataset-path "s3://cpnam-s3-tfbackend/data/bank.csv" `
    --data-format "csv" `
    --target-column "age" `
    --validation-config '{\"max_missing\": 0.02, \"drop_null_rows\": true}' `
    --report-uri "s3://cpnam-s3-tfbackend/output/bank_marketing/validation_summary.json"

```

## Execution flow

Train flow (`python -m pipeline_worker.cli.train_model`).
- Parse CLI args in `pipeline_worker/cli/train_model.py`.
- Download data (and optional tuned params) with `ensure_local_artifact`.
- Call `train_and_save_model` in `pipeline_worker/train_utils.py`.
- `train_and_save_model` loads data via `load_train_data`, splits train/test, selects model via `select_model`, fits, computes metrics, logs to MLflow, saves the model artifact locally, and uploads to S3 if the output path is an S3 URI.

Tune flow (`python -m pipeline_worker.cli.tune_model`).
- Parse CLI args and tuning JSON in `pipeline_worker/cli/tune_model.py`.
- If tuning is disabled or search space is empty, return base params.
- Otherwise load data and call `perform_hyperparameter_search` in `pipeline_worker/hpo.py`.
- `perform_hyperparameter_search` builds the search space, splits train/valid, runs Optuna trials in `_trial_objective` (which calls `select_model`), and returns the best params.
- The best params are saved to `best_params.json` and uploaded via `upload_local_artifact`.

Notes:
- `load_train_data` drops non-numeric feature columns and fails if no numeric features remain.

## Adding metrics and logging to MLflow

Metrics are computed and logged inside `train_and_save_model()` in
`containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py`.

Exact locations (line numbers reflect the current file and may shift after edits):
- `_record_metric()` helper: `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:214`
- Predictions used for metrics: `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:314`
- Classification metrics block: `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:318`
- Regression metrics block: `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:342`
- MLflow metric logging loop: `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:363`

Checklist:
- Add your metric calculation in the classification/regression block after
  `y_pred = model.predict(X_test)`.
- Use `_record_metric(metrics, name, compute_fn)` so invalid values are safely
  skipped (non-numeric, NaN/inf, exceptions).
- Metrics are logged to MLflow via the existing loop:
  `mlflow.log_metric(name, value)`.

Example (classification metric with probabilities):
```
from sklearn.metrics import roc_auc_score

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)
    _record_metric(
        metrics,
        "roc_auc",
        lambda: roc_auc_score(y_test, y_proba, multi_class="ovr"),
    )
```

Example (regression metric):
```
from sklearn.metrics import median_absolute_error

_record_metric(metrics, "median_ae", lambda: median_absolute_error(y_test, y_pred))
```

Notes:
- Prefer `_record_metric` instead of direct `mlflow.log_metric` so invalid values
  do not fail the training run.
- If your metric needs extra model outputs (probabilities, decision scores),
  guard with `hasattr(model, ...)` to avoid exceptions.

## Adding a model 

### Train model flow (only training, no HPO)

Files and functions you must touch (line numbers reflect current file; may shift after edits).
- `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py`
  - `_MODEL_REGISTRY` at line 50: add a new key, example `"xgboost_classifier": XGBClassifier`.
  - `select_model()` at line 240: uses `_MODEL_REGISTRY`, so `--model-name` must match the key.
  - `train_and_save_model()` at line 253: add or adjust metrics via `_record_metric(...)`.
  - `_record_metric()` at line 216: shared validation/logging for metrics.
  - `_validate_hyperparameters()` at line 167: requires `get_params()` to exist.
- `containers/ml_pipeline_worker/src/pipeline_worker/cli/train_model.py`
  - `main()` at line 22: reads CLI args and calls `train_and_save_model()`.
  - `train_and_save_model()` import at line 9.

Step-by-step checklist (train).
- Add the import.
- Add the registry entry in `_MODEL_REGISTRY`.
- Update metrics in `train_and_save_model()` if needed.
- Run the `train_model` CLI with the new `--model-name`.

### HPO flow (only tuning)

Files and functions you must touch (line numbers reflect current file; may shift after edits).
- `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py`
  - `MODEL_DIRECTIONS` at line 25: add model direction.
  - `DEFAULT_SEARCH_SPACES` at line 32: add default search space (optional).
  - `_infer_task_type()` at line 121: return `"classification"` or `"regression"`.
  - `_trial_objective()` at line 91: update scoring if needed.
  - `perform_hyperparameter_search()` at line 132: entry point used by the CLI.
- `containers/ml_pipeline_worker/src/pipeline_worker/cli/tune_model.py`
  - `main()` at line 17: reads CLI args and calls `perform_hyperparameter_search()`.
  - `perform_hyperparameter_search()` import at line 12.

Step-by-step checklist (HPO).
- Add direction in `MODEL_DIRECTIONS`.
- Update `_infer_task_type()`.
- Add default search space (optional).
- Update `_trial_objective()` if your model uses a different scoring metric.
- Run the `tune_model` CLI with the new `--model-name`.

### Non-scikit-learn model notes (applies to both train + HPO)

- Create a wrapper class with a scikit-learn-like API:
  - `fit(X, y)` and `predict(X)` are required.
  - `get_params()` is required by `_validate_hyperparameters()`.
  - `set_params(**kwargs)` is required for Optuna to tune parameters.
- Register the wrapper in `_MODEL_REGISTRY`.
- Update `MODEL_DIRECTIONS`, `_infer_task_type()`, and possibly `_trial_objective()` to align scoring.

### Cheat-sheet: model to file/line mapping

Line numbers reflect the current files and will shift after edits.

| Model name | Train registry | HPO direction | HPO search space | HPO task type |
| --- | --- | --- | --- | --- |
| `random_forest_classifier` | `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:51` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:26` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:33` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:123` |
| `logistic_regression` | `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:52` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:27` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:45` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:123` |
| `linear_regression` | `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:53` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:29` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:49` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:125` |
| `random_forest_regressor` | `containers/ml_pipeline_worker/src/pipeline_worker/train_utils.py:54` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:28` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:39` | `containers/ml_pipeline_worker/src/pipeline_worker/hpo.py:125` |
