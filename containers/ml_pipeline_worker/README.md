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