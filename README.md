# Airflow MLOps Project

## Project Tree
```
Airflow/
├─ .env.template                 # Sample environment file for Airflow services
├─ docker-compose.yaml           # Airflow stack with Postgres, Redis, and Celery components
├─ dags/
│  └─ data/
│     └─ dynamic.py              # Dynamic ML pipeline DAG (DockerOperator-based)
├─ containers/
│  └─ ml_pipeline_worker/
│     ├─ Dockerfile              # Image used for ingest/train/save tasks
│     ├─ pyproject.toml          # Dependencies for the worker image
│     └─ src/
│        └─ pipeline_worker/
│           ├─ cli/              # Entrypoints used by DockerOperator commands
│           ├─ ingestion.py      # Dataset download/parsing utilities
│           ├─ storage.py        # Object storage helper classes
│           └─ train_utils.py    # Model training, MLflow logging, artifact upload
├─ config/                       # Airflow configuration overrides (mounted into containers)
├─ data/                         # Local datasets/cache (shared with worker containers)
├─ logs/                         # Airflow logs
└─ plugins/                      # Optional Airflow plugins
```

## Overview
This repository wraps an Apache Airflow deployment (CeleryExecutor + Redis + Postgres) that orchestrates a containerised ML pipeline. The DAG at `dags/data/dynamic.py` runs three DockerOperator tasks:

1. **ingest_dataset** – downloads the requested dataset from object storage, caches it under `/srv/pipeline/data/object_storage_cache`, and returns the local file path.
2. **train_model** – reads the cached dataset, trains the configured model, and logs metrics/artifacts to MLflow.
3. **save_results** – computes and uploads the final artifact location.

Each task runs inside the `ml_pipeline_worker` image built from `containers/ml_pipeline_worker`. The project directory (including `.env`, data, and scripts) is mounted at `/srv/pipeline` in every task container so artifacts can be shared between steps.

## End-to-End Bring-Up (Airflow + MLflow + BentoML)
Follow the sequence below when you want to stand up the complete lab stack on a fresh machine.

1. **Prerequisites**
   - Docker Desktop (compose v2) with at least 6 GB RAM available for containers.
   - Python 3.10+ on the host for running MLflow/BentoML CLIs.
   - Access credentials for the object store holding training data and artifacts.
2. **Prepare configuration**
   - Copy `.env.template` to `.env` and fill in:
     - `AIRFLOW_PROJ_DIR` → absolute path to this folder.
     - `ML_TASK_IMAGE`/`AIRFLOW_IMAGE_NAME` if you maintain your own images.
     - `MLFLOW_TRACKING_URI` (defaults to `http://host.docker.internal:5000` so containers can reach an MLflow server on the host).
     - All `OBJECT_STORAGE_*` values and any extras you need inside Airflow.
   - Populate `bento_service/.envbento` with the same storage credentials plus (optionally) `BENTOML_MODEL_TAG`.
3. **Start MLflow tracking server on the host**
   ```powershell
   python -m venv mlflow-env
   .\mlflow-env\Scripts\Activate.ps1
   pip install mlflow boto3
   mlflow server `
     --host 0.0.0.0 --port 5000 `
     --backend-store-uri sqlite:///mlruns.db `
     --default-artifact-root s3://<bucket>/mlflow-artifacts/
   ```
   - Use `file:./mlruns` for `--default-artifact-root` if you prefer to store artifacts locally.
   - Ensure the same AWS/MinIO variables configured in `.env` are exported in this terminal (MLflow will forward uploads to your bucket).
4. **Launch the Airflow stack**
   ```powershell
   cd Airflow-dynamic-param
   docker compose up airflow-init
   docker compose up -d
   ```
   - The UI is available at `http://localhost:8080` (default creds: `airflow` / `airflow`).
   - Verify that the `ml_dynamic_pipeline_with_ingestion_and_training` DAG is healthy before triggering runs.
5. **Trigger a training run**
   - Adjust one of the `param*.json` files (or create a new one) to describe your dataset, hyperparameters, and artifact destination.
   - From the Airflow UI, trigger the DAG with your JSON payload (paste into "Config"); alternatively:
     ```powershell
     docker compose run --rm airflow-worker \
       airflow dags trigger ml_dynamic_pipeline_with_ingestion_and_training \
       --conf "$(Get-Content param_driver_dw.json -Raw)"
     ```
   - Confirm in the MLflow UI (`http://localhost:5000`) that the run logged metrics and the artifact URI.
6. **Import the artifact into BentoML**
   - Inside `bento_service/`, install dependencies (`python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt`).
   - Load the credentials from `.envbento` and execute:
     ```powershell
     python import_model.py `
       --model-path s3://<bucket>/models/random_forest.pkl `
       --bento-tag driver_prediction:2024-11-25
     ```
7. **Serve and test predictions**
   - Run `bentoml serve service:svc --reload` (or `bentoml build && bentoml serve driver_prediction_service:latest`).
   - Send a smoke-test request:
     ```powershell
     curl -X POST http://127.0.0.1:3000/predict `
       -H "Content-Type: application/json" `
       -d "{\"instances\":[{\"driver_id\":1003,...}]}"
     ```
   - Iterate on Bento configuration or redeploy the model as needed.

## Environment & Config
1. Copy `.env.template` to `.env` and fill in the required values:
   - Airflow image/tag, ML worker image, and project path.
   - `AIRFLOW_UID` matching your host user (keeps file ownership sane).
   - `_PIP_ADDITIONAL_REQUIREMENTS` for extra python packages in Airflow.
   - `MLFLOW_TRACKING_URI` pointing to your MLflow server.
   - All `OBJECT_STORAGE_*` settings so ingestion & training can access S3/MinIO.
2. Ensure Docker has access to the host socket (`/var/run/docker.sock`) as defined in `docker-compose.yaml`.
3. Build/push the `ml_pipeline_worker` image whenever code in `containers/ml_pipeline_worker` changes:  
   `docker build -t mlops/pipeline-worker:latest containers/ml_pipeline_worker`
4. Provide dataset/model parameters via JSON (see below) when triggering the DAG.

## Parameter Files (e.g. `param.json`)
`param.json` stores a ready-made parameter payload for the dynamic DAG. The keys map directly to `dag.params`:

- `data_source`, `data_format`, `input_schema_version`: describe the dataset to ingest.
- `model_name`, `model_version`, `hyperparameters`: control which model is trained and with what settings (e.g. logistic regression solver/C/max_iter).
- `training_scenario`, `target_output_path`, `test_size`, `random_state`: control training behaviour and artifact destinations.
- `ingestion_config`: optional overrides for the ingestion step. Use `object_storage.object_key` + `bucket` to pull from S3 and prefer **relative** `cache_dir` paths so the file remains inside `/srv/pipeline`.
- `target_column`: column name used as the prediction target.

You can duplicate `param.json` (or use `param_gradient_boost.json`) as a template for different scenarios; pass the JSON when triggering the DAG through the Airflow UI or CLI.

Example (`param.json.example`):
```json
{
  "data_source": "s3://bucket-store/data/bank.csv",
  "data_format": "csv",
  "input_schema_version": "v2",
  "model_name": "logistic_regression",
  "model_version": "2",
  "hyperparameters": {
    "C": 0.5,
    "solver": "liblinear",
    "max_iter": 200
  },
  "training_scenario": "incremental_train",
  "target_output_path": "s3://bucket-store/output/bank_marketing/",
  "test_size": 0.15,
  "random_state": 123,
  "ingestion_config": {
    "object_storage": {
      "enabled": true,
      "bucket": "bucket-store",
      "object_key": "data/bank.csv",
      "cache_dir": "data/cache_bank"
    },
    "file_extension": ".csv"
  },
  "target_column": "deposit"
}
```

## Using the Driver Metrics Warehouse
If you want the DAG to train on the driver-behaviour dataset that lives inside the new `data_warehouse/` stack:
1. Follow `data_warehouse/README.md` to start Postgres + MinIO, run `scripts/bootstrap_dw.py`, and publish the mart with `scripts/publish_training_dataset.py --s3-uri s3://driver-training/driver_training_dataset.csv`.
2. Copy `.env.sample` from `data_warehouse/` into this directory (or align the `OBJECT_STORAGE_*` values manually) so the Airflow containers can authenticate with MinIO.
3. Trigger the DAG with `param_driver_dw.json`. It points to the warehouse export, uses `target_high_value` as the label, and saves trained models back into the same MinIO bucket under `models/`.

## Serving the Trained Model with BentoML
Use `bento_service/` to expose any artifact produced by the DAG:
1. Run the pipeline and capture the artifact URI printed by `train_model`.
2. `cd bento_service && pip install -r requirements.txt`.
3. `python import_model.py --model-path <artifact_path> --bento-tag driver_prediction:latest` to register the pickle with BentoML.
4. `bentoml serve service:svc --reload` for local testing or `bentoml build && bentoml serve driver_prediction_service:latest` for production-ready bundles. The REST API accepts `POST /predict` with a JSON body `{"instances": [...]}` that mirrors the training features.
