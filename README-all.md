# Airflow Dynamic ML Pipeline – Detailed Guide

This repository packages a complete Airflow stack plus a containerised ML worker that can ingest datasets, validate them with Great Expectations, train a model, and publish artifacts. The stack is optimised for local experimentation with Docker Compose but mirrors production-style components (CeleryExecutor, S3-compatible storage, MLflow tracking, etc.).

The goals of this document are:

- Explain how the Airflow DAGs are wired and what each task does.
- Show the control-flow, data-flow, and failure-handling in detail.
- Highlight how supporting modules (ingestion, validation, storage, training) cooperate with the DAGs.

---

## 1. Runtime Architecture Overview

| Component | Location | Role |
| --- | --- | --- |
| Airflow stack (scheduler, workers, Redis, Postgres) | `docker-compose.yaml` | Orchestrates DAGs; all containers share `./dags`, `./logs`, `./config`, and the Docker socket. |
| Dynamic DAGs | `dags/data/dynamic.py` | Define the ML workflow with DockerOperators running inside the ML worker image. |
| ML pipeline worker image | `containers/ml_pipeline_worker` | Python package installed into a slim image; exposes CLI entrypoints for ingestion, validation, training, and result publishing. |
| Artifact bucket/prefix | `s3://<bucket>/<prefix>` derived from `ML_PIPELINE_ARTIFACT_BUCKET`/`OBJECT_STORAGE_BUCKET` | Every task reads and writes intermediate datasets/artifacts via S3, keeping the worker containers stateless. |
| Object storage + MLflow | External services referenced via env vars | Provide durable data sources and tracking; credentials are passed in via `.env`. |

The Airflow containers mount the Docker socket so each task can spin up the `ml_pipeline_worker` image. Each task runs in its own container and hands off every intermediate artifact via S3 URIs, so no shared host volume is required.

---

## 2. DAG Catalog

| DAG ID | File | Schedule | Summary |
| --- | --- | --- | --- |
| `ml_dynamic_pipeline_with_ingestion_and_training` | `dags/data/dynamic.py` | Manual (`schedule=None`, `catchup=False`) | Parameter-driven ML pipeline: ingest dataset → validate with Great Expectations → train model → emit artifact destination. |

At the moment there is a single DAG, but the helper classes (`PipelineDockerOperator`, CLI entrypoints) are reusable for future DAGs.

---

## 3. Control Flow of `ml_dynamic_pipeline_with_ingestion_and_training`

```
ingest_dataset
      │  (XCom: local dataset path)
      ▼
validate_dataset
      │  (fails fast on invalid data)
      ▼
train_model
      │  (XCom: model artifact path)
      ▼
save_results
      │  (stdout: final destination)
```

**Key characteristics**

- **Fully parameterised:** Every run receives a JSON payload (`params`) that controls data source, schema, hyperparameters, artifact target, and validation rules.
- **Container-per-task isolation:** Each task invokes the ML worker image with module-mode Python (`python -m pipeline_worker.cli.<task>`).
- **Shared context via environment:** `BASE_ENV` in `dynamic.py` forwards object-storage, MLflow, and cache settings to all tasks.
- **Deterministic XCom usage:** `ingest_dataset` and `train_model` push their outputs explicitly; downstream tasks pull via templated `ti.xcom_pull`.
- **Validation gate:** Training will never run if the dataset violates required expectations, preventing wasted compute or polluted models.

---

## 4. DAG Mechanics in Detail

### 4.1 Common helpers (`dags/data/dynamic.py`)

1. **`PipelineDockerOperator`** wraps Airflow’s DockerOperator to:
   - Normalize `command`/`entrypoint` lists into plain strings (fixes template rendering quirks).
   - Force `ti.hostname` to a predictable value (`LOG_STREAM_HOST`), improving log aggregation when tasks run inside nested containers.
2. **S3-first hand-off:** `ARTIFACT_BASE_PREFIX` is built from `ML_PIPELINE_ARTIFACT_BUCKET` (or `OBJECT_STORAGE_BUCKET`) and is required, so every task exchanges inputs/outputs through run-scoped paths like `s3://<bucket>/ml-pipeline-runs/<run_id>/...`.
3. **Environment propagation:** `BASE_ENV` forwards `OBJECT_STORAGE_*`, `MLFLOW_*`, and cache settings, plus two helper vars:
   - `PIPELINE_PROJECT_ROOT`: points to `/opt/pipeline` (the path baked into the worker image).
   - `PIPELINE_ENV_FILE`: absolute path to `/opt/pipeline/.env`, enabling modules like `storage.py` to load credentials automatically when running inside the container.

### 4.2 Parameters (Airflow `params`)

| Parameter | Purpose | Typical source |
| --- | --- | --- |
| `data_source`, `data_format`, `input_schema_version` | Tell the ingestion CLI where and how to load data. | `param.json`, `param_driver_dw.json`, etc. |
| `model_name`, `model_version`, `hyperparameters` | Select estimator type and hyperparameters used in `train_utils.select_model`. | Payload supplied on trigger. |
| `training_scenario`, `target_output_path`, `test_size`, `random_state` | Control evaluation split and artifact placement. | Payload. |
| `ingestion_config` | Extra ingestion overrides (object storage info, cache dirs, zip handling). | Payload. |
| `target_column` | Label column enforced by ingestion + validation + training. | Payload. |
| `data_validation` | Great Expectations overrides (min rows, non-null columns, ranges). | Payload; defaults to `{"min_row_count": 10}`. |

Parameters are rendered with the `render_template_as_native_obj=True` flag, meaning dict/list params stay as Python objects when passed to the CLI JSON arguments.

### 4.3 Task-by-task breakdown

#### Task: `ingest_dataset`

- **Command:** `python -m pipeline_worker.cli.ingest_data ...`
- **Inputs:** `params.data_source`, `params.data_format`, `params.input_schema_version`, `params.ingestion_config`.
- **Logic (`pipeline_worker/cli/ingest_data.py` + `ingestion.py`):**
  1. Merges CLI args with the `ingestion_config` JSON.
  2. If the data source is `s3://bucket/key`, it auto-populates `object_storage.bucket`/`object_key`.
  3. Downloads the dataset using `build_storage_client()` (S3-compatible), storing it temporarily inside the container before re-uploading it to the run-specific S3 prefix.
  4. Picks a `DataIngestor` based on file extension (`csv`, `parquet`, `json`, `xlsx`, `tsv`, zipped archives, etc.) and reads into a DataFrame.
  5. Writes the local path to stdout, which Airflow captures and stores in XCom (`return_value`).
- **Failure modes:** Missing credentials (raises `ObjectStorageConfigurationError`), unsupported extension, dataset not found, etc.

#### Task: `validate_dataset`

- **Command:** `python -m pipeline_worker.cli.validate_data ...`
- **Inputs:** Dataset path from XCom, `params.data_format`, `params.target_column`, `params.data_validation`.
- **Logic (`cli/validate_data.py` + `validation.py`):**
  1. Loads the dataset with the same `DataIngestorFactory` used during ingestion to avoid drift.
  2. Builds a `ValidationConfig`, automatically ensuring the `target_column` is required and non-null.
  3. Runs a curated suite of Great Expectations checks via `PandasDataset`:
     - Table row count bounds.
     - Column existence.
     - Non-null & uniqueness.
     - Value ranges (per column `min/max`, optional `strict` flags).
     - Allowed categorical values.
  4. Writes a JSON summary to stdout and exits with `0` on success or `1` with failure details. Any failure blocks downstream tasks.
- **Why this gate matters:** It prevents model training on incomplete or malformed data and keeps downstream metrics/MLflow runs trustworthy.

#### Task: `train_model`

- **Command:** `python -m pipeline_worker.cli.train_model ...`
- **Inputs:** Local dataset path from ingestion (not the validation step), `params.target_column`, `params.model_*`, hyperparameters, split sizes, target artifact path.
- **Logic (`cli/train_model.py` + `train_utils.py`):**
  1. Loads the dataset (CSV/Parquet/JSON) and splits into features/target.
  2. Performs a train/test split using `test_size` and `random_state`.
  3. Instantiates one of the supported models (`RandomForestClassifier`, `RandomForestRegressor`, `LogisticRegression`, `LinearRegression`) via `select_model`.
  4. Configures MLflow with `MLFLOW_TRACKING_URI` and propagates S3 credentials (so MLflow can upload artifacts if needed).
  5. Logs params (`model_*`, hyperparameters, scenario, data source) and metrics (accuracy for classifiers, MSE for regressors).
  6. Saves the trained estimator as `<model_name>_v<model_version>.pkl`:
     - Directly under the provided path when `target_output_path` is local.
     - Inside a short-lived temp directory before uploading to the configured S3 bucket/prefix (no shared host folder needed).
  7. Emits the final artifact URI through stdout (captured in XCom).

#### Task: `save_results`

- **Command:** `python -m pipeline_worker.cli.save_results ...`
- **Inputs:** Model artifact path from XCom (`train_model`) and `params.target_output_path`.
- **Logic:** Pure formatting helper—constructs a final destination path (joining base output path with artifact filename) and prints both a human-friendly log (stderr) and the final location (stdout). Downstream systems can consume the printed path or rely on Airflow logs.

---

## 5. Data & Artifact Flow

1. **Run-scoped S3 layout:**
   - `s3://<bucket>/<prefix>/<run_id>/ingested/`: dataset emitted by the ingestion task.
   - `.../validation/`: JSON summary from `validate_dataset`.
   - `.../tuning/`: best hyperparameters from Optuna (if enabled).
   - `.../training/models/`: pickled estimators saved by `train_model`.
2. **Object storage integration:**
   - `build_storage_client()` reads env vars like `OBJECT_STORAGE_ENDPOINT_URL`, `OBJECT_STORAGE_ACCESS_KEY`, etc., optionally loading them from `.env`.
   - With MinIO, point `OBJECT_STORAGE_ENDPOINT_URL` to `http://host.docker.internal:9000` (or whatever map) so in-container clients can resolve it.
   - Both ingestion and artifact upload re-use this client for consistency.
3. **MLflow tracking:**
   - `TRAIN` step sets `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` via `BASE_ENV`.
   - Additional environment propagation ensures boto3 inside MLflow inherits the same credentials, so artifact logging to S3 works even though Airflow runs inside Docker.

---

## 6. Triggering Runs

1. Copy `.env.template` to `.env` and provide:
   - `AIRFLOW_PROJ_DIR` = absolute path to this repo.
   - `ML_TASK_IMAGE` = tag of your `ml_pipeline_worker` build (defaults to `mlops/pipeline-worker:latest`).
   - All `OBJECT_STORAGE_*` and `MLFLOW_*` settings.
2. Build/push the worker image whenever you modify `containers/ml_pipeline_worker`:
   ```powershell
   docker build -t mlops/pipeline-worker:latest containers/ml_pipeline_worker
   ```
3. Start Airflow with Compose:
   ```powershell
   docker compose up airflow-init
   docker compose up -d
   ```
4. Trigger the DAG via UI or CLI:
   ```powershell
   docker compose run --rm airflow-worker `
     airflow dags trigger ml_dynamic_pipeline_with_ingestion_and_training `
     --conf "$(Get-Content param.json -Raw)"
   ```
5. Monitor task logs in the Airflow UI. `validate_dataset` failures will appear early with detailed GE output, while `train_model` logs include MLflow run IDs and accuracy/MSE metrics.

---

## 7. Extending or Adding DAGs

The existing DAG demonstrates the recommended building blocks:

- Import `PipelineDockerOperator` from `dags/data/dynamic.py` (or refactor it into a module) to keep container tasks consistent.
- Always read `params` for runtime variability instead of hard-coding dataset/model settings.
- Persist artifacts by writing them to the shared S3 prefix (or another durable URI) so downstream tasks can consume them.
- Leverage the CLI modules under `pipeline_worker/cli/` as single-responsibility entrypoints; when adding new steps (e.g., feature engineering, evaluation), create a dedicated CLI that can be invoked via the DockerOperator.

When creating additional DAGs, keep the same environment variable keys so base infrastructure (`BASE_ENV`, `.env`) works without duplication.

---

## 8. Troubleshooting Tips

- **`ModuleNotFoundError` inside tasks:** Rebuild `ml_pipeline_worker` to ensure Python dependencies (e.g., `great-expectations`, `boto3`) are baked into the image.
- **Object storage downloads fail:** Confirm `OBJECT_STORAGE_*` vars are visible both to Airflow and inside the worker image. Remember Docker containers cannot see `localhost`; use `host.docker.internal` or the actual host IP.
- **Validation blocking legitimate data:** Override `data_validation` when triggering the DAG (e.g., adjust `min_row_count`, add `value_ranges`).
- **MLflow artifacts missing:** Ensure `MLFLOW_TRACKING_URI` resolves from inside the container and that access keys are exported through `.env`.

---

With this guide you can reason about each stage of the Airflow-driven ML pipeline, trace how data and configuration move between tasks, and safely extend the workflow with new DAGs or tasks.
