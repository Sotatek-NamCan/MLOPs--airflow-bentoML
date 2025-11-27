# BentoML Serving for Airflow Dynamic Pipeline Models

This folder contains everything required to expose a model trained by the `ml_dynamic_pipeline_with_ingestion_and_training` DAG through a BentoML API.

## Workflow Overview
1. **Train the model via Airflow** and note the artifact path that the `train_model` task prints (local path or `s3://` URI).
2. **Install serving dependencies** (ideally in a virtual environment):
   ```powershell
   cd Airflow-dynamic-param/bento_service
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. **Provide object-store credentials**: create `bento_service/.envbento` (ignored by git) and add:
   ```ini
   OBJECT_STORAGE_BUCKET=cpnam-s3-tfbackend
   OBJECT_STORAGE_ENDPOINT_URL=https://s3.ap-northeast-2.amazonaws.com
   OBJECT_STORAGE_REGION=ap-northeast-2
   OBJECT_STORAGE_ACCESS_KEY=xxx
   OBJECT_STORAGE_SECRET_KEY=yyy
   # optional override
   # BENTOML_MODEL_TAG=driver_prediction:2025-11-25
   ```
   The new `import_model.py` loads this file automatically, so the Bento workflow stays decoupled from the Airflow worker package.
4. **Import the artifact into BentoML**:
   ```powershell
   python import_model.py `
     --model-path s3://driver-training/models/random_forest_classifier_v1.pkl `
     --bento-tag driver_prediction:2024-11-25
   ```
   The script understands both local paths and S3/MinIO URIs (it now reads creds straight from `.envbento`). It registers the estimator inside the Bento model store with batchable `predict`/`predict_proba` signatures.
5. **Serve locally**:
   ```powershell
   bentoml serve service:svc --reload
   ```
   or **build a distributable Bento**:
   ```powershell
   bentoml build
   bentoml serve driver_prediction_service:latest
   # Optional container image
   bentoml containerize driver_prediction_service:latest
   docker run -p 3000:3000 driver_prediction_service:latest
   ```

## Testing the Model
- **Check that the model import succeeded**:
  ```powershell
  bentoml models list
  bentoml models get driver_prediction:latest --summary
  ```
- **Smoke-test the API locally** (use the same payload structure that Airflow training used):
  ```powershell
  bentoml serve service:svc --reload
  ```
  In another terminal, send a request:
  ```powershell
  curl -X POST http://127.0.0.1:3000/predict ^
    -H "Content-Type: application/json" ^
    -d "{\"instances\":[{\"driver_id\":1003,\"conv_rate\":0.58,\"acc_rate\":0.61,\"avg_daily_trips\":640,\"activity_index\":0.59,\"trip_completion_estimate\":371}]}"
  ```
  You should get the `predictions` array (and `probabilities` if the estimator exposes `predict_proba`). Feel free to paste multiple feature rows inside `instances` to test batching.

## API Contract
- **Endpoint**: `POST /predict`
- **Request body**:
  ```json
  {
    "instances": [
      {
        "driver_id": 1003,
        "conv_rate": 0.58,
        "acc_rate": 0.61,
        "avg_daily_trips": 640,
        "activity_index": 0.59,
        "trip_completion_estimate": 371
      }
    ]
  }
  ```
  The keys must align with the columns that were available during training (all columns except the target column). The payload can contain multiple rows.
- **Response body**:
  ```json
  {
    "predictions": [1],
    "probabilities": [[0.12, 0.88]]
  }
  ```
  When the underlying estimator exposes `predict_proba`, the Bento also returns probabilities.

## Configuration Notes
- The service loads the Bento model tag specified by `BENTOML_MODEL_TAG` (defaults to `driver_prediction:latest`). Set the variable in `.envbento` or your shell before running `bentoml serve` if you want to use a different tag.
- Credentials are isolated to `.envbento` so the serving code can run outside the Airflow repo layout. Populate the `OBJECT_STORAGE_*` keys with whichever object store (AWS S3, MinIO, etc.) you use; no extra imports are required.
- If you already maintain `.env` for Airflow, you can copy the storage-related lines into `.envbento` to keep both stacks in sync.
