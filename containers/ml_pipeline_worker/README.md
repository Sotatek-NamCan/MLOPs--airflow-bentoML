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
