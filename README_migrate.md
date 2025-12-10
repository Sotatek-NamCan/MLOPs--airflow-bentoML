# README Migrate

Huong dan nay giai thich cac buoc can lam khi chuyen DAG `dags/data/dynamic.py` sang Airflow + AWS ECS va dong goi lai anh `containers/ml_pipeline_worker`.

## 1. Thu muc `containers/ml_pipeline_worker`

1. **Cap nhat ma nguon CLI**
   - Cac tac vu trong DAG goi truc tiep cac module `pipeline_worker.cli.ingest_data`, `pipeline_worker.cli.validate_data`, `pipeline_worker.cli.train_model`, `pipeline_worker.cli.save_results`.
   - Moi thay doi luong du lieu phai duoc cap nhat trong cac file CLI tuong ung va commit cung voi thay doi DAG.

2. **Cap nhat phu thuoc**
   - Sua `containers/ml_pipeline_worker/pyproject.toml` khi them bot goi thu vien.
   - Kiem tra nhanh bang `pip install -e .` (hoac `poetry install`) ngay tai thu muc nay.

3. **Build & push image**
   ```powershell
   cd containers/ml_pipeline_worker
   docker build -t <registry>/ml-pipeline-worker:<tag> .
   docker push <registry>/ml-pipeline-worker:<tag>
   ```
   - Voi AWS ECS nen push len ECR (vi du `123456789012.dkr.ecr.eu-central-1.amazonaws.com/ml-pipeline-worker:latest`).
   - Task definition tren ECS se tai image nay, vi vay de dang thay doi bang cach doi tag trong bien moi truong.

4. **Smoke test CLI**
   ```powershell
   docker run --rm -it `
     --env-file ..\..\ .env `
     <registry>/ml-pipeline-worker:<tag> `
     python -m pipeline_worker.cli.validate_data --help
   ```
   - Dat chu y den cac bien `OBJECT_STORAGE_*`, `MLFLOW_*`, `PIPELINE_*` phai ton tai trong `.env` vi Airflow se truyen nguyen ven vao container.

## 2. Thu muc `dags/data/dynamic.py`

Phien ban moi su dung `EcsRunTaskOperator` va gan cac constant theo mau trong `dags/data/dags-demo.py`.

1. **Hang so ECS**
   - `CLUSTER_NAME`, `TASK_DEFINITION`, `LAUNCH_TYPE`, `REGION_NAME`, `AWS_CONN_ID`, `SUBNETS`, `SECURITY_GROUPS`, `CONTAINER_NAME`, `NETWORK_CONFIGURATION` duoc khoi tao giong demo.
   - Moi gia tri co the override qua cac bien:
     - `ML_PIPELINE_AWS_CONN_ID`
     - `ML_PIPELINE_AWS_REGION`
     - `ML_PIPELINE_ECS_CLUSTER`
     - `ML_PIPELINE_ECS_TASK_DEFINITION`
     - `ML_PIPELINE_ECS_LAUNCH_TYPE`
     - `ML_PIPELINE_ECS_CONTAINER_NAME`
     - `ML_PIPELINE_ECS_SUBNETS` (CSV)
     - `ML_PIPELINE_ECS_SECURITY_GROUPS` (CSV)
   - `NETWORK_CONFIGURATION` la dictionary giong file demo (khong dung ham trung gian) de Airflow luon day dung `subnets`, `securityGroups`, `assignPublicIp`.

2. **BASE_ECS_OPERATOR_KWARGS**
   - Gom toan bo tham so chung (`cluster`, `task_definition`, `launch_type`, `aws_conn_id`, `region_name`, `network_configuration`, `wait_for_completion`).
   - Moi task goi `_pipeline_task` se giai nen dict nay, giup ban khong phai lap lai cau hinh cho tung `EcsRunTaskOperator`.

3. **Dynamic parameters**
   - Phan `params={...}` cua DAG giu nguyen, cho phep truyen JSON tu Airflow UI hoac CLI.
   - Cac duong dan S3 (`RUN_S3_PREFIX`, `DATASET_S3_URI`, `MODEL_ARTIFACT_URI`...) duoc xay dung tu `ML_PIPELINE_ARTIFACT_BUCKET` va `ML_PIPELINE_ARTIFACT_PREFIX`. Neu khong set, DAG se bao loi ngay khi import.

4. **Moi truong va artifact**
   - `BASE_ENV` gom cac bien `OBJECT_STORAGE_*`, `MLFLOW_*`, `MODEL_ARTIFACT_CACHE_DIR`. Airflow se chen vao override cua container.
   - Moi task chi viec chuyen lenh `python -m pipeline_worker.cli.<command>` va upload artifact len S3 thong qua cac tham so `--upload-bucket`, `--upload-object-key` (ingest) hoac `--report-upload-*`, `--target-output-path`.

## 3. Checklist truoc khi trigger DAG tren ECS

- [ ] Build va push lai image `containers/ml_pipeline_worker` sau khi thay doi ma nguon.
- [ ] Tao hoac cap nhat Task Definition tren ECS tro dung image va co command entrypoint `python`.
- [ ] Cap nhat `.env` (hoac Airflow Variables) voi tat ca bien `ML_PIPELINE_*`, `OBJECT_STORAGE_*`, `MLFLOW_*`.
- [ ] Sync file `dags/data/dynamic.py` len Airflow scheduler va kiem tra no su dung `EcsRunTaskOperator` (khong con DockerOperator).
- [ ] Tu giao dien Airflow, trigger mot run bang JSON trong `param.json` va theo doi tren AWS Console/S3 de chac chan artifact duoc tao dung noi.

Hoan thanh cac buoc tren se giup qua trinh migrate tu DockerOperator sang EcsRunTaskOperator dien ra on dinh va de quan ly hon.
