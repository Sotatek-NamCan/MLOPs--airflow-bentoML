# README Migrate

Tài liệu này mô tả các bước cần thực hiện khi muốn **di chuyển DAG `dags/data/dynamic.py` sang kiến trúc chạy trên ECS** và **đóng gói lại thư mục `containers/` (đặc biệt là `containers/ml_pipeline_worker`)** để các task mới hoạt động ổn định.

## 1. Việc cần làm với `containers/ml_pipeline_worker`

Thư mục này chứa toàn bộ mã nguồn và Dockerfile dùng để xây dựng image mà các task trong DAG sẽ chạy. Khi migrate, luôn làm các bước sau:

1. **Cập nhật mã nguồn CLI**  
   - Các entrypoint mà DAG gọi tới là `pipeline_worker.cli.ingest_data`, `pipeline_worker.cli.validate_data`, `pipeline_worker.cli.train_model` và `pipeline_worker.cli.save_results` (xem dưới `src/pipeline_worker/cli/`).  
   - Nếu thay đổi luồng xử lý, hãy chỉnh trong các file tương ứng (`containers/ml_pipeline_worker/src/pipeline_worker/cli/*.py`) và commit cùng lúc với thay đổi DAG.

2. **Cập nhật dependency**  
   - Sửa `containers/ml_pipeline_worker/pyproject.toml` nếu cần thêm/thay gói.  
   - Có thể kiểm tra nhanh bằng `pip install -e .` (hoặc `poetry install`) ngay trong thư mục này.

3. **Build & push image mới**  
   ```powershell
   cd containers/ml_pipeline_worker
   docker build -t <your-registry>/ml-pipeline-worker:<tag> .
   docker push <your-registry>/ml-pipeline-worker:<tag>
   ```
   - Với AWS ECS, nên đẩy lên Amazon ECR (ví dụ `aws_account_id.dkr.ecr.region.amazonaws.com/ml-pipeline-worker:latest`).
   - Ghi nhớ tên image để gán cho `ML_PIPELINE_ECS_TASK_DEFINITION` (khi tạo task definition bạn sẽ tham chiếu tới image này).

4. **Kiểm tra cục bộ**  
   - Có thể chạy thử CLI trực tiếp:  
     `docker run --rm -it --env-file ../../.env <image-tag> python -m pipeline_worker.cli.validate_data --help`.
   - Đảm bảo các biến môi trường như `OBJECT_STORAGE_*`, `MLFLOW_TRACKING_URI` hoạt động đúng (được truyền từ Airflow vào task).

## 2. Việc cần làm với `dags/data/dynamic.py`

Phiên bản mới dùng `airflow.providers.amazon.aws.operators.ecs.EcsRunTaskOperator`. Các bước migrate:

1. **Thiết lập biến môi trường cho Airflow**  
   - Tối thiểu cần:  
     `ML_PIPELINE_AWS_CONN_ID`, `ML_PIPELINE_ECS_CLUSTER`, `ML_PIPELINE_ECS_TASK_DEFINITION`, `ML_PIPELINE_ECS_LAUNCH_TYPE`, `ML_PIPELINE_ECS_CONTAINER_NAME`.  
   - Với mạng VPC, cung cấp `ML_PIPELINE_ECS_SUBNETS` (chuỗi CSV) và `ML_PIPELINE_ECS_SECURITY_GROUPS`.  
   - Nếu cần phiên bản platform cụ thể: `ML_PIPELINE_ECS_PLATFORM_VERSION`.  
   - Dòng artifact chung: `ML_PIPELINE_ARTIFACT_BUCKET` và (tuỳ chọn) `ML_PIPELINE_ARTIFACT_PREFIX`.

2. **Cập nhật connections/permissions**  
   - Tạo Airflow Connection có ID trùng `ML_PIPELINE_AWS_CONN_ID`, chứa role hoặc access key có quyền `ecs:RunTask`, `iam:PassRole`, cũng như quyền đọc/ghi S3 bucket nêu ở trên.
   - Task definition ECS phải mount biến môi trường từ Airflow (operator đính vào trường `overrides.containerOverrides.environment`). Đảm bảo container hiểu các biến `PIPELINE_PROJECT_ROOT`, `PIPELINE_ENV_FILE`, `OBJECT_STORAGE_*`, `MLFLOW_*`, …

3. **Tùy chỉnh lệnh chạy (nếu cần)**  
   - Mỗi task dùng `_pipeline_task` để gom tham số chung. Command mặc định (trong `dynamic.py`) chạy `python -m pipeline_worker.cli.<...>`. Khi bổ sung stage mới, tái sử dụng `_pipeline_task(...)` và truyền `command` list phù hợp.
   - `_stringify_command` tự động chuyển list → list[str]; tránh đưa object phức tạp ngoài phạm vi JSON.

4. **Quản lý artifact qua S3**  
   - DAG thiết lập `RUN_S3_PREFIX`, `DATASET_S3_URI`, `MODEL_ARTIFACT_URI`... dựa trên `ML_PIPELINE_ARTIFACT_BUCKET`. Kiểm tra bucket tồn tại và worker container có quyền `s3:PutObject`/`s3:GetObject`.
   - Nếu vẫn cần lưu file local, chỉnh `pipeline_worker` để tải lại từ S3 (định vị qua các biến ở `BASE_ENV`).

5. **Kích hoạt DAG sau khi migrate**  
   - Deploy file `dynamic.py` mới lên Airflow scheduler.  
   - Từ UI, tạo run thử bằng JSON trong `param*.json`. Kiểm tra trong AWS Console xem ECS task xuất hiện và hoàn thành, đồng thời kiểm chứng artifact thực sự ghi vào bucket mong muốn.

## Checklist nhanh trước khi triển khai

- [ ] Thư mục `containers/ml_pipeline_worker` đã build image mới và push lên registry/ECR.  
- [ ] Task Definition ECS trỏ đúng image vừa build và expose command `python -m pipeline_worker.cli.*`.  
- [ ] `.env` hoặc Airflow Variables chứa đủ các biến `ML_PIPELINE_*`, `OBJECT_STORAGE_*`, `MLFLOW_*`.  
- [ ] `dags/data/dynamic.py` được sync lên Airflow và không còn tham chiếu tới DockerOperator cũ.  
- [ ] Đã chạy thử 1 DAG run và xác nhận dataset, validation report, model artifact xuất hiện trên S3.

Hoàn thành các bước trên sẽ giúp môi trường Airflow chuyển sang mô hình chạy task trên ECS một cách an toàn và dễ bảo trì.
