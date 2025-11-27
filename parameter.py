# from datetime import datetime
# from airflow import DAG
# from airflow.decorators import task
# from airflow.models.param import Param

# with DAG(
#     dag_id="generate_episode_links",
#     start_date=datetime(2024, 1, 1),
#     schedule=None,          # chỉ chạy manual
#     catchup=False,
#     params={                # 👈 chính chỗ này tạo ra form “DAG conf Parameters”
#         "model_id": Param(
#             default="",
#             type="string",
#             description="ID của model dùng để phân tích episode links"
#         ),
#         "row_limit": Param(
#             default=1000,
#             type="integer",
#             minimum=1,
#             description="Số dòng tối đa cần xử lý"
#         ),
#     },
# ) as dag:

#     @task
#     def generate_links(model_id: str, row_limit: int, **_):
#         # logic của bạn ở đây
#         print(f"Using model_id={model_id}, row_limit={row_limit}")

#     # dùng params trong DAG
#     generate_links(
#         model_id="{{ params.model_id }}",
#         row_limit="{{ params.row_limit }}",
#     )
