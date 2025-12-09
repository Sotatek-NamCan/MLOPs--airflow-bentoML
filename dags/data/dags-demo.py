from airflow import DAG
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from datetime import datetime, timedelta

# Constants
CLUSTER_NAME = "apache-airflow-worker-cluster"
TASK_DEFINITION = "airflow-worker"
LAUNCH_TYPE = "FARGATE"
REGION_NAME = "eu-central-1"
AWS_CONN_ID = "aws_default"
SUBNETS = ["subnet-"]
SECURITY_GROUPS = ["sg-"]

# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 18),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ECS Task Overrides
def get_overrides():
    return {
        "containerOverrides": [
            {
                "name": "airflow-worker",
                "command": ["python","domain_analysis.py"]
            }
        ]
    }

# Network Configuration
network_configuration = {
    "awsvpcConfiguration": {
        "subnets": SUBNETS,
        "securityGroups": SECURITY_GROUPS,
        "assignPublicIp": "ENABLED",
    }
}

# DAG definition
with DAG(
    dag_id="analyse-domains-and-scrapes",
    default_args=default_args,
    catchup=False,
    schedule_interval=None
) as dag:

    run_ecs_task = EcsRunTaskOperator(
        task_id="run_ecs_fargate_task",
        cluster=CLUSTER_NAME,
        task_definition=TASK_DEFINITION,
        launch_type=LAUNCH_TYPE,
        region_name=REGION_NAME,
        aws_conn_id=AWS_CONN_ID,
        overrides=get_overrides(),
        network_configuration=network_configuration,
    )

    run_ecs_task
