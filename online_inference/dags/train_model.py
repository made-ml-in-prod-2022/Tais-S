from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import pendulum


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=pendulum.today('UTC').add(days=-1),
) as dag:

    preprocess = DockerOperator(
        image="airflow-model-training",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}2",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    split = DockerOperator(
        image="airflow-model-training",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-model-training",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/predicted/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-model-training",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/predicted/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    preprocess >> split >> train >> validate
