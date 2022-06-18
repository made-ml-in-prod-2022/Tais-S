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
        "make_predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=pendulum.today('UTC').add(days=-1),
) as dag:

    predict = DockerOperator(
        image="airflow-model-predict",
        command="--data-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )
