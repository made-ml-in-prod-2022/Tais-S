from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from docker.types import Mount
import pendulum


def custom_failure_function(context):
    dag_run = context.get("dag_run")
    task_instances = dag_run.get_task_instances()
    print("============================================ Task instances failed:", task_instances)


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    'on_failure_callback': custom_failure_function,
}

with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=pendulum.today('UTC').add(days=-1),
) as dag:

    sensor = FileSensor(task_id="file_sensor",
                        poke_interval=30,
                        filepath="/opt/airflow/data/raw/{{ ds }}")

    split = DockerOperator(
        image= "airflow-model-split",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/split/{{ ds }}",
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )

    transform = DockerOperator(
        image="airflow-model-transform",
        command="--input-dir /data/split/{{ ds }} --output-dir /data/processed/{{ ds }} --transformer-dir /data/models/{{ ds }}",
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-transform",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-model-train",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-model-validate",
        command="--input-dir /data/split/{{ ds }} --output-dir /data/models/{{ ds }}",
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )

    sensor >> split >> transform >> train >> validate
