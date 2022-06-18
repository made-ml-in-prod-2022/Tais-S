from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
import pendulum
from docker.types import Mount


with DAG(
    dag_id="generate_train_data",
    start_date=pendulum.today('UTC').add(days=-1),
    schedule_interval="@daily",
    max_active_runs=1,
) as dag:

    generate_data = DockerOperator(
        image="airflow-model-generate",
        command="--input-dir /data/raw/ --output-dir /data/raw/{{ ds }}",
        environment={"path_to_model": "/data/models/2022-06-18"},
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )
