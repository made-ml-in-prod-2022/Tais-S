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
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    'on_failure_callback': custom_failure_function,
}


with DAG(
        "make_predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=pendulum.today('UTC').add(days=-1),
) as dag:

    sensor = FileSensor(task_id="file_sensor",
                        poke_interval=30,
                        filepath="/opt/airflow/data/raw/{{ ds }}")

    predict = DockerOperator(
        image="airflow-model-predict",
        command="--data-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        docker_url='unix://var/run/docker.sock',
        task_id="docker-airflow-predict",
        environment={"path_to_model": "/data/models/2022-06-18"},
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source="D:/ML files/MADE_2sem/ML in prod/Tais-S/airflow-ml-dags/data/", target="/data", type='bind')]
    )

    sensor >> predict
