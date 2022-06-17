from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import pendulum


def get_float_distributions(n):
    import numpy as np
    # эмпирически подобранные распределения
    float_distributions = {"age": np.random.uniform(1, 83, n),
                           "avg_glucose_level": np.random.lognormal(4.8, 0.3, n),
                           "bmi": np.random.lognormal(3.2, 0.3, n)}
    return float_distributions


def generate_column(data, col_name, n, float_distributions):
    import random
    col = data[col_name]
    if col.dtypes == "float64":
        generated_col = [round(x, 1) for x in float_distributions[col_name]]
    else:
        generated_col = []
        col_values = dict(data[col_name].value_counts(normalize=True))
        for val, share in col_values.items():
            generated_col += [val for _ in range(round(share * n))]
        random.shuffle(generated_col)
    generated_col = generated_col[:(n-1)]
    return generated_col


def generate_synthetic_data(initial_data_dir, output_data_dir, output_target_dir, n_rows=500):
    import pandas as pd
    data = pd.read_csv(initial_data_dir)
    id_col = [x for x in range(n_rows)]
    synthetic_data = pd.DataFrame(id_col, columns=["id"])
    float_distributions = get_float_distributions(n_rows)
    for col_name in list(data.columns.drop("id")):
        generated_col = generate_column(data, col_name, n_rows+1, float_distributions)
        synthetic_data[col_name] = generated_col

    synthetic_target = synthetic_data["stroke"]
    synthetic_data = synthetic_data.drop("stroke", 1)

    with open(output_data_dir, "wb") as f:
        synthetic_data.to_csv(f, index=False)
    with open(output_target_dir, "wb") as f:
        synthetic_target.to_csv(f, index=False)


with DAG(
    dag_id="generate_train_data",
    start_date=pendulum.today('UTC').add(days=-1),
    schedule_interval="@daily",
    max_active_runs=1,
) as dag:

    create_dir = BashOperator(
        task_id="create_directory",
        bash_command="mkdir -p /opt/airflow/data/raw/{{ds}} && chmod 777 /opt/airflow/data/raw/{{ds}} -R",
    )

    generate_data = PythonOperator(
        task_id="generate_data",
        python_callable=generate_synthetic_data,
        op_kwargs={
            "initial_data_dir": "/opt/airflow/data/raw/initial_data.csv",
            "output_data_dir": "/opt/airflow/data/raw/{{ ds }}/data.csv",
            "output_target_dir": "/opt/airflow/data/raw/{{ ds }}/target.csv",
            "n_rows": 500,
        }
    )

    log = BashOperator(
        task_id="log_generated_data",
        bash_command="echo 'Generated data for date $({{ ds }})'",
    )

    create_dir >> generate_data >> log
