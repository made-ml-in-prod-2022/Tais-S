import pandas as pd
import numpy as np
import random
import click
from sklearn.model_selection import train_test_split


def get_float_distributions(n):
    # эмпирически подобранные распределения
    float_distributions = {"age": np.random.uniform(1, 83, n),
                           "avg_glucose_level": np.random.lognormal(4.8, 0.3, n),
                           "bmi": np.random.lognormal(3.2, 0.3, n)}
    return float_distributions


def generate_column(data, col_name, n, float_distributions):
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


def generate_synthetic_data(data, n=200):
    id_col = [x for x in range(n)]
    synthetic_data = pd.DataFrame(id_col, columns=["id"])
    float_distributions = get_float_distributions(n)
    for col_name in list(data.columns.drop("id")):
        generated_col = generate_column(data, col_name, n+1, float_distributions)
        synthetic_data[col_name] = generated_col
    synthetic_data_train, synthetic_data_test = train_test_split(synthetic_data,
                                                                 test_size=0.2,
                                                                 random_state=2022)
    synthetic_data_test = synthetic_data_test.drop("stroke", 1)
    with open("tests/synthetic_data_train.csv", "wb") as f:
        synthetic_data_train.to_csv(f)
    with open("tests/synthetic_data_test.csv", "wb") as f:
        synthetic_data_test.to_csv(f)
    return synthetic_data_train, synthetic_data_test


@click.command(name="generate_data")
@click.argument("n", type=int)
def generate_data_command(n):
    click.echo("started...")
    data = pd.read_csv("data/raw/train_df.csv")
    generate_synthetic_data(data, n)


if __name__ == "__main__":
    generate_data_command()
