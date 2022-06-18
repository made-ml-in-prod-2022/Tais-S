import click


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


@click.command("generate")
@click.option("--input-dir")
@click.option("--output-dir")
def generate(input_dir, output_dir, n_rows=500):
    import pandas as pd
    import os
    data = pd.read_csv(os.path.join(input_dir, "initial_data.csv"))
    id_col = [x for x in range(n_rows)]
    synthetic_data = pd.DataFrame(id_col, columns=["id"])
    float_distributions = get_float_distributions(n_rows)
    for col_name in list(data.columns.drop("id")):
        generated_col = generate_column(data, col_name, n_rows+1, float_distributions)
        synthetic_data[col_name] = generated_col

    synthetic_target = synthetic_data["stroke"]
    synthetic_data = synthetic_data.drop("stroke", 1)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "data.csv"), "wb") as f:
        synthetic_data.to_csv(f, index=False)
    with open(os.path.join(output_dir, "target.csv"), "wb") as f:
        synthetic_target.to_csv(f, index=False)


if __name__ == '__main__':
    generate()
