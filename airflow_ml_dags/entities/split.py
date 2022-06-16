import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("validate")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    train_data, val_data, train_target, val_target = train_test_split(data,
                                                                      target,
                                                                      test_size=0.2,
                                                                      random_state=2022)

    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"))
    val_data.to_csv(os.path.join(output_dir, "val_data.csv"))
    train_target.to_csv(os.path.join(output_dir, "train_target.csv"))
    val_target.to_csv(os.path.join(output_dir, "val_target.csv"))


if __name__ == '__main__':
    split()
