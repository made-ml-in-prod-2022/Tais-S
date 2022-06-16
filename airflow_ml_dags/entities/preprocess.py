import os
import pandas as pd
import click
import pickle


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir, output_dir, transformer_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    data = data.drop("id", 1)
    train_features = make_features(transformer, train_df)
    train_features, train_target = SMOTE().fit_resample(train_features, train_target)
    with open(transformer_dir, "rb") as f:
        transformer = pickle.load(f)

    transformer.transform(data)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"))


if __name__ == '__main__':
    preprocess()
