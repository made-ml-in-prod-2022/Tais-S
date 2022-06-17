import os
import pickle
import pandas as pd
from sklearn.svm import SVC
import click


@click.command("validate")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir, output_dir):
    train_data = pd.read_csv(os.path.join(input_dir, "train_features.csv"))
    train_features = pd.read_csv(os.path.join(input_dir, "train_target.csv"))
    model = SVC(kernel="SVC",
                gamma="sigmoid",
                C=100)
    model.fit(train_data, train_features)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
