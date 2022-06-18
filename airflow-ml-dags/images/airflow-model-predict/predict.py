import os
import pickle
import pandas as pd
import numpy as np
import click


@click.command("predict")
@click.option("--data-dir")
@click.option("--output-dir")
def predict(data_dir, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    path = os.getenv("path_to_model")
    print("===================path:", path)
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(path, "transformer.pkl"), "rb") as f:
        transformer = pickle.load(f)
    data = data.drop("id", 1)
    features = transformer.transform(data)

    predictions = model.predict(features)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions.csv"), "wb") as predictions_file:
        np.savetxt(predictions_file, predictions, delimiter=",")


if __name__ == '__main__':
    predict()
