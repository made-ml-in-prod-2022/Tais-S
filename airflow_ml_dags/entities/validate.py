import os
import pickle
import pandas as pd
import click
import json
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(predictions, target):
    acc_score = accuracy_score(target, predictions)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    false_negative_rate = fn / (tp + fn)
    return {"accuracy": acc_score,
            "false_negative_rate": false_negative_rate}


@click.command("validate")
@click.option("--input-dir")
@click.option("--output-dir")
def validate(input_dir, output_dir):
    with open(os.path.join(output_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(output_dir, "transformer.pkl"), "rb") as f:
        transformer = pickle.load(f)

    val_data = pd.read_csv(os.path.join(input_dir, "val_data.csv"))
    val_target = pd.read_csv(os.path.join(input_dir, "val_target.csv"))
    val_data = val_data.drop("id", 1)
    val_features = transformer.transform(val_data)

    predictions = model.predict(val_features)

    metrics = evaluate_model(predictions, val_target)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == '__main__':
    validate()
