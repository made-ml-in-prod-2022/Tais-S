import os
import pandas as pd
import numpy as np
import click
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_categorical_pipeline():
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline():
    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scale", StandardScaler())
        ]
    )
    return numerical_pipeline


def build_transformer():
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
            ),
        ]
    )
    return transformer


@click.command("validate")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--transformer-dir")
def preprocess(input_dir, output_dir, transformer_dir):
    train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    train_target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))
    train_data = train_data.drop("id", 1)

    transformer = build_transformer()

    transformer.fit(train_data)
    train_features = transformer.transform(train_data)
    train_features, train_target = SMOTE().fit_resample(train_features, train_target)
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "transformer.pkl"), "wb") as f:
        pickle.dump(transformer, f)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(train_features).to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
    pd.DataFrame(train_target).to_csv(os.path.join(output_dir, "train_target.csv"), index=False)


if __name__ == '__main__':
    preprocess()
