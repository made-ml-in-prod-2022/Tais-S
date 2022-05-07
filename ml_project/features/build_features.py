import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


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
            ("impute", SimpleImputer(missing_values=np.nan, strategy="median"))
        ]
    )
    return numerical_pipeline


def process_categorical_features(categorical_df):
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def process_numerical_features(numerical_df):
    numerical_pipeline = build_numerical_pipeline()
    return pd.DataFrame(numerical_pipeline.fit_transform(numerical_df))


def build_transformer(feature_params):
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                feature_params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                feature_params.numerical_features,
            ),
        ]
    )
    return transformer


def make_features(transformer, df):
    return transformer.transform(df)


def extract_target(df, feature_params):
    return df[feature_params.target_col]