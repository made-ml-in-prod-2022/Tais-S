import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
            ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scale", StandardScaler())
        ]
    )
    return numerical_pipeline


def build_transformer(feature_params):
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                feature_params.categorical_features
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                feature_params.numerical_features
            ),
        ]
    )
    return transformer


def make_features(transformer, df):
    return transformer.transform(df)


def extract_target(df, feature_params):
    return df[feature_params.target_col]