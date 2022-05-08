import pytest
import pandas as pd

from ml_project.entities.feature_params import FeatureParams
from ml_project.data.make_dataset import read_data
from ml_project.features.build_features import make_features, build_transformer


@pytest.fixture
def feature_params(numerical_features, categorical_features, features_to_drop, target_col):
    feature_params = FeatureParams(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        features_to_drop=features_to_drop,
        target_col=target_col
    )
    return feature_params


def test_make_features(feature_params, dataset_path):
    df = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(df)
    features = make_features(transformer, df)
    assert len(transformer.get_params()["transformers"]) == 2
    assert not pd.isnull(features).any().any()
