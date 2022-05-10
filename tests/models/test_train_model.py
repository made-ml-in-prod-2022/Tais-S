import pytest

from ml_project.data.make_dataset import read_data
from ml_project.entities.feature_params import FeatureParams
from ml_project.features.build_features import make_features, extract_target, build_transformer
from ml_project.models.model_fit import train_model
from ml_project.entities import TrainingParams
from sklearn.svm import SVC


@pytest.fixture
def features_and_target(dataset_path, categorical_features, numerical_features, target_col, features_to_drop):
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, SVC)
