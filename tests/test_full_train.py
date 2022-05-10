import os

from ml_project.train_pipeline import run_train_pipeline
from ml_project.entities import TrainingPipelineParams, SplittingParams, FeatureParams, TrainingParams


def test_full_train(tmpdir, dataset_path, numerical_features, categorical_features, target_col, features_to_drop):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_transformer_path = tmpdir.join("transformer.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        train_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        transformer_path=expected_transformer_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop
        ),
        train_params=TrainingParams()
    )
    model_path, metrics, path_to_transformer = run_train_pipeline(params)
    assert metrics["accuracy"] > 0
    assert metrics["false_negative_rate"] > 0
    assert os.path.exists(model_path)
    assert os.path.exists(params.metric_path)