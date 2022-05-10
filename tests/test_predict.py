from ml_project.predict_pipeline import run_predict_pipeline
from ml_project.entities import PredictPipelineParams, FeatureParams


def test_predict(tmpdir, dataset_path_test, model_path, transformer_path,
                 target_col, numerical_features, categorical_features, features_to_drop):
    expected_predictions_path = tmpdir.join("predictions.csv")
    params = PredictPipelineParams(
        test_data_path=dataset_path_test,
        predictions_path=expected_predictions_path,
        output_model_path=model_path,
        transformer_path=transformer_path,
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop
        )
    )
    predictions = run_predict_pipeline(params)
    assert len(predictions) > 1
