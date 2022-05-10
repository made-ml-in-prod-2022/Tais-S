import logging
import sys
import json
import click
from imblearn.over_sampling import SMOTE

from ml_project.entities.train_pipeline_params import read_training_pipeline_params
from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.features.build_features import extract_target, build_transformer
from ml_project.features import make_features
from ml_project.models.model_fit import train_model, evaluate_model, serialize_model, serialize_transformer
from ml_project.models.model_predict import create_inference_pipeline, predict_model

# from entities.train_pipeline_params import read_training_pipeline_params
# from data.make_dataset import read_data, split_train_val_data
# from features.build_features import extract_target, build_transformer
# from features import make_features
# from models.model_fit import train_model, evaluate_model, serialize_model, serialize_transformer
# from models.model_predict import create_inference_pipeline, predict_model


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path):
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    logger.info(f"===== start of train pipeline with params {training_pipeline_params}")
    df = read_data(training_pipeline_params.train_data_path)
    logger.info(f"===== loaded data")
    train_df, val_df = split_train_val_data(df, training_pipeline_params.splitting_params)
    logger.info(f"===== split data into {train_df.shape} and {val_df.shape}")

    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    train_df = train_df.drop(training_pipeline_params.feature_params.features_to_drop, 1)

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.features_to_drop, 1)

    transformer = build_transformer(training_pipeline_params.feature_params)

    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    train_features, train_target = SMOTE().fit_resample(train_features, train_target)
    path_to_transformer = serialize_transformer(transformer, training_pipeline_params.transformer_path)
    logger.info(f"===== transformed train data {train_features.shape}")

    model = train_model(train_features, train_target, training_pipeline_params.train_params)
    logger.info(f"===== trained model {model}")

    inference_pipeline = create_inference_pipeline(model, transformer)
    predictions = predict_model(inference_pipeline, val_df)
    logger.info(f"===== made validation predictions")

    metrics = evaluate_model(predictions, val_target)
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"===== model performance on validation data: {metrics}")

    path_to_model = serialize_model(inference_pipeline, training_pipeline_params.output_model_path)

    return path_to_model, metrics, path_to_transformer


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path):
    click.echo("started...")
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()