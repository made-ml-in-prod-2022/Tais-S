import logging
import sys
import click
import pickle
import numpy as np

from ml_project.entities.predict_pipeline_params import read_predict_pipeline_params
from ml_project.data.make_dataset import read_data
from ml_project.models.model_predict import create_inference_pipeline, predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path):
    predict_pipeline_params = read_predict_pipeline_params(config_path)
    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params):
    with open(predict_pipeline_params.output_model_path, "rb") as f:
        model = pickle.load(f)
    with open(predict_pipeline_params.transformer_path, "rb") as f:
        transformer = pickle.load(f)
    test_df = read_data(predict_pipeline_params.test_data_path)
    test_df = test_df.drop(predict_pipeline_params.feature_params.features_to_drop, 1)
    logger.info(f"===== test columns {test_df.columns}")
    inference_pipeline = create_inference_pipeline(model, transformer)
    predictions = predict_model(model, test_df)
    logger.info(f"===== made predictions")
    with open(predict_pipeline_params.predictions_path, "wb") as predictions_file:
        np.savetxt(predictions_file, predictions, delimiter=",")

    return predictions


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path):
    click.echo("started...")
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()