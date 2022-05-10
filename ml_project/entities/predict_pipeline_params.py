from dataclasses import dataclass

from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    test_data_path: str
    predictions_path: str
    output_model_path: str
    transformer_path: str
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path):
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
