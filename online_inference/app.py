import logging
import os
import pickle
import pandas as pd
from imblearn.pipeline import Pipeline
from pydantic import BaseModel, conlist
from typing import List, Union, Optional
import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class StrokeModel(BaseModel):
    data: List[conlist(Union[float, str, None])] # , min_items=10, max_items=11
    features: List[str]


class StrokeResponse(BaseModel):
    # id: int
    stroke: int


model: Optional[Pipeline] = None


def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_predict(data, features, model):
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)
    return [StrokeResponse(stroke=int(stroke)) for stroke in predicts]


app = FastAPI()


@app.get("/")
def root():
    return "Predictor entry point"


@app.on_event("startup")
def load_model():
    global model
    path_to_model = "online_inference/models/model.pkl"
    # path_to_model = os.getenv("PATH_TO_MODEL")
    if path_to_model is None:
        # err = f"PATH_TO_MODEL {path_to_model} is None"
        path_to_model = "online_inference/models/model.pkl"
        # logger.error(err)
        # raise RuntimeError(err)

    model = load_object(path_to_model)


@app.get("/healz")
def health():
    if model is None:
        return "Loading"
    else:
        return 200


@app.get("/predict", response_model=List[StrokeResponse])
def predict(request: StrokeModel):
    print('req', request)
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
