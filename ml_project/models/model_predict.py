from sklearn.pipeline import Pipeline


def predict_model(model, features):
    predictions = model.predict(features)
    return predictions


def create_inference_pipeline(model, transformer):
    return Pipeline([("features_part", transformer), ("model_part", model)])