from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline


def train_model(features, target, train_params):
    model = KNeighborsClassifier(n_neighbors=train_params.n_neighbors)
    model.fit(features, target)
    return model


def predict_model(model, features):
    predictions = model.predict(features)
    return predictions


def evaluate_model(predictions, target):
    acc_score = accuracy_score(target, predictions)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    false_negative_rate = fn / (tp + fn)
    return {"accuracy": acc_score,
            "false_negative_rate": false_negative_rate}


def create_inference_pipeline(model, transformer):
    return Pipeline([("features_part", transformer), ("model_part", model)])
