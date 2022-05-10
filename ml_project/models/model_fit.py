from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle


def train_model(features, target, train_params):
    if train_params.model_type == "SVC":
        model = SVC(kernel=train_params.kernel_type,
                    gamma=train_params.gamma,
                    C=train_params.inverted_regularization)
    elif train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(n_neighbors=train_params.n_neighbors)
    else:
        raise NotImplementedError()
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


def serialize_transformer(transformer, transformer_output):
    with open(transformer_output, "wb") as f:
        pickle.dump(transformer, f)
    return transformer_output


def serialize_model(model, model_output):
    with open(model_output, "wb") as f:
        pickle.dump(model, f)
    return model_output