from fastapi.testclient import TestClient
import pickle

from online_inference.app import app

client = TestClient(app)


def test_predict():
    global model
    with open("online_inference/models/model.pkl", "rb") as f:
        model = pickle.load(f)
    print(model)

    response = client.get('/predict',
                          json={"data": [['Female', 72.0, 0, 1, 'No', 'Self-employed',
                                          'Rural', 124.38, 23.4, 'formerly smoked']],
                                "features": ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                                             'work_type','Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']})

    assert response.status_code == 200
    assert response.json() == [{'stroke': 1}]
