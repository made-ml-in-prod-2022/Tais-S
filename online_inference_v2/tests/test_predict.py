from fastapi.testclient import TestClient

from online_inference_v2.app import app


def test_predict():
    with TestClient(app) as client:
        response = client.get('/predict',
                              json={"data": [['Female', 72.0, 0, 1, 'No', 'Self-employed',
                                              'Rural', 124.38, 23.4, 'formerly smoked']],
                                    "features": ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                                                 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                                                 'smoking_status']})

        assert response.status_code == 200
        assert response.json() == [{'stroke': 1}]
