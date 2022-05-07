import pytest


@pytest.fixture()
def dataset_path():
    return "tests/data_test.csv"


@pytest.fixture()
def target_col():
    return "stroke"


@pytest.fixture()
def categorical_features():
    return [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]


@pytest.fixture
def numerical_features():
    return [
        "id",
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi"
    ]