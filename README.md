# Stroke Prediction 

The project solves a classification task based on https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?datasetId=1120859 dataset.

Installation: 
~~~
env\Scripts\activate.bat
pip install -r requirements.txt
~~~
Usage:
~~~
python ml_project/train_pipeline.py configs/train_config.yaml
python ml_project/predict_pipeline.py configs/predict_config.yaml
~~~

Tests:
~~~
pytest tests/
~~~
Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so it can be imported
    ├── ml_project         <- Source code for use in this project
    │   ├── __init__.py    <- Makes ml_project a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   └── models         <- code to train models and then use trained models to make
    │       │                 predictions
    │       ├── model_fit.py
    │       └── model_predict.py
    │
    ├── configs            <- Yaml files with different configurations
    │
    └── tests              <- Testing different modules
        ├── data           <- code to test making data
        │
        ├── features       <- code to test getting features ready for modeling
        │
        └── models         <- code to test training model