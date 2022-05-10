# Stroke Prediction 

The project solves a classification task based on https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?datasetId=1120859 dataset.

Installation: 
~~~
py -m venv env
env\Scripts\activate.bat
pip install -r requirements.txt
~~~
Usage: (to run the commands one needs to switch to relative imports in the corresponding files - train_pipeline.py 
and predict_pipeline.py - i.e. without 'ml_project.')
~~~
python ml_project/train_pipeline.py configs/train_config.yaml
python ml_project/predict_pipeline.py configs/predict_config.yaml
~~~

Tests: (to run the tests one needs to get the imports in the corresponding files - train_pipeline.py 
and predict_pipeline.py - back to starting with 'ml_project.')
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
    ├── .github            <- Jupyter notebooks
    │   └── workflows      <- yaml file with CI configuration
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so it can be imported
    ├── ml_project         <- Source code for use in this project
    │   ├── __init__.py    <- Makes ml_project a Python module
    │   │
    │   ├── data           <- code to read data
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