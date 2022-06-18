# Практика применения Airflow

Реализованы dag-и периодических генерации данных, обучения модели и создания предсказаний. 
Исходные данные https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?datasetId=1120859

~~~
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
~~~