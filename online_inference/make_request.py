import numpy as np
import pandas as pd
import requests


if __name__ == "__main__":
    data = pd.read_csv("data/data.csv").fillna("nan")
    data = data.drop("id", 1)
    request_features = list(data.columns)
    for i in range(len(data)):
        request_data = [x.item() if isinstance(x, np.generic) else x for x in data.loc[i]]
        print(request_data)
        response = requests.get("http://localhost:8000/predict/",
                                json={"data": [request_data], "features": request_features})
        print(response.status_code)
        print(response.json())
