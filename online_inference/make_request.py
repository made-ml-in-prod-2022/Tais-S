import numpy as np
import pandas as pd
import yaml
import requests


if __name__ == "__main__":
    path_to_config = "configs/config.yaml"
    with open(path_to_config, "r") as input_stream:
        path_to_data = yaml.safe_load(input_stream)["data_path"]
    data = pd.read_csv(path_to_data).fillna("nan")

    request_features = list(data.columns)
    for i in range(len(data)):
        request_data = [x.item() if isinstance(x, np.generic) else x for x in data.loc[i]]
        print(request_data)
        response = requests.get("http://localhost:8000/predict/",
                                json={"data": [request_data], "features": request_features})
        print(response.status_code)
        print(response.json())
