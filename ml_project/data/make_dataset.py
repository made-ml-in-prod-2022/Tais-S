import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path):
    data = pd.read_csv(path)
    return data


def split_train_val_data(df, splitting_params):
    train_df, val_df = train_test_split(df,
                                        test_size=splitting_params.val_size,
                                        random_state=splitting_params.random_state)
    return train_df, val_df
