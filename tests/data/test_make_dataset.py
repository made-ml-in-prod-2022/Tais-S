from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.entities import SplittingParams


def test_load_dataset(dataset_path, target_col):
    df = read_data(dataset_path)
    assert len(df) > 1
    assert target_col in df.keys()


def test_split_dataset(dataset_path):
    val_size = 0.2
    splitting_params = SplittingParams(val_size=val_size, random_state=2022)
    df = read_data(dataset_path)
    train_df, val_df = split_train_val_data(df, splitting_params)
    assert train_df.shape[0] > 1
    assert val_df.shape[0] > 1
