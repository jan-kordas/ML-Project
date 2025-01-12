import pandas as pd

def load_train_data(path="data/train.csv"):
    train_df = pd.read_csv(path)
    return train_df

def load_test_data(path="data/test.csv"):
    test_df = pd.read_csv(path)
    return test_df
