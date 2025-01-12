import pandas as pd
import os

def load_train_data():
    train_path = os.path.abspath("../Data/train.csv")
    return pd.read_csv(train_path)

def load_test_data():
    test_path = os.path.abspath("../Data/test.csv")
    return pd.read_csv(test_path)
