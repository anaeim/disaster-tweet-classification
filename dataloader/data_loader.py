from pathlib import Path
import pandas as pd

def load():
    data_path = Path().cwd() / "data" / "train.csv"
    # print(data_path)
    df = pd.read_csv(data_path)

    return df