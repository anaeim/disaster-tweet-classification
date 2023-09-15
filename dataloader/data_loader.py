from pathlib import Path
import pandas as pd

def load():
    """load the data from the data_path provided on config.py

    Returns
    -------
    pandas.DataFrame
    """
    data_path = Path().cwd() / "data" / "train.csv"
    # print(data_path)
    df = pd.read_csv(data_path)

    return df