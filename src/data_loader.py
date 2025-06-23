# data_loader.py

import pandas as pd

def load_data(path="data/lending_club_clean.csv"):
    """
    Loads cleaned Lending Club dataset with numeric 'term' and 'emp_length'.
    """
    df = pd.read_csv(path)

    # These two are usually already numeric, but this is a fallback:
    if df["term"].dtype == object:
        df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
    if df["emp_length"].dtype == object:
        df["emp_length"] = df["emp_length"].str.extract(r"(\d+)").fillna(0).astype(int)

    return df
