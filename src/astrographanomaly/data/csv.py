import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize columns
    if "distance" not in df.columns and "parallax" in df.columns:
        df["distance"] = 1000.0 / df["parallax"]
    return df.dropna()
