import pandas as pd
from pathlib import Path

def load_ctr_data(path: str | Path):
    """
    Load CTR dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    n_arms = df.shape[1]  # number of columns = number of arms

    return df, n_arms
