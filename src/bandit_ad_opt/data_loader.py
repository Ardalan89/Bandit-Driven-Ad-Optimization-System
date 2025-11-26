import pandas as pd
from pathlib import Path

def load_ctr_data(path: str | Path, ad_column: str, click_column: str):
    """
    Load CTR dataset and return a DataFrame with required columns.
    
    Parameters:
    - path: path to CSV dataset
    - ad_column: column containing ad IDs (arm labels)
    - click_column: binary reward (0 or 1)

    Returns:
    - df: DataFrame with 'arm' (int) and 'reward' (0 or 1)
    - n_arms: number of unique arms
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Map ad values to integer arm IDs
    unique_ads = df[ad_column].unique()
    ad_to_arm = {ad: idx for idx, ad in enumerate(unique_ads)}

    df["arm"] = df[ad_column].map(ad_to_arm)
    df["reward"] = df[click_column].astype(int)

    n_arms = len(unique_ads)
    return df[["arm", "reward"]], n_arms
