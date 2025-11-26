import pandas as pd
from src.bandit_ad_opt.data_loader import load_ctr_data

def test_data_loader_basic(tmp_path):
    # Create a test CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "Ad": ["A", "A", "B"],
        "Click": [1, 0, 1]
    })
    df.to_csv(csv_path, index=False)

    loaded, n_arms = load_ctr_data(csv_path, "Ad", "Click")
    assert n_arms == 2
    assert set(loaded.columns) == {"arm", "reward"}
