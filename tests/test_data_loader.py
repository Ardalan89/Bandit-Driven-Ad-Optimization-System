import pandas as pd
from bandit_ad_opt.data_loader import load_ctr_data

def test_data_loader_wide(tmp_path):
    # Create synthetic bandit data
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "Ad 1": [1, 0, 0],
        "Ad 2": [0, 1, 0],
        "Ad 3": [0, 0, 1],
    })
    df.to_csv(csv_path, index=False)

    loaded_df, n_arms = load_ctr_data(csv_path)

    assert n_arms == 3            # 3 arms = 3 columns
    assert loaded_df.shape == (3, 3)  # 3 rows, 3 cols
    assert list(loaded_df.columns) == ["Ad 1", "Ad 2", "Ad 3"]
