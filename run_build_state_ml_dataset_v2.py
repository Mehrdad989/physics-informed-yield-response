import pandas as pd
import numpy as np


def prep(params_path: str, crop: str, feats: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(params_path, dtype={"state_fips": str})
    p["crop"] = crop

    df = p.merge(feats, on=["state_fips", "crop"], how="left", suffixes=("_param", "_feat"))

    # targets
    df["log_ymax"] = np.log(df["ymax"])
    df["log_k"] = np.log(df["k"])

    # Use feature-engineered precip columns (the _feat ones)
    # These come from state_features_from_panel.csv
    df["log_precip_mean"] = np.log(df["precip_mean"])
    df["log_precip_med"] = np.log(df["precip_med_feat"])

    # Make sure downstream scripts can use consistent names
    df["precip_med"] = df["precip_med_feat"]

    return df


def main():
    feats = pd.read_csv("data/processed/state_features_from_panel.csv", dtype={"state_fips": str})

    maize = prep("data/processed/maize_state_saturating2_params.csv", "maize", feats)
    wheat = prep("data/processed/wheat_state_saturating2_params.csv", "wheat", feats)

    maize.to_csv("data/processed/ml_state_maize_v2.csv", index=False)
    wheat.to_csv("data/processed/ml_state_wheat_v2.csv", index=False)

    print("Maize rows:", len(maize), "missing feat rows:", maize["precip_mean"].isna().sum())
    print("Wheat rows:", len(wheat), "missing feat rows:", wheat["precip_mean"].isna().sum())


if __name__ == "__main__":
    main()