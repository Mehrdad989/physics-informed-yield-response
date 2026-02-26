import pandas as pd
import numpy as np


def main():
    df = pd.read_csv(
        "data/processed/panel_county_year.csv",
        dtype={"state_fips": str, "county_fips": str, "crop": str},
    )

    # keep positive yields only (consistent with curve fitting)
    df = df[df["yield_value"] > 0].copy()

    # Aggregate county-year -> state-year (so a state with many counties isn't overweighted)
    state_year = (
        df.groupby(["state_fips", "crop", "year"], as_index=False)
        .agg(
            precip_mm_mean=("precip_mm", "mean"),
            yield_mean=("yield_value", "mean"),
        )
    )

    # Then aggregate across years -> state (features)
    def cv(x):
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return s / m if m > 0 else np.nan

    feats = (
        state_year.groupby(["state_fips", "crop"], as_index=False)
        .agg(
            n_years=("year", "nunique"),
            precip_mean=("precip_mm_mean", "mean"),
            precip_std=("precip_mm_mean", "std"),
            precip_med=("precip_mm_mean", "median"),
            yield_mean=("yield_mean", "mean"),
            yield_std=("yield_mean", "std"),
        )
    )

    feats["precip_cv"] = feats.apply(lambda r: (r["precip_std"] / r["precip_mean"]) if r["precip_mean"] > 0 else np.nan, axis=1)
    feats["yield_cv"] = feats.apply(lambda r: (r["yield_std"] / r["yield_mean"]) if r["yield_mean"] > 0 else np.nan, axis=1)

    out_path = "data/processed/state_features_from_panel.csv"
    feats.to_csv(out_path, index=False)

    print(f"Wrote: {out_path} (rows={len(feats)})")
    print(feats.head(10))


if __name__ == "__main__":
    main()