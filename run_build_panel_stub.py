from pathlib import Path
import pandas as pd

from piyrm.data.sources.usda_nass import NassConfig, load_or_stub_nass_yield
from piyrm.data.sources.climate import ClimateConfig, load_or_stub_climate_panel


def main():
    out_path = Path("data/processed/panel_county_year.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data sources
    nass = load_or_stub_nass_yield(NassConfig())
    climate = load_or_stub_climate_panel(ClimateConfig())

    # Planned merge keys
    keys = ["year", "county_fips","crop"]

    # Final panel schema (fixed order)
    panel_cols = [
        "year",
        "state_fips",
        "county_fips",
        "crop",
        "yield_value",
        "yield_unit",
        "precip_mm",
        "gdd",
        "heat_days",
        "drought_index",
]

    # ---- Build panel ----
    if len(nass) == 0:
        # No yield data available
        panel = pd.DataFrame(columns=panel_cols)

    elif len(climate) == 0:
        # Yield-only panel (climate columns filled with NA)
        panel = nass.copy()
        for col in ["precip_mm", "gdd", "heat_days", "drought_index"]:
            panel[col] = pd.NA

        # Ensure all expected columns exist
        for col in panel_cols:
            if col not in panel.columns:
                panel[col] = pd.NA

        panel = panel.loc[:, panel_cols]

    else:
        # Full merge
        panel = nass.merge(climate, on=keys, how="left")

        # Ensure all expected columns exist
        for col in panel_cols:
            if col not in panel.columns:
                panel[col] = pd.NA

        panel = panel.loc[:, panel_cols]

    # Write output
    panel.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  (rows={len(panel)})")


if __name__ == "__main__":
    main()