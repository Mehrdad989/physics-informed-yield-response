from pathlib import Path
import pandas as pd

from piyrm.data.sources.usda_nass import NassConfig, load_or_stub_nass_yield
from piyrm.data.sources.climate import ClimateConfig, load_or_stub_climate_panel


def main():
    out_path = Path("data/processed/panel_county_year.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nass = load_or_stub_nass_yield(NassConfig())
    climate = load_or_stub_climate_panel(ClimateConfig())

    # Planned merge keys:
    keys = ["year", "state_fips", "county_fips"]

    if len(nass) == 0 or len(climate) == 0:
        # For now we just write an empty file with expected columns
        panel = pd.DataFrame(columns=keys + ["crop", "yield_value", "yield_unit", "precip_mm", "gdd", "heat_days", "drought_index"])
    else:
        panel = nass.merge(climate, on=keys, how="left")

    panel.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  (rows={len(panel)})")


if __name__ == "__main__":
    main()